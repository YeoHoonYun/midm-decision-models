#!/usr/bin/env python3
"""GPU job queue: one sequential worker per GPU, built on Huey (SQLite storage, no broker).

Celery is not supported on Windows and needs a broker; Huey's SqliteHuey runs here with a
thread worker. Each queue has exactly one worker, so jobs on a GPU run one after another.
Before a job starts, the worker waits until the GPU reports at least --mem GiB free, so it
does not collide with other processes on the host (e.g. another experiment's Ollama).

    python gpuq.py workers start                 # detached consumers for gpu0, gpu1, cpu
    python gpuq.py submit --queue gpu1 --mem 7 --name ptr35-pilot -- python pointer_lora.py train ...
    python gpuq.py status [--all]
    python gpuq.py cancel 12                     # queued jobs only; running jobs keep running
    python gpuq.py workers stop

Queues: gpu0 = physical GPU 0 (RTX 3090), gpu1 = physical GPU 1 (RTX 2080 Ti), cpu = no GPU.
Job rows (runs/gpuq/jobs.db): id, queue, name, cmd, mem, state, times, rc, log path.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import time

from huey import SqliteHuey

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = os.path.join(HERE, "runs", "gpuq")
os.makedirs(os.path.join(QDIR, "logs"), exist_ok=True)
JOBS_DB = os.path.join(QDIR, "jobs.db")
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
GPU_INDEX = {"gpu0": "0", "gpu1": "1", "cpu": ""}

HUEY_GPU0 = SqliteHuey("gpu0", filename=os.path.join(QDIR, "huey_gpu0.db"), results=False)
HUEY_GPU1 = SqliteHuey("gpu1", filename=os.path.join(QDIR, "huey_gpu1.db"), results=False)
HUEY_CPU = SqliteHuey("cpu", filename=os.path.join(QDIR, "huey_cpu.db"), results=False)
HUEYS = {"gpu0": HUEY_GPU0, "gpu1": HUEY_GPU1, "cpu": HUEY_CPU}


def db():
    c = sqlite3.connect(JOBS_DB, timeout=30)
    c.execute("""create table if not exists jobs (id integer primary key autoincrement, queue text, name text,
                 cmd text, mem real, state text, submitted text, started text, finished text, rc integer,
                 log text, note text)""")
    cols = {r[1] for r in c.execute("pragma table_info(jobs)")}
    for col, typ in (("cwd", "text"), ("env", "text"), ("max_util", "real"), ("owner", "text"), ("pid", "integer"),
                     ("priority", "integer"), ("retries", "integer"), ("attempts", "integer")):
        if col not in cols:
            c.execute(f"alter table jobs add column {col} {typ}")
    return c


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def set_job(job_id, **kw):
    with db() as c:
        c.execute(f"update jobs set {', '.join(f'{k}=?' for k in kw)} where id=?", (*kw.values(), job_id))


def gpu_state(index: str) -> tuple[float, float]:
    """-> (free GiB, utilisation %) of a physical GPU."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits",
                          "-i", index], capture_output=True, text=True).stdout.strip()
    if not out:
        return 0.0, 100.0
    free, util = (float(v) for v in out.split(","))
    return free / 1024, util


def pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except ImportError:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out


def reconcile():
    """Mark running jobs whose process is gone (driver reset, killed worker) as 'lost'."""
    with db() as c:
        rows = c.execute("select id, pid from jobs where state='running'").fetchall()
    for job_id, pid in rows:
        if pid and not pid_alive(pid):
            set_job(job_id, state="lost", finished=now(), note="process gone; see log / outputs")


def wait_gpu(job_id, idx, mem, max_util):
    """True once the GPU had enough free VRAM and low utilisation for 3 polls in a row."""
    ok = 0
    while ok < 3:
        free, util = gpu_state(idx)
        ok = ok + 1 if free >= (mem or 0) and util <= (max_util if max_util is not None else 50) else 0
        if ok < 3:
            time.sleep(30)
        with db() as c:
            if c.execute("select state from jobs where id=?", (job_id,)).fetchone()[0] == "cancelled":
                return False
    return True


def execute(job_id: int):
    with db() as c:
        row = c.execute("select queue, name, cmd, mem, state, cwd, env, max_util, retries from jobs where id=?",
                        (job_id,)).fetchone()
    if row is None or row[4] != "queued":   # cancelled, or a duplicate message for a job already taken
        return
    queue, name, cmd, mem, _, cwd, env_json, max_util, retries = row
    idx = GPU_INDEX[queue]
    log = os.path.join(QDIR, "logs", f"{job_id:04d}_{name}.log")
    extra = json.loads(env_json or "{}")
    keep = extra.pop("__keep_env__", False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=idx, PYTHONUNBUFFERED="1")
    if not keep:   # default for this experiment: isolated from the global PYTHONPATH
        env.update(PYTHONPATH="", HF_HOME=os.path.join(HERE, "hf_cache"), HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    env.update(extra)
    argv = json.loads(cmd)
    if argv and argv[0] == "python" and not keep:
        argv[0] = PY
    t0, rc = time.time(), None
    for attempt in range(1, (retries or 0) + 2):   # retries: GPU driver resets (nvlddmkm 153) kill CUDA jobs
        if idx:
            set_job(job_id, state="waiting_gpu", attempts=attempt)
            if not wait_gpu(job_id, idx, mem, max_util):
                return
        with open(log, "a" if attempt > 1 else "w", encoding="utf-8") as f:
            if attempt > 1:
                f.write(f"\n===== gpuq retry attempt {attempt} (previous rc={rc}) {now()} =====\n")
                f.flush()
            p = subprocess.Popen(argv, cwd=cwd or HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
            set_job(job_id, state="running", started=now(), log=log, pid=p.pid, attempts=attempt)
            rc = p.wait()
        if rc == 0:
            break
    set_job(job_id, state="done" if rc == 0 else "failed", finished=now(), rc=rc,
            note=f"{(time.time() - t0) / 60:.1f} min, {attempt} attempt(s)")


@HUEY_GPU0.task()
def run_gpu0(job_id):
    execute(job_id)


@HUEY_GPU1.task()
def run_gpu1(job_id):
    execute(job_id)


@HUEY_CPU.task()
def run_cpu(job_id):
    execute(job_id)


TASKS = {"gpu0": run_gpu0, "gpu1": run_gpu1, "cpu": run_cpu}


def submit(queue, name, cmd, mem, cwd=None, env=None, max_util=None, owner=None, priority=0, retries=0):
    with db() as c:
        cur = c.execute("insert into jobs (queue, name, cmd, mem, state, submitted, cwd, env, max_util, owner, "
                        "priority, retries) values (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (queue, name, json.dumps(cmd), mem, "queued", now(), cwd, json.dumps(env or {}), max_util,
                         owner, priority, retries))
        job_id = cur.lastrowid
    TASKS[queue](job_id, priority=priority)   # higher priority is dequeued first
    return job_id


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--queue", choices=list(HUEYS), required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--mem", type=float, default=0.0, help="GiB that must be free on the GPU before start")
    s.add_argument("--max-util", type=float, default=None, help="GPU utilisation %% that counts as idle (default 50)")
    s.add_argument("--cwd", default=None, help="working directory (default: this experiment)")
    s.add_argument("--keep-env", action="store_true",
                   help="keep the caller's environment (PYTHONPATH etc.) and run argv[0] as given")
    s.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    s.add_argument("--owner", default=None, help="session / project that submitted the job")
    s.add_argument("--priority", type=int, default=0, help="higher runs first among queued jobs (default 0)")
    s.add_argument("--retries", type=int, default=1, help="re-run after a non-zero exit (default 1)")
    s.add_argument("command", nargs=argparse.REMAINDER)
    ad = sub.add_parser("adopt", help="attach a running process (pid) to a job row, e.g. after a worker restart")
    ad.add_argument("id", type=int)
    ad.add_argument("pid", type=int)
    st = sub.add_parser("status")
    st.add_argument("--all", action="store_true")
    ca = sub.add_parser("cancel")
    ca.add_argument("ids", type=int, nargs="+")
    rq = sub.add_parser("requeue", help="re-send queued / lost / failed jobs")
    rq.add_argument("ids", type=int, nargs="+")
    w = sub.add_parser("workers")
    w.add_argument("action", choices=["start", "stop", "list"])
    a = ap.parse_args()

    if a.cmd == "submit":
        cmd = a.command[1:] if a.command and a.command[0] == "--" else a.command
        env = dict(kv.split("=", 1) for kv in a.env)
        if a.keep_env:
            env["__keep_env__"] = True
        job = submit(a.queue, a.name, cmd, a.mem, a.cwd, env, a.max_util, a.owner, a.priority, a.retries)
        print(f"job {job} queued on {a.queue}")
    elif a.cmd == "adopt":
        set_job(a.id, state="running", pid=a.pid)
        print(f"job {a.id} now tracks pid {a.pid}")
    elif a.cmd == "status":
        reconcile()
        with db() as c:
            q = "select id, queue, owner, name, state, submitted, started, finished, rc, note from jobs"
            rows = c.execute(q + ("" if a.all else " where state not in ('done','cancelled')") + " order by id").fetchall()
        for r in rows:
            print(" | ".join("" if v is None else str(v) for v in r))
    elif a.cmd == "cancel":
        for i in a.ids:
            with db() as c:
                n = c.execute("update jobs set state='cancelled' where id=? and state in ('queued','waiting_gpu')",
                              (i,)).rowcount
            print(f"job {i}: {'cancelled' if n else 'not cancellable (running or finished)'}")
    elif a.cmd == "requeue":
        for i in a.ids:
            with db() as c:
                r = c.execute("select queue, state from jobs where id=?", (i,)).fetchone()
            if r and r[1] in ("queued", "lost", "failed", "cancelled"):
                with db() as c:
                    prio = c.execute("select coalesce(priority,0) from jobs where id=?", (i,)).fetchone()[0]
                set_job(i, state="queued", pid=None)
                TASKS[r[0]](i, priority=prio)
                print(f"job {i} re-sent to {r[0]} (priority {prio})")
    elif a.cmd == "workers":
        pidfile = os.path.join(QDIR, "workers.json")
        if a.action == "start":
            pids = {}
            for q in HUEYS:
                p = subprocess.Popen([PY, "-m", "huey.bin.huey_consumer", f"gpuq.HUEY_{q.upper()}", "-w", "1", "-k", "thread"],
                                     cwd=HERE, env=dict(os.environ, PYTHONPATH=HERE),
                                     stdout=open(os.path.join(QDIR, f"consumer_{q}.log"), "a"), stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
                pids[q] = p.pid
            json.dump(pids, open(pidfile, "w"))
            print(f"workers started: {pids}")
        elif a.action == "stop":
            pids = json.load(open(pidfile)) if os.path.exists(pidfile) else {}
            for q, pid in pids.items():
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
                print(f"stopped {q} worker {pid}")
        else:
            print(open(pidfile).read() if os.path.exists(pidfile) else "no workers")


if __name__ == "__main__":
    # run through the importable module so tasks are registered as gpuq.run_*, the names the consumers know
    sys.path.insert(0, HERE)
    import gpuq
    sys.exit(gpuq.main())
