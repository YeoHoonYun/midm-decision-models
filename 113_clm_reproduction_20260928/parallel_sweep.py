#!/usr/bin/env python3
"""Run many cached-embedding head trainings in parallel on CPU.

Each job is one run_choice_hf.py invocation (encoder x workflow x seed x patience).
Embeddings must already be in runs/emb_cache; jobs are CPU-only (CUDA hidden) and
each process gets --threads torch threads. Finished runs are skipped.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")


def job(enc, wf, seed, pat, threads):
    tag = f"choice_{enc.split('/')[-1].lower()}_{wf}_p{pat}_s{seed}"
    out = os.path.join(HERE, "runs", "sweep", tag)
    if os.path.exists(os.path.join(out, "finetune_summary.json")):
        return tag, "skip", 0.0
    env = dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), HF_HOME=os.path.join(HERE, "hf_cache"),
               HF_HUB_DISABLE_SYMLINKS_WARNING="1", HF_HUB_OFFLINE="1")
    cmd = [PY, os.path.join(HERE, "run_choice_hf.py"), "--embed-model", enc, "--dtype", "float16",
           "--device-map", "cpu", "--", "--task", "choice", "--data", os.path.join(HERE, "data", "typed-decisions"),
           "--workflow", wf, "--out-dir", out, "--embed-cache", os.path.join(HERE, "runs", "emb_cache"),
           "--seed", str(seed), "--patience", str(pat)]
    os.makedirs(os.path.join(HERE, "logs", "sweep"), exist_ok=True)
    t0 = time.time()
    with open(os.path.join(HERE, "logs", "sweep", tag + ".log"), "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    return tag, "ok" if rc == 0 else f"rc={rc}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"])
    ap.add_argument("--workflows", nargs="+", default=["agent_trace_observability", "customer_service",
                                                         "invoice_processing", "security_incidents"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1234, 1, 2])
    ap.add_argument("--patience", nargs="+", type=int, default=[5, 20])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    jobs = list(itertools.product(a.encoders, a.workflows, a.seeds, a.patience))
    print(f"[sweep] {len(jobs)} jobs, {a.workers} workers x {a.threads} threads", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for tag, status, sec in ex.map(lambda j: job(*j, a.threads), jobs):
            print(f"[sweep] {status:6s} {sec:6.1f}s {tag}", flush=True)
    print(f"[sweep] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
