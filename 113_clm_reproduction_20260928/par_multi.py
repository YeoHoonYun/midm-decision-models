#!/usr/bin/env python3
"""Run multi_train.py configs with bounded concurrency on one GPU.
par_multi.py --gpu 0 --workers 2 --tag m1 -- "<args>" ..."""
import argparse
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--tag", required=True)
ap.add_argument("configs", nargs="+")
a = ap.parse_args()
env = dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1")
os.makedirs(os.path.join(HERE, "logs", "multi"), exist_ok=True)


def run(i_cfg):
    i, cfg = i_cfg
    out = os.path.join(HERE, "results", "multi", f"{a.tag}_{i:02d}.json")
    if os.path.exists(out):
        return i, cfg, "skip", 0
    log = os.path.join(HERE, "logs", "multi", f"{a.tag}_{i:02d}.log")
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run([PY, os.path.join(HERE, "multi_train.py"), *shlex.split(cfg), "--out", out],
                            stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    lines = [l for l in open(log).read().splitlines() if l.strip()]
    return i, cfg, (lines[-1] if lines else "") if rc == 0 else f"rc={rc} {lines[-1] if lines else ''}", time.time() - t0


with ThreadPoolExecutor(a.workers) as ex:
    for i, cfg, msg, sec in ex.map(run, enumerate(a.configs)):
        print(f"[{a.tag}_{i:02d}] {sec:4.0f}s | {msg}", flush=True)
