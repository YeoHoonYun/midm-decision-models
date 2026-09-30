#!/usr/bin/env python3
"""Run head_search.py configs in parallel: par.py --workers 4 --tag r1 -- "<args>" "<args>" ...
Each config string gets --out results/search/<tag>_<i>.json; prints one line per finished job."""
import argparse
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--tag", required=True)
ap.add_argument("--gpu", default=None, help="physical GPU index to use (adds --device cuda)")
ap.add_argument("configs", nargs="+")
a = ap.parse_args()
env = dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES=a.gpu or "", OMP_NUM_THREADS=str(a.threads))
extra = ["--device", "cuda"] if a.gpu is not None else []
os.makedirs(os.path.join(HERE, "logs", "search"), exist_ok=True)


def run(i_cfg):
    i, cfg = i_cfg
    out = os.path.join(HERE, "results", "search", f"{a.tag}_{i:02d}.json")
    if os.path.exists(out):
        return i, cfg, "skip", 0
    cmd = [PY, os.path.join(HERE, "head_search.py"), *shlex.split(cfg), *extra, "--threads", str(a.threads), "--out", out]
    t0 = time.time()
    with open(os.path.join(HERE, "logs", "search", f"{a.tag}_{i:02d}.log"), "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    line = open(os.path.join(HERE, "logs", "search", f"{a.tag}_{i:02d}.log")).read().strip().splitlines()
    return i, cfg, (line[-1] if line else "") if rc == 0 else f"rc={rc} {line[-1] if line else ''}", time.time() - t0


with ThreadPoolExecutor(a.workers) as ex:
    for i, cfg, msg, sec in ex.map(run, enumerate(a.configs)):
        print(f"[{a.tag}_{i:02d}] {sec:5.0f}s | {cfg} | {msg}", flush=True)
