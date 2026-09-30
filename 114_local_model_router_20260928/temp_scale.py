#!/usr/bin/env python3
"""Temperature scaling per tier, fitted on DEV only (kev_dev + td_holdout, minimum NLL over a grid).
Writes calibrated copies T_<tier>.jsonl next to the originals (p_i^(1/T) renormalised = logits / T)
so calibrate_cascade.py can be re-run on them. Prints the dev/eval ECE before and after.

    python temp_scale.py qwen3-0.6b B_qwen35-4b-1ep B_qwen3-8b-2ep
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

PROBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "113_clm_reproduction_20260928", "results", "probs")
DEV = {"kev_dev", "td_holdout"}


def rows(tier):
    return [json.loads(l) for l in open(os.path.join(PROBS, f"{tier}.jsonl"), encoding="utf-8")]


def scale(p, T):
    lp = np.log(np.clip(np.asarray(p, float), 1e-12, 1)) / T
    lp -= lp.max()
    e = np.exp(lp)
    return e / e.sum()


def nll(rs, T):
    return -np.mean([np.log(max(scale(r["probs"], T)[r["label"]], 1e-12)) for r in rs])


def ece(rs, T, bins=10):
    conf, corr = [], []
    for r in rs:
        q = scale(r["probs"], T)
        conf.append(q.max()); corr.append(int(q.argmax() == r["label"]))
    conf, corr = np.array(conf), np.array(corr)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return e


def main():
    for tier in sys.argv[1:]:
        rs = rows(tier)
        dev = [r for r in rs if r["suite"] in DEV]
        grid = np.exp(np.linspace(np.log(0.25), np.log(8), 121))
        T = float(grid[int(np.argmin([nll(dev, t) for t in grid]))])
        by = defaultdict(list)
        for r in rs:
            by[r["suite"]].append(r)
        print(f"{tier}: T = {T:.3f} (dev NLL {nll(dev, 1):.4f} -> {nll(dev, T):.4f})")
        for s in sorted(by):
            print(f"   {s:12s} ECE {ece(by[s], 1):.3f} -> {ece(by[s], T):.3f}")
        with open(os.path.join(PROBS, f"T_{tier}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({**r, "probs": scale(r["probs"], T).tolist(), "temperature": T}) + "\n")


if __name__ == "__main__":
    main()
