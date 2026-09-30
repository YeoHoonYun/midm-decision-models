#!/usr/bin/env python3
"""Probability-average ensembles of saved MiDM dumps (CPU). The weight is chosen on DEV only
(kev_dev + td_holdout). The ensemble dump is then written once as results/probs/ENS_<name>.jsonl for
stats.py. Adopt only if the dev accuracy of both dev suites is >= each member's.

    python ensemble.py B_qwen35-4b-1ep B_qwen3-8b-2ep --name q35x4b_q3x8b
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "results", "probs")
DEV = ("kev_dev", "td_holdout")


def load(t):
    rows, pos = {}, defaultdict(int)
    for l in open(os.path.join(P, f"{t}.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        rows[(r["suite"], pos[r["suite"]])] = r
        pos[r["suite"]] += 1
    return rows


def acc(rs):
    return float(np.mean([int(np.argmax(r["probs"]) == r["label"]) for r in rs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("members", nargs=2)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    A, B = load(a.members[0]), load(a.members[1])
    keys = [k for k in A if k in B]
    for k in keys:
        if A[k]["qid"] != B[k]["qid"] or A[k]["label"] != B[k]["label"]:
            raise ValueError(f"misaligned {k}")

    def mix(w):
        return {k: {**A[k], "probs": (w * np.asarray(A[k]["probs"]) + (1 - w) * np.asarray(B[k]["probs"])).tolist()}
                for k in keys}

    dev = {s: [k for k in keys if k[0] == s] for s in DEV}
    best = None
    for w in np.linspace(0, 1, 11):
        m = mix(w)
        score = np.mean([acc([m[k] for k in dev[s]]) for s in DEV])
        if best is None or score > best[1] + 1e-12:
            best = (float(w), score)
    w = best[0]
    m = mix(w)
    single = {name: {s: acc([R[k] for k in dev[s]]) for s in DEV} for name, R in ((a.members[0], A), (a.members[1], B))}
    ens_dev = {s: acc([m[k] for k in dev[s]]) for s in DEV}
    adopt = all(ens_dev[s] >= max(single[n][s] for n in single) - 1e-12 for s in DEV)
    print(f"dev-chosen weight on {a.members[0]}: {w:.1f}; dev ens {ens_dev} vs members {single}; adopt={adopt}")
    if adopt:
        with open(os.path.join(P, f"ENS_{a.name}.jsonl"), "w", encoding="utf-8") as f:
            for k in keys:
                f.write(json.dumps({**m[k], "ensemble": {"members": a.members, "w": w}}) + "\n")
        by = defaultdict(list)
        for k in keys:
            by[k[0]].append(m[k])
        print("eval (read once):", {s: round(acc(v), 4) for s, v in sorted(by.items()) if s not in DEV})


if __name__ == "__main__":
    main()
