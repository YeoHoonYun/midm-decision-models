#!/usr/bin/env python3
"""Choose decider-auto cascade thresholds from per-question probabilities (exp 113 --save-probs dumps).

A cascade over tiers T1..Tk: a question stays at tier i if max prob >= t_i, else moves to i+1; the last
tier always answers. Thresholds are chosen on DEV suites only (kev_dev, td_holdout) with the rule in
DECISIONS.md: the smallest escalation (mean tier cost) whose dev accuracy is within --tol of the best
single tier's dev accuracy. The chosen thresholds are then applied once to the evaluation suites.

    python calibrate_cascade.py --tiers qwen3-0.6b qwen3-4b --cost 1 6 --out results/cascade_0.6b_4b.json
"""
import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROBS = os.path.join(HERE, "..", "113_clm_reproduction_20260928", "results", "probs")
DEV = {"kev_dev", "td_holdout"}


def load(tier):
    """Key = (suite, position within suite). (suite, qid, group) is NOT unique (several questions share
    it), and keying by it silently dropped rows. Dumps list each suite's examples in the same
    decision_data order for every model, so the position aligns tiers; qid/group are checked for agreement."""
    rows, pos = {}, defaultdict(int)
    for line in open(os.path.join(PROBS, f"{tier}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        rows[(r["suite"], pos[r["suite"]])] = r
        pos[r["suite"]] += 1
    return rows


def check_alignment(tiers_rows):
    ref = tiers_rows[0]
    for rows in tiers_rows[1:]:
        for k in set(ref) & set(rows):
            a, b = ref[k], rows[k]
            if (a["qid"], a["group"], a["label"]) != (b["qid"], b["group"], b["label"]):
                raise ValueError(f"tiers disagree at {k}: {a['qid']} vs {b['qid']}")


def run(tiers_rows, thr, keys):
    """-> per-key (correct, tier index)"""
    out = {}
    for k in keys:
        for i, rows in enumerate(tiers_rows):
            r = rows[k]
            p = np.asarray(r["probs"])
            if i == len(tiers_rows) - 1 or p.max() >= thr[i]:
                out[k] = (int(p.argmax()) == r["label"], i)
                break
    return out


def summarize(res, keys, cost):
    acc = np.mean([res[k][0] for k in keys])
    tiers = np.array([res[k][1] for k in keys])
    share = [float(np.mean(tiers == i)) for i in range(len(cost))]
    # expected cost per question: every question pays tier 0; escalated ones also pay later tiers
    exp_cost = float(np.mean([sum(cost[:t + 1]) for t in tiers]))
    return {"acc": float(acc), "share_answered_by_tier": share, "escalated": 1 - share[0], "mean_cost": exp_cost}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", required=True)
    ap.add_argument("--cost", nargs="+", type=float, required=True, help="relative cost/latency per tier")
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tiers_rows = [load(t) for t in a.tiers]
    check_alignment(tiers_rows)
    common = set.intersection(*(set(r) for r in tiers_rows))
    dev = sorted(k for k in common if k[0] in DEV)
    evalk = sorted(k for k in common if k[0] not in DEV)
    grid = [round(x, 2) for x in np.arange(0.30, 1.001, 0.02)]
    single = {t: summarize(run([rows], [], dev), dev, [1]) ["acc"] for t, rows in zip(a.tiers, tiers_rows)}
    best_single = max(single.values())
    curve = []
    for thr in itertools.product(grid, repeat=len(a.tiers) - 1):
        s = summarize(run(tiers_rows, list(thr), dev), dev, a.cost)
        curve.append({"thr": list(thr), **s})
    ok = [c for c in curve if c["acc"] >= best_single - a.tol]
    chosen = min(ok, key=lambda c: (c["mean_cost"], -c["acc"])) if ok else max(curve, key=lambda c: c["acc"])
    by_suite = defaultdict(list)
    for k in evalk:
        by_suite[k[0]].append(k)
    ev = {s: summarize(run(tiers_rows, chosen["thr"], ks), ks, a.cost) for s, ks in sorted(by_suite.items())}
    ev_single = {t: {s: summarize(run([rows], [], ks), ks, [1])["acc"] for s, ks in sorted(by_suite.items())}
                 for t, rows in zip(a.tiers, tiers_rows)}
    res = {"tiers": a.tiers, "cost": a.cost, "tol": a.tol, "dev_n": len(dev), "dev_single_acc": single,
           "chosen": chosen, "eval_cascade": ev, "eval_single": ev_single,
           "pareto": sorted(({"thr": c["thr"], "acc": c["acc"], "mean_cost": c["mean_cost"]} for c in curve),
                            key=lambda c: c["mean_cost"])[::max(1, len(curve) // 40)]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"dev_single": single, "chosen": chosen}, indent=1))
    for s, v in ev.items():
        print(f"{s:12s} cascade acc {v['acc']:.3f} escalated {v['escalated']:.2f} | single "
              + " ".join(f"{t}={ev_single[t][s]:.3f}" for t in a.tiers))


if __name__ == "__main__":
    main()
