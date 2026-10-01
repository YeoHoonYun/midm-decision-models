#!/usr/bin/env python3
"""Selection accuracy from the selector probabilities. The rule's free parameter is chosen on Spider dev only and
then applied once to Spider test, the private holdout and LiveCodeBench hard.

    python score_selector.py --tag bx

Rules (per question, over the deduplicated options):
  vote      largest cluster (ties -> first option)
  midm      argmax MiDM probability
  vote+tb   largest cluster, ties broken by MiDM
  blend     argmax  log p_midm + lam * log n_votes   (lam tuned on spider_dev)
  oracle    any correct option
Baselines from the pool files: Solar direct (per-question correctness), each single arm.
Paired bootstrap (clustered by db_id for SQL, by problem for code) for blend vs vote and blend vs Solar.
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["spider_dev", "spider_test", "holdout", "lcb_hard"]
LAMS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def load(split, tag, rag):
    p = os.path.join(HERE, "results", f"probs_{split}_{tag}_rag{rag}.jsonl")
    if not os.path.exists(p):
        return None
    probs = {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
    pool = {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, "data", f"{split}.jsonl"), encoding="utf-8"))}
    return [(pool[k], probs[k]) for k in pool if k in probs and not pool[k].get("label_excluded_nondeterministic")]


def pick(rule, pr, lam=0.0):
    p, v = pr["probs"], pr["n_votes"]
    ex = pr.get("executable") or [True] * len(v)   # failed executions are never picked by the vote rules
    if rule == "vote":
        return max(range(len(v)), key=lambda j: (ex[j], v[j], -j))
    if rule == "midm":
        return max(range(len(p)), key=p.__getitem__)
    if rule == "vote+tb":
        return max(range(len(v)), key=lambda j: (ex[j], v[j], p[j]))
    if rule == "blend":
        return max(range(len(p)), key=lambda j: math.log(max(p[j], 1e-9)) + lam * math.log(v[j]))
    raise ValueError(rule)


def acc_vec(rows, rule, lam=0.0):
    if rule == "oracle":
        return np.array([float(any(pr["correct"])) for _, pr in rows])
    if rule == "solar":
        return np.array([float(bool((rec.get("baseline") or {}).get("solar"))) for rec, _ in rows])
    return np.array([float(pr["correct"][pick(rule, pr, lam)]) for _, pr in rows])


def boot(rows, a, b, B=4000, seed=0):
    groups = defaultdict(list)
    for i, (rec, _) in enumerate(rows):
        groups[rec.get("db_id") or rec["id"]].append(i)
    g = list(groups.values())
    da = np.array([a[ix].sum() for ix in g]); db = np.array([b[ix].sum() for ix in g]); n = np.array([len(ix) for ix in g])
    idx = np.random.default_rng(seed).integers(0, len(g), size=(B, len(g)))
    d = (db[idx].sum(1) - da[idx].sum(1)) / n[idx].sum(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float((b.sum() - a.sum()) / len(b)), "ci95": [float(lo), float(hi)],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bx")
    a = ap.parse_args()
    res = {}
    for rag in (0, 5):
        dev = load("spider_dev", a.tag, rag)
        if not dev:
            continue
        lam = max(LAMS, key=lambda l: (acc_vec(dev, "blend", l).mean(), -l))   # chosen on dev only
        res[f"rag{rag}"] = {"lambda_from_spider_dev": lam, "splits": {}}
        for split in SPLITS:
            rows = load(split, a.tag, rag)
            if not rows:
                continue
            accs = {r: acc_vec(rows, r, lam if r == "blend" else 0.0) for r in ("vote", "midm", "vote+tb", "blend", "oracle", "solar")}
            has_solar = any((rec.get("baseline") or {}).get("solar") is not None for rec, _ in rows)
            out = {"n": len(rows), "acc": {r: float(v.mean()) for r, v in accs.items() if r != "solar" or has_solar},
                   "blend_vs_vote": boot(rows, accs["vote"], accs["blend"]),
                   "midm_vs_vote": boot(rows, accs["vote"], accs["midm"])}
            if has_solar:
                out["blend_vs_solar"] = boot(rows, accs["solar"], accs["blend"])
            res[f"rag{rag}"]["splits"][split] = out
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", f"SELECTION_{a.tag}.json"), "w"), indent=1)
    md = [f"# MiDM as a best-of-N selector ({a.tag})", "",
          "Selection rule parameter (lambda) chosen on Spider dev only; every other split read once.", ""]
    for k, v in res.items():
        md += [f"## {k} (lambda = {v['lambda_from_spider_dev']})", "",
               "| split | n | vote | MiDM | vote + MiDM tie-break | blend | oracle | Solar | blend - vote (CI, p) | blend - Solar (CI, p) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for s, o in v["splits"].items():
            f = lambda x: "" if x is None else f"{x:.4f}"
            bv, bs = o["blend_vs_vote"], o.get("blend_vs_solar")
            md.append(f"| {s} | {o['n']} | {f(o['acc']['vote'])} | {f(o['acc']['midm'])} | {f(o['acc']['vote+tb'])} | "
                      f"{f(o['acc']['blend'])} | {f(o['acc']['oracle'])} | {f(o['acc'].get('solar'))} | "
                      f"{100 * bv['diff']:+.2f} [{100 * bv['ci95'][0]:+.2f}, {100 * bv['ci95'][1]:+.2f}] p={bv['p']:.3f} | "
                      + (f"{100 * bs['diff']:+.2f} [{100 * bs['ci95'][0]:+.2f}, {100 * bs['ci95'][1]:+.2f}] p={bs['p']:.3f} |" if bs else " |"))
        md.append("")
    open(os.path.join(HERE, "results", f"SELECTION_{a.tag}.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
