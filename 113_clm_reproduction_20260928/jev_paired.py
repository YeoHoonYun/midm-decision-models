#!/usr/bin/env python3
"""Paired comparison with TypeSafe Jev on the 231 public JevBench items.

JevBench publishes, for every system, the per-item outcome on the public items
(results/v1.2/jevbench-v1.2-per-task.json -> systems[...]["public_tasks"][task_id] = ["c"|"w"|..., latency]).
We align those with our per-item probability dumps (same public items, read once) and report accuracy by
tier and by hard-tier family, plus a paired bootstrap over items for MiDM vs Jev.

    python jev_paired.py      # -> results/jev_paired/jev_paired.{json,md}
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PER_TASK = os.path.join(HERE, "ext", "jevbench", "results", "v1.2", "jevbench-v1.2-per-task.json")
PUB = os.path.join(HERE, "ext", "jevbench", "datasets", "public")
OURS = {"MiDM-4B-q35-e1-bx": "E_q35_4b_e1_breadth_mc", "MiDM-4B-q35-e1": "B_qwen35-4b-1ep",
        "MiDM-8B-q3-e2": "B_qwen3-8b-2ep"}
OTHERS = ["jev-1.13.0", "kev-8b", "kev-4b"]   # other systems with public per-item results
OUT = os.path.join(HERE, "results", "jev_paired")


def public_items():
    """Item ids in the order decision_data reads them (file order per tier), with tier and family/topic."""
    items = []
    for tier, f in (("original", "original.jsonl"), ("easy", "easy.jsonl"), ("hard", "hard.jsonl")):
        for l in open(os.path.join(PUB, f), encoding="utf-8"):
            r = json.loads(l)
            items.append({"id": r["id"], "tier": tier, "family": r.get("family") or r.get("topic") or ""})
    return items


def ours(tag, items):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "results", "probs", f"{tag}.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["suite"].startswith("jb_")]
    assert len(rows) == len(items), (len(rows), len(items))
    out = {}
    for it, r in zip(items, rows):
        assert r["suite"] == f"jb_{it['tier']}", (r["suite"], it)
        p = r["probs"]
        out[it["id"]] = int(max(range(len(p)), key=p.__getitem__) == r["label"])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    items = public_items()
    d = json.load(open(PER_TASK, encoding="utf-8"))
    sysres = {}
    for s in OTHERS:
        if s in d["systems"]:
            pt = d["systems"][s]["public_tasks"]
            missing = [it["id"] for it in items if it["id"] not in pt]
            if missing:
                print(f"[jev] {s}: {len(missing)} public ids not found, e.g. {missing[:3]}")
            sysres[s] = {it["id"]: int(pt.get(it["id"], ["w"])[0] == "c") for it in items}
    for name, tag in OURS.items():
        sysres[name] = ours(tag, items)

    def acc(sysname, sel):
        v = [sysres[sysname][it["id"]] for it in sel]
        return sum(v) / len(v) if v else float("nan")

    groups = {"all public (231)": items}
    for t in ("original", "easy", "hard"):
        groups[f"tier {t}"] = [it for it in items if it["tier"] == t]
    fam = defaultdict(list)
    for it in items:
        if it["tier"] == "hard":
            fam[it["family"]].append(it)
    for f, v in sorted(fam.items()):
        groups[f"hard: {f}"] = v
    names = list(sysres)
    table = {g: {"n": len(sel), **{s: acc(s, sel) for s in names}} for g, sel in groups.items()}

    # paired bootstrap over items: MiDM-bx minus Jev
    rng = np.random.default_rng(0)
    paired = {}
    for g in ("all public (231)", "tier hard", "tier original"):
        sel = groups[g]
        a = np.array([sysres["jev-1.13.0"][it["id"]] for it in sel], float)
        b = np.array([sysres["MiDM-4B-q35-e1-bx"][it["id"]] for it in sel], float)
        idx = rng.integers(0, len(sel), size=(4000, len(sel)))
        boots = b[idx].mean(1) - a[idx].mean(1)
        paired[g] = {"diff": float(b.mean() - a.mean()), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
                     "midm_only_correct": int(((b == 1) & (a == 0)).sum()), "jev_only_correct": int(((a == 1) & (b == 0)).sum())}
    json.dump({"table": table, "paired_midm_bx_minus_jev": paired, "source": os.path.relpath(PER_TASK, HERE)},
              open(os.path.join(OUT, "jev_paired.json"), "w"), indent=1)
    md = ["# JevBench public items: per-item paired comparison with Jev 1.13.0", "",
          "Jev and Kev outcomes are JevBench's published per-item results (v1.2 per-task file); ours are our own runs "
          "on the same 231 public items.", "",
          "| subset | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for g, v in table.items():
        md.append(f"| {g} | {v['n']} | " + " | ".join(f"{v[s]:.3f}" for s in names) + " |")
    md += ["", "Paired (MiDM-4B-q35-e1-bx minus Jev, bootstrap over items):", ""]
    for g, v in paired.items():
        md.append(f"- {g}: {100 * v['diff']:+.1f} pp [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]; "
                  f"only MiDM correct {v['midm_only_correct']}, only Jev correct {v['jev_only_correct']}")
    open(os.path.join(OUT, "jev_paired.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
