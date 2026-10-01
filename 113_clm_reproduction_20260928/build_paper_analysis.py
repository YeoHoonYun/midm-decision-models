#!/usr/bin/env python3
"""Collect the paper's analysis tables into the research folder (paper/analysis/). Reads only result files
that already exist (no model runs, no new test reads).

    python build_paper_analysis.py            # -> research_topics/clm_decision_heads_novelty_20260928/paper/analysis

Writes results_master.{md,csv}, seeds.md, per_source.md, cascade.md, and copies stats/, ledger/, the
DeepSWE audit files and the search/sweep summaries. Rerun after new FINAL_*.json files arrive.
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTER = os.path.join(HERE, "..", "114_local_model_router_20260928")
DEFAULT_OUT = os.path.join(HERE, "..", "..", "research_topics", "clm_decision_heads_novelty_20260928",
                           "paper", "analysis")
SUITES = ["td_holdout", "kev_dev", "td_test", "kevT_dev", "kevT_test", "kevT9_dev", "jb_public"]
JB = ("jb_original", "jb_easy", "jb_hard")

# result tag -> (MiDM name, base, epochs, training data)
NAMES = {
    "qwen3-0.6b": ("MiDM-0.6B-q3-e1", "Qwen3-0.6B", 1, "td+kev"),
    "qwen3-1.7b": ("MiDM-1.7B-q3-e1", "Qwen3-1.7B", 1, "td+kev"),
    "qwen3-4b": ("MiDM-4B-q3-e1", "Qwen3-4B", 1, "td+kev"),
    "qwen3-8b": ("MiDM-8B-q3-e1", "Qwen3-8B", 1, "td+kev"),
    "B_qwen3-4b-2ep": ("MiDM-4B-q3-e2", "Qwen3-4B", 2, "td+kev"),
    "B_qwen3-8b-2ep": ("MiDM-8B-q3-e2", "Qwen3-8B", 2, "td+kev"),
    "B_qwen35-4b-1ep": ("MiDM-4B-q35-e1", "Qwen3.5-4B-Base", 1, "td+kev"),
    "B_qwen35-4b-2ep": ("MiDM-4B-q35-e2 (fp16)", "Qwen3.5-4B-Base", 2, "td+kev"),
    "B_qwen35-4b-2ep-bf16": ("MiDM-4B-q35-e2", "Qwen3.5-4B-Base", 2, "td+kev"),
    "D_qwen3-4b-1ep-broad": ("MiDM-4B-q3-e1-D1", "Qwen3-4B", 1, "td+kev+arc/obqa/csqa"),
    "E_q35_4b_e1_breadth": ("MiDM-4B-q35-e1-b (E1)", "Qwen3.5-4B-Base", 1, "td+kev+breadth_v1"),
    "E_q35_4b_e1_breadth_mc": ("MiDM-4B-q35-e1-bx (E2)", "Qwen3.5-4B-Base", 1, "td+kev+breadth_v1+pp6_new"),
    "T_q35_4b_e1_tta4": ("MiDM-4B-q35-e1 + TTA4", "Qwen3.5-4B-Base", 1, "td+kev"),
    "R_q35_0p8b_e1": ("MiDM-0.8B-q35-e1", "Qwen3.5-0.8B-Base", 1, "td+kev"),
    "R_q35_2b_e1": ("MiDM-2B-q35-e1", "Qwen3.5-2B-Base", 1, "td+kev"),
    "P_q3_4b_e1_bf16": ("MiDM-4B-q3-e1 (bf16 control)", "Qwen3-4B", 1, "td+kev"),
    "P_q35_2b_e1_bf16": ("MiDM-2B-q35-e1 (bf16 control)", "Qwen3.5-2B-Base", 1, "td+kev"),
    "R_q35_2b_e1_bx": ("MiDM-2B-q35-e1-bx", "Qwen3.5-2B-Base", 1, "td+kev+breadth_v1+pp6_new"),
    "R_q35_0p8b_e1_bx": ("MiDM-0.8B-q35-e1-bx", "Qwen3.5-0.8B-Base", 1, "td+kev+breadth_v1+pp6_new"),
    "H_q3_4b_headonly": ("ablation: joint options, no LoRA (Qwen3-4B, bf16)", "Qwen3-4B", 1, "td+kev"),
}
# per-source breakdowns: which probs files, in which order
BREAKDOWN = ["B_qwen3-8b-2ep", "B_qwen35-4b-1ep", "D_qwen3-4b-1ep-broad", "E_q35_4b_e1_breadth",
             "E_q35_4b_e1_breadth_mc", "R_q35_2b_e1", "R_q35_0p8b_e1"]
BREAKDOWN_SUITES = ["kevT_test", "kevT_dev", "kevT9_dev", "td_test"]


def scores(path):
    e = json.load(open(path))["eval"]
    out = {k: v["acc"] for k, v in e.items()}
    if all(k in e for k in JB):
        out["jb_public"] = sum(e[k]["acc"] * e[k]["n"] for k in JB) / sum(e[k]["n"] for k in JB)
    return out


def fmt(x):
    return "" if x is None else f"{x:.3f}"


def master(out):
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "results", "pointer", "FINAL_*.json"))) + \
            sorted(glob.glob(os.path.join(HERE, "results", "seeds", "FINAL_*.json"))):
        tag = os.path.basename(p)[6:-5]
        m = re.match(r"(.+)_s(\d+)$", tag)
        seed_of = {"q3_4b_e1": "qwen3-4b", "q35_4b_e1": "B_qwen35-4b-1ep", "q35_4b_e1_bx": "E_q35_4b_e1_breadth_mc"}
        if tag not in NAMES and m and m.group(1) in seed_of:
            n0, base, ep, data = NAMES[seed_of[m.group(1)]]
            name = f"{n0.split(' (')[0]} seed {m.group(2)}"
        else:
            name, base, ep, data = NAMES.get(tag, (tag, "", "", ""))
        sc = scores(p)
        rows.append({"tag": tag, "name": name, "base": base, "epochs": ep, "data": data,
                     "file": os.path.relpath(p, HERE).replace("\\", "/"), **{s: sc.get(s) for s in SUITES}})
    with open(os.path.join(out, "results_master.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["# Master results (accuracy, argmax over options)", "",
          "Dev suites: td_holdout, kev_dev (used for selection). Held-out suites: td_test, kevT_dev, kevT_test, "
          "kevT9_dev, jb_public (231 public JevBench items). Each held-out suite was read once per model; SHA-256 of "
          "each result file is in `ledger/`.", "",
          "| model | base | ep | train data | " + " | ".join(SUITES) + " |",
          "|---|---|---|---|" + "---|" * len(SUITES)]
    for r in rows:
        md.append(f"| {r['name']} | {r['base']} | {r['epochs']} | {r['data']} | "
                  + " | ".join(fmt(r[s]) for s in SUITES) + " |")
    md += ["", "Source files: `results_master.csv` (column `file`)."]
    open(os.path.join(out, "results_master.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    return rows


def seeds(out, rows):
    by = {r["tag"]: r for r in rows}
    groups = {}
    for r in rows:
        m = re.match(r"(.+)_s(\d+)$", r["tag"])
        if m:
            groups.setdefault(m.group(1), []).append(r)
    s0 = {"q3_4b_e1": "qwen3-4b", "q35_4b_e1": "B_qwen35-4b-1ep", "q35_4b_e1_bx": "E_q35_4b_e1_breadth_mc"}
    md = ["# Seed variation (mean ± sd over seeds; seed 0 = the main run)", "",
          "| group | n seeds | " + " | ".join(SUITES) + " |", "|---|---|" + "---|" * len(SUITES)]
    for g, rs in sorted(groups.items()):
        if s0.get(g) in by:
            rs = [by[s0[g]]] + rs
        cells = []
        for s in SUITES:
            v = [r[s] for r in rs if r[s] is not None]
            cells.append("" if not v else f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}" if len(v) > 1
                         else f"{v[0]:.3f}")
        md.append(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
    md += ["", "Paired-bootstrap seed-noise reference is in `stats/stats.md`."]
    open(os.path.join(out, "seeds.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


def per_source(out):
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # (suite, source) -> tag -> [hit, n]
    have = []
    for tag in BREAKDOWN:
        p = os.path.join(HERE, "results", "probs", f"{tag}.jsonl")
        if not os.path.exists(p):
            continue
        have.append(tag)
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["suite"] not in BREAKDOWN_SUITES:
                continue
            src = re.sub(r"^v\d+-legacy-transfer-|[-_]\d+$", "", r["group"].split("/")[0])
            pr = r["probs"]
            c = acc[(r["suite"], src)][tag]
            c[0] += int(max(range(len(pr)), key=pr.__getitem__) == r["label"])
            c[1] += 1
    md = ["# Per-source accuracy on held-out suites", "",
          "Source = first path component of each row's group id. Columns: " +
          ", ".join(f"`{NAMES[t][0]}`" for t in have) + ".", ""]
    for suite in BREAKDOWN_SUITES:
        srcs = sorted(s for (su, s) in acc if su == suite)
        if not srcs:
            continue
        md += [f"## {suite}", "", "| source | n | " + " | ".join(NAMES[t][0] for t in have) + " |",
               "|---|---|" + "---|" * len(have)]
        for s in srcs:
            d = acc[(suite, s)]
            n = max(v[1] for v in d.values())
            md.append(f"| {s} | {n} | " + " | ".join(fmt(d[t][0] / d[t][1]) if d[t][1] else "" for t in have) + " |")
        md.append("")
    open(os.path.join(out, "per_source.md"), "w", encoding="utf-8").write("\n".join(md))


def kevt9_new(out):
    """kevT9_dev restricted to rows whose source does not occur in kevT_dev (the independent part)."""
    def src(g):
        return re.sub(r"^v\d+-legacy-transfer-|[-_]\d+$", "", g.split("/")[0])
    md = ["# kevT9_dev without the kevT_dev sources", "",
          "kevT9_dev contains every kevT_dev source with identical items. This table scores only the new sources, so it "
          "is independent of kevT_dev (which was used to choose the unknown-domain model).", "",
          "| model | kevT9_dev (all) | kevT9_dev new sources only | n new | new sources |", "|---|---|---|---|---|"]
    for tag in NAMES:
        p = os.path.join(HERE, "results", "probs", f"{tag}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        dev_src = {src(r["group"]) for r in rows if r["suite"] == "kevT_dev"}
        t9 = [r for r in rows if r["suite"] == "kevT9_dev"]
        if not t9 or not dev_src:
            continue
        hit = lambda r: int(max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["label"])
        new = [r for r in t9 if src(r["group"]) not in dev_src]
        md.append(f"| {NAMES[tag][0]} | {sum(map(hit, t9)) / len(t9):.3f} | {sum(map(hit, new)) / len(new):.3f} | "
                  f"{len(new)} | {', '.join(sorted({src(r['group']) for r in new}))} |")
    open(os.path.join(out, "kevT9_new_sources.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


def cascade(out):
    md = ["# Confidence cascades (exp 114 router)", "",
          "Thresholds chosen on dev (kev_dev + td_holdout) as the cheapest setting within `tol` of the best "
          "single tier; cost units: 0.6B = 1. Rows below apply the chosen thresholds to held-out suites.", ""]
    for p in sorted(glob.glob(os.path.join(ROUTER, "results", "cascade*.json"))):
        d = json.load(open(p))
        ch = d["chosen"]
        md += [f"## {os.path.basename(p)[:-5]}", "",
               f"tiers {d['tiers']}, cost {d['cost']}, thresholds {ch['thr']}; dev acc {ch['acc']:.3f} "
               f"(single tiers: " + ", ".join(f"{k} {v:.3f}" for k, v in d["dev_single_acc"].items()) +
               f"), dev mean cost {ch['mean_cost']:.2f}", "",
               "| suite | acc | escalated | mean cost | share by tier |", "|---|---|---|---|---|"]
        for s, v in sorted(d.get("eval_cascade", {}).items()):
            md.append(f"| {s} | {v['acc']:.3f} | {v['escalated']:.2f} | {v['mean_cost']:.2f} | "
                      + " / ".join(f"{x:.2f}" for x in v["share_answered_by_tier"]) + " |")
        md.append("")
    open(os.path.join(out, "cascade.md"), "w", encoding="utf-8").write("\n".join(md))


def copies(out):
    for sub, files in {"stats": ["results/stats/stats.md", "results/stats/stats.json"],
                       "ledger": ["results/ledger/test_read_ledger.md", "results/ledger/test_read_ledger.jsonl"],
                       "deepswe_audit": ["results/r1_deepswe_heldout38_released_head.json",
                                         "results/r2_deepswe_heldout38_zeroshot_v0.1_head.json",
                                         "results/deepswe_lto/summary.md", "results/deepswe_lto/summary.json"],
                       "search": ["results/sweep_summary.md", "results/choice_summary.md"],
                       "router": ["../114_local_model_router_20260928/DECISIONS.md",
                                  "../114_local_model_router_20260928/MODELS.md"]}.items():
        os.makedirs(os.path.join(out, sub), exist_ok=True)
        for f in files:
            src = os.path.join(HERE, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out, sub, os.path.basename(f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    out = os.path.abspath(ap.parse_args().out)
    os.makedirs(out, exist_ok=True)
    rows = master(out)
    seeds(out, rows)
    per_source(out)
    kevt9_new(out)
    cascade(out)
    copies(out)
    print(f"wrote {out}")
    for f in sorted(os.listdir(out)):
        print("  ", f)


if __name__ == "__main__":
    main()
