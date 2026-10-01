#!/usr/bin/env python3
"""Speed summary for the paper: MiDM batched throughput (from the one-time eval logs), single-question latency
(results/latency/latency.json, if measured) and the per-item response times JevBench publishes for Jev and Kev.

    python speed_summary.py      # -> results/speed/speed.{md,json}

Batched throughput = questions / eval seconds over all suites of a FINAL_*.json (token-budget batches of 4096,
4-bit NF4 base, transformers on Windows, no vLLM). JevBench times are end-to-end per item as published by the
benchmark (hosted API, so network round trip included); they are not a like-for-like model speed comparison.
"""
import json
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "speed")
MODELS = [("qwen3-0.6b", "MiDM-0.6B-q3-e1"), ("qwen3-1.7b", "MiDM-1.7B-q3-e1"), ("R_q35_0p8b_e1", "MiDM-0.8B-q35-e1"),
          ("R_q35_2b_e1", "MiDM-2B-q35-e1"), ("P_q35_2b_e1_bf16", "MiDM-2B-q35-e1 (bf16)"), ("qwen3-4b", "MiDM-4B-q3-e1"),
          ("P_q3_4b_e1_bf16", "MiDM-4B-q3-e1 (bf16)"), ("B_qwen35-4b-1ep", "MiDM-4B-q35-e1"),
          ("E_q35_4b_e1_breadth_mc", "MiDM-4B-q35-e1-bx"), ("qwen3-8b", "MiDM-8B-q3-e1"), ("B_qwen3-8b-2ep", "MiDM-8B-q3-e2")]
JEV_SYSTEMS = ["jev-1.13.0", "kev-8b", "kev-4b", "kev-0.6b"]
PER_TASK = os.path.join(HERE, "ext", "jevbench", "results", "v1.2", "jevbench-v1.2-per-task.json")


def gpu_of_results():
    out = {}
    c = sqlite3.connect(os.path.join(HERE, "runs", "gpuq", "jobs.db"))
    for q, cmd, env in c.execute("select queue, cmd, env from jobs where state='done'"):
        for a in json.loads(cmd):
            if a.endswith(".json") and "FINAL_" in a:
                bf16 = "PTR_BF16" in (env or "")
                out[os.path.basename(a)] = {"gpu0": "RTX 3090", "gpu1": "RTX 2080 Ti"}[q] + (" bf16" if bf16 else " fp16")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    where = gpu_of_results()
    thr = []
    for tag, name in MODELS:
        f = os.path.join(HERE, "results", "pointer", f"FINAL_{tag}.json")
        if not os.path.exists(f):
            continue
        e = json.load(open(f))["eval"]
        n, s = sum(v["n"] for v in e.values()), sum(v["sec"] for v in e.values())
        td = e.get("td_test")
        thr.append({"model": name, "hw": where.get(os.path.basename(f), "manual run (before queue)"), "questions": n,
                    "seconds": s, "q_per_s": n / s, "ms_per_q": 1000 * s / n,
                    "td_test_ms_per_q": 1000 * td["sec"] / td["n"] if td else None})
    jev = {}
    if os.path.exists(PER_TASK):
        d = json.load(open(PER_TASK, encoding="utf-8"))
        for s in JEV_SYSTEMS:
            v = sorted(x[1] for x in d["systems"][s]["public_tasks"].values() if isinstance(x, list) and len(x) > 1 and x[1] is not None)
            jev[s] = {"n": len(v), "p50_s": v[len(v) // 2], "p95_s": v[int(len(v) * 0.95)]}
    lat_path = os.path.join(HERE, "results", "latency", "latency.json")
    lat = json.load(open(lat_path)) if os.path.exists(lat_path) else None
    json.dump({"throughput": thr, "jevbench_published_latency": jev, "single_question_latency": lat},
              open(os.path.join(OUT, "speed.json"), "w"), indent=1)

    md = ["# Speed", "",
          "## 1. MiDM batched throughput (from the eval logs)", "",
          "Token-budget batches (4096 tokens), 4-bit NF4 base, transformers on Windows, no vLLM. Each row covers all "
          "eval suites of that model (~6.5–7.1k questions).", "",
          "| model | hardware | questions | q/s | ms per question | td_test ms/q |", "|---|---|---|---|---|---|"]
    md += [f"| {r['model']} | {r['hw']} | {r['questions']} | {r['q_per_s']:.1f} | {r['ms_per_q']:.1f} | "
           f"{r['td_test_ms_per_q']:.1f} |" for r in thr]
    md += ["", "## 2. Single-question latency (batch 1)", ""]
    if lat:
        md += [f"Measured on {lat['gpu']} ({lat['dtype']}), n={lat['n']} td_holdout questions.", "",
               "| tier | p50 ms | p90 ms | mean ms | cost ratio vs smallest |", "|---|---|---|---|---|"]
        md += [f"| {k} | {v['p50_ms']:.1f} | {v['p90_ms']:.1f} | {v['mean_ms']:.1f} | {v['cost_ratio']:.2f} |"
               for k, v in lat["tiers"].items()]
    else:
        md += ["Pending: queue job `LAT_cascade_tiers` (latency.py) runs after the seed jobs."]
    if lat:   # cascade expected latency with measured per-tier latency (sequential escalation)
        alias = {"qwen3-0.6b": "MiDM-0.6B-q3-e1", "qwen3-1.7b": "MiDM-1.7B-q3-e1",
                 "B_qwen35-4b-1ep": "MiDM-4B-q35-e1", "B_qwen3-8b-2ep": "MiDM-8B-q3-e2"}
        rdir = os.path.join(HERE, "..", "114_local_model_router_20260928", "results")
        rows = []
        for f in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
            if not f.startswith("cascade") or not f.endswith(".json"):
                continue
            c = json.load(open(os.path.join(rdir, f)))
            names = [alias.get(t) for t in c["tiers"]]
            if None in names or any(n not in lat["tiers"] for n in names):
                continue
            ms = [lat["tiers"][n]["mean_ms"] for n in names]
            share = c["chosen"]["share_answered_by_tier"]
            reach = [sum(share[i:]) for i in range(len(share))]
            exp_ms = sum(r * m for r, m in zip(reach, ms))
            rows.append((f[:-5], " → ".join(names), c["chosen"]["acc"], exp_ms, ms[-1]))
        if rows:
            md += ["", "### Cascade expected latency with the measured tiers (dev escalation shares)", "",
                   "| cascade | tiers | dev acc | expected ms/question | top tier alone ms |", "|---|---|---|---|---|"]
            md += [f"| {a} | {b} | {c:.3f} | {d:.0f} | {e:.0f} |" for a, b, c, d, e in rows]
            md += ["", "At batch 1 in this stack (4-bit NF4, transformers, Windows) per-call overhead dominates: the 0.6B tier "
                   "costs ~150 ms and the 8B ~196 ms, and Qwen3.5-4B is slowest (~309 ms) because its linear-attention "
                   "kernels fall back to the reference PyTorch implementation (no flash-linear-attention / causal-conv1d). "
                   "The provisional cost weights (0.6B=1, 4B=6, 8B=12) therefore overstate the saving; with measured "
                   "latency a cascade is not faster than the 8B tier alone here."]
    md += ["", "## 3. Published per-item response time on JevBench public (231 items)", "",
           "From JevBench's per-task results file. Hosted endpoints, so this is end-to-end time including the network; "
           "it is not a like-for-like model speed comparison with the local numbers above.", "",
           "| system | p50 s | p95 s |", "|---|---|---|"]
    md += [f"| {k} | {v['p50_s']:.3f} | {v['p95_s']:.3f} |" for k, v in jev.items()]
    md += ["", "## Notes",
           "- At equal size, Qwen3.5-4B is about 40% slower than Qwen3-4B in this stack (70 vs 50 ms/q). Its hybrid "
           "linear-attention layers are less optimised in transformers; accuracy favours Qwen3.5.",
           "- MiDM-4B-q35 and MiDM-8B-q3-e2 reach the same accuracy (p = 0.69); the 4B is slightly faster (70 vs 82 ms/q) "
           "and needs about half the memory.",
           "- In the cascade, about 59% of dev questions stop at the 0.6B tier (~15 ms/q), so mean latency falls; the "
           "cascade cost will be recomputed from the measured single-question latencies."]
    open(os.path.join(OUT, "speed.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
