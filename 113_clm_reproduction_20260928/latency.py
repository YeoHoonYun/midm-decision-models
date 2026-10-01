#!/usr/bin/env python3
"""Per-question latency of the cascade tiers on one GPU (dev questions only; no test reads).

    PTR_BF16=1 python latency.py --n 300     # -> results/latency/latency.json, latency.md

Each tier is loaded alone, warmed up, then scores --n td_holdout questions one at a time (batch 1),
as the router does. Reported: p50 / p90 / mean ms and the cost ratio relative to the smallest tier.
"""
import argparse
import gc
import json
import os
import statistics
import time

import torch

import decision_data
import pointer_lora as P

HERE = os.path.dirname(os.path.abspath(__file__))
TIERS = [("MiDM-0.6B-q3-e1", "Qwen/Qwen3-0.6B", "runs/ptr0p6b/final"),
         ("MiDM-1.7B-q3-e1", "Qwen/Qwen3-1.7B", "runs/ptr1p7b/final"),
         ("MiDM-4B-q35-e1", "Qwen/Qwen3.5-4B-Base", "runs/B/qwen35_4b_1ep/final"),
         ("MiDM-8B-q3-e2", "Qwen/Qwen3-8B", "runs/B/qwen3_8b_2ep/final")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    ex = decision_data.load("td_holdout")[:a.n]
    out = {"gpu": torch.cuda.get_device_name(0), "dtype": str(P.DTYPE), "n": len(ex), "tiers": {}}
    for name, base, ckpt in TIERS:
        if not os.path.isdir(os.path.join(HERE, ckpt)):
            print(f"[lat] skip {name}: {ckpt} missing", flush=True)
            continue
        tok, model, head = P.load_model(base, ckpt=os.path.join(HERE, ckpt), train=False)
        model.eval()
        enc = P.Encoder(tok, 1024)
        items = [enc.build(e.state_text, e.keys, e.candidates, list(range(len(e.keys)))) for e in ex]
        with torch.no_grad():
            for it in items[:10]:   # warm-up
                P.forward(model, head, [it])
            torch.cuda.synchronize()
            ms = []
            for it in items:
                t = time.perf_counter()
                P.forward(model, head, [it])
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t) * 1000)
        ms.sort()
        out["tiers"][name] = {"p50_ms": ms[len(ms) // 2], "p90_ms": ms[int(len(ms) * 0.9)], "mean_ms": statistics.mean(ms),
                              "max_mem_gib": torch.cuda.max_memory_allocated() / 2**30}
        print(f"[lat] {name}: {out['tiers'][name]}", flush=True)
        del model, head, tok
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    first = next(iter(out["tiers"].values()))["mean_ms"]
    for v in out["tiers"].values():
        v["cost_ratio"] = v["mean_ms"] / first
    os.makedirs(os.path.join(HERE, "results", "latency"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "latency", "latency.json"), "w"), indent=1)
    md = [f"# Per-question latency ({out['gpu']}, {out['dtype']}, batch 1, n={out['n']} td_holdout)", "",
          "| tier | p50 ms | p90 ms | mean ms | cost ratio | peak GiB |", "|---|---|---|---|---|---|"]
    md += [f"| {k} | {v['p50_ms']:.1f} | {v['p90_ms']:.1f} | {v['mean_ms']:.1f} | {v['cost_ratio']:.2f} | {v['max_mem_gib']:.1f} |"
           for k, v in out["tiers"].items()]
    open(os.path.join(HERE, "results", "latency", "latency.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
