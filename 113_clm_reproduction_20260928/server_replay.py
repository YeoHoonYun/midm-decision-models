#!/usr/bin/env python3
"""Replay typed-decisions test rows through a running CLM server: accuracy + latency.

Pass 1 sequential (cold vector cache), pass 2 sequential (warm), pass 3 concurrent
(--concurrency clients, warm). Accuracy uses clm.schema.label_of against gold labels,
the same decisions the offline evaluation scores.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "repo", "src"), os.path.join(HERE, "repo", "train")]
from clm.schema import label_of  # noqa: E402
from adapters import _key_of, _maybe_json  # noqa: E402


def load_rows(workflow):
    import pyarrow.parquet as pq
    return pq.read_table(os.path.join(HERE, "data", "typed-decisions", workflow,
                                      "test-00000-of-00001.parquet")).to_pylist()


def call(url, row, model):
    body = {"state": _maybe_json(row["state"]), "questions": _maybe_json(row["questions"]), "model": model}
    t0 = time.perf_counter()
    r = requests.post(url, json=body, timeout=120)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    gold = _maybe_json(row["gold"])
    hits = n = 0
    for qid, ans in r.json()["answers"].items():
        g = gold.get(qid)
        if g and "label" in g:
            hits += label_of(ans) == _key_of(g["label"]); n += 1
    return ms, float(r.headers.get("X-CLM-Latency-Ms", "nan")), hits, n


def summarize(name, res, wall):
    ms = sorted(r[0] for r in res)
    srv = sorted(r[1] for r in res)
    hits, n = sum(r[2] for r in res), sum(r[3] for r in res)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    out = {"pass": name, "requests": len(res), "decisions": n, "accuracy": hits / n,
           "client_ms_p50": round(q(ms, .5), 1), "client_ms_p95": round(q(ms, .95), 1),
           "server_ms_p50": round(q(srv, .5), 1), "server_ms_p95": round(q(srv, .95), 1),
           "throughput_rps": round(len(res) / wall, 2), "wall_s": round(wall, 1)}
    print(json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8700/v1/systemone")
    ap.add_argument("--model", default="clm-latest")
    ap.add_argument("--workflow", default="all")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    rows = load_rows(a.workflow)
    results = []
    for name in ("sequential_cold", "sequential_warm"):
        t0 = time.time()
        res = [call(a.url, r, a.model) for r in rows]
        results.append(summarize(name, res, time.time() - t0))
    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(lambda r: call(a.url, r, a.model), rows))
    results.append(summarize(f"concurrent{a.concurrency}_warm", res, time.time() - t0))
    if a.output:
        json.dump({"url": a.url, "model": a.model, "workflow": a.workflow, "passes": results},
                  open(a.output, "w"), indent=1)


if __name__ == "__main__":
    main()
