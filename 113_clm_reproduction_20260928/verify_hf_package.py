#!/usr/bin/env python3
"""Load a MiDM release folder with ONLY its own midm.py, then (1) run the public predict() API on a demo
request and (2) re-score a dev suite through the package path. It should match the training-repo
eval of the same checkpoint (small numeric drift allowed).

    python verify_hf_package.py ../../research_topics/clm_decision_heads_novelty_20260928/huggingface/MiDM-4B-q35-e1-bx --suite td_holdout --expect 0.8333
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "repo", "src"), os.path.join(HERE, "repo", "train")]

ap = argparse.ArgumentParser()
ap.add_argument("package")
ap.add_argument("--suite", default="td_holdout")
ap.add_argument("--expect", type=float, default=None)
ap.add_argument("--dtype", default="bfloat16")
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("midm_pkg", os.path.join(a.package, "midm.py"))
midm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(midm)
t0 = time.time()
m = midm.MiDM.from_pretrained(a.package, dtype=a.dtype)
print(f"[verify] loaded in {time.time() - t0:.0f}s")

demo = m.predict("Refund policy: a receipt is required and the purchase must be within 30 days. "
                 "The customer bought the item 12 days ago and has no receipt.",
                 {"permitted": {"type": "noul", "instructions": "Is a refund permitted under the policy?"},
                  "action": {"type": "choice", "instructions": "What should the agent do?",
                             "criteria": {"refund": "Issue the refund.", "deny": "Politely deny the refund.",
                                          "escalate": "Escalate to a human."}}})
print("[verify] demo:", json.dumps(demo))

import decision_data  # noqa: E402
ex = decision_data.load(a.suite)
hit, t0 = 0, time.time()
lat = []
for e in ex:
    ids, pos = m._build(e.state_text, e.keys, e.candidates)
    s = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=getattr(torch, a.dtype)):
        h = m.model(input_ids=torch.tensor([ids], device=m.device)).last_hidden_state
    p = m.head(h[0, pos].float()).squeeze(-1).softmax(-1)
    torch.cuda.synchronize()
    lat.append((time.perf_counter() - s) * 1000)
    hit += int(int(p.argmax()) == e.label)
acc = hit / len(ex)
lat.sort()
res = {"package": a.package, "suite": a.suite, "n": len(ex), "acc": acc, "expect": a.expect,
       "match": None if a.expect is None else abs(acc - a.expect) <= 0.005,
       "latency_ms_p50_single_question": lat[len(lat) // 2], "latency_ms_p95": lat[int(len(lat) * 0.95)],
       "demo": demo}
print("[verify]", json.dumps({k: v for k, v in res.items() if k != "demo"}))
json.dump(res, open(os.path.join(a.package, "..", f"verify_{os.path.basename(a.package)}.json"), "w"), indent=1)
