#!/usr/bin/env python3
"""Mock System One backend for router tests (no GPU): answers every question with a fixed top-option
probability, so the cascade's escalation can be exercised.

    python mock_decider.py --port 8710 --top 0.55     # low confidence -> escalates
    python mock_decider.py --port 8712 --top 0.95     # confident
"""
import argparse
import os
import sys

from fastapi import FastAPI, Request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "113_clm_reproduction_20260928", "repo", "src"))
from clm.client import question_to_dict  # noqa: E402
from clm.schema import answer_from_probs, build_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--top", type=float, default=0.9)
a = ap.parse_args()
app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/systemone")
async def systemone(request: Request):
    body = await request.json()
    qs = {k: question_to_dict(q) for k, q in body["questions"].items()}
    out = {}
    for qid, (_, keys, _) in build_pairs(body["state"], qs).items():
        rest = (1 - a.top) / (len(keys) - 1)
        out[qid] = answer_from_probs(qs[qid], keys, [a.top] + [rest] * (len(keys) - 1))
    return {"model": f"mock{a.port}", "answers": out}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
