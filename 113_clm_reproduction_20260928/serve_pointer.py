#!/usr/bin/env python3
"""System One API (/v1/systemone, /v1/models, /health) for the QLoRA pointer model.

All questions of a request go through the model in one padded batch; answers are built
with the repo's clm.schema (build_pairs / answer_from_probs), so clients written for
TypeSafe or clm-serve (e.g. examples/t_rex) work unchanged.

    python serve_pointer.py --model Qwen/Qwen3-4B --ckpt runs/ptr4b/final --port 8700
"""
import argparse
import os
import sys
import threading
import time

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "repo", "src")]
from clm.client import question_to_dict  # noqa: E402
from clm.schema import answer_from_probs, build_pairs  # noqa: E402
from pointer_lora import Encoder, forward, load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-4B")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--name", default="ptr-latest")
ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8700)
a = ap.parse_args()

tok, model, head = load_model(a.model, ckpt=a.ckpt, train=False)
model.eval()
enc = Encoder(tok, a.max_len)
lock = threading.Lock()
app = FastAPI(title="pointer decision model")


@app.get("/health")
def health():
    return {"ok": True, "embedder": True, "models": [a.name]}


@app.get("/v1/models")
def models():
    return {"models": [{"name": a.name, "description": f"QLoRA pointer model on {a.model}", "release_date": "2026-09-28"}]}


@app.post("/v1/systemone")
async def systemone(request: Request):
    body = await request.json()
    if not isinstance(body, dict) or "state" not in body or not isinstance(body.get("questions"), dict):
        raise HTTPException(422, "body must be {state, model, questions}")
    t0 = time.perf_counter()
    try:
        qs = {k: question_to_dict(q) for k, q in body["questions"].items()}
        pairs = build_pairs(body["state"], qs)
        items = [enc.build(st, keys, cands, list(range(len(keys)))) for st, keys, cands in pairs.values()]
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(422, f"invalid request: {e}") from e
    temp = float(body.get("temperature", 1.0))
    with lock, torch.no_grad():
        logits = forward(model, head, items)
    answers = {}
    for (qid, (_, keys, _)), lg in zip(pairs.items(), logits):
        answers[qid] = answer_from_probs(qs[qid], keys, (lg / temp).softmax(-1).tolist())
    ntok = sum(len(i) for i, _ in items)
    return JSONResponse({"model": a.name, "answers": answers,
                         "usage": {"billing_units": len(qs), "input_tokens": ntok, "output_tokens": 0}},
                        headers={"X-CLM-Latency-Ms": f"{(time.perf_counter() - t0) * 1000:.1f}"})


if __name__ == "__main__":
    import uvicorn
    print(f"[ptr-serve] {a.model} + {a.ckpt} at http://{a.host}:{a.port}/v1/systemone", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
