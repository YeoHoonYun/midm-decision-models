#!/usr/bin/env python3
"""OpenAI-compatible /v1/embeddings server backed by transformers (vLLM stand-in on Windows).

Matches the training recipe in run_choice_hf.py: tokens without special tokens, tail kept
(``truncate_prompt_tokens`` keeps the last N, like vLLM), final-layer last-token state,
L2-normalised, returned as float32 (base64 when ``encoding_format`` is ``base64``).

    python embed_server.py --model Qwen/Qwen3-0.6B --port 8090 --dtype float16
"""
from __future__ import annotations

import argparse
import base64
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request


def build(model_name: str, dtype: str, device: str, max_len: int, served_name: str) -> FastAPI:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=getattr(torch, dtype)).to(device).eval()
    lock = threading.Lock()   # one forward at a time on the GPU
    app = FastAPI(title="embed-server")
    stats = {"requests": 0, "texts": 0, "tokens": 0}

    @torch.no_grad()
    def embed(id_lists: list[list[int]]) -> np.ndarray:
        L = max(len(s) for s in id_lists)
        ids = torch.zeros((len(id_lists), L), dtype=torch.long)
        att = torch.zeros((len(id_lists), L), dtype=torch.long)
        for r, s in enumerate(id_lists):
            ids[r, :len(s)] = torch.tensor(s)
            att[r, :len(s)] = 1
        with lock:
            h = model(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
            v = h[torch.arange(len(id_lists), device=h.device), (att.sum(1) - 1).to(h.device)].float().cpu().numpy()
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "root": model_name}]}

    @app.get("/stats")
    def get_stats():
        return stats

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        body = await request.json()
        inp = body.get("input")
        if isinstance(inp, str) or (isinstance(inp, list) and inp and isinstance(inp[0], int)):
            inp = [inp]
        if not isinstance(inp, list) or not inp:
            raise HTTPException(422, "input must be a string, list of strings or list of token-id lists")
        cap = min(int(body.get("truncate_prompt_tokens") or max_len), max_len - 1)
        id_lists = []
        for x in inp:
            ids = x if isinstance(x, list) else tok(x, add_special_tokens=False)["input_ids"]
            ids = list(ids)[-cap:] or tok(" ", add_special_tokens=False)["input_ids"]
            id_lists.append(ids)
        t0 = time.perf_counter()
        vecs = embed(id_lists)
        n_tok = sum(map(len, id_lists))
        stats["requests"] += 1; stats["texts"] += len(id_lists); stats["tokens"] += n_tok
        b64 = body.get("encoding_format") == "base64"
        data = [{"object": "embedding", "index": i,
                 "embedding": base64.b64encode(v.astype(np.float32).tobytes()).decode() if b64 else v.tolist()}
                for i, v in enumerate(vecs)]
        return {"object": "list", "model": served_name, "data": data,
                "usage": {"prompt_tokens": n_tok, "total_tokens": n_tok},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    import uvicorn
    app = build(a.model, a.dtype, a.device, a.max_len, a.served_model_name or a.model.split("/")[-1].lower())
    print(f"[embed-server] {a.model} ({a.dtype}) on {a.device} at http://{a.host}:{a.port}/v1/embeddings", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
