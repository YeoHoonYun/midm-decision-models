#!/usr/bin/env python3
"""Embed every unique text of every decision_data suite with one encoder into the shared
TextCache (runs/emb_cache/choice_<model>_2048.npz, final-norm last token, L2), reusing entries
already cached. Same recipe as run_choice_hf.py.

    python embed_all.py --model Qwen/Qwen3-4B --device-map cuda:0
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "repo", "src"), os.path.join(HERE, "repo", "train")]
import decision_data  # noqa: E402
import finetune  # noqa: E402
from embed_utils import Recipe  # noqa: E402
from run_choice_hf import HFBackend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dtype", default="float16")
ap.add_argument("--device-map", default="cuda:0")
ap.add_argument("--max-gpu-gib", type=float, default=None)
ap.add_argument("--token-budget", type=int, default=8192)
ap.add_argument("--chunk", type=int, default=4000, help="texts per cache save")
a = ap.parse_args()

cache = finetune.TextCache(os.path.join(HERE, "runs", "emb_cache", f"choice_{finetune._slug(a.model)}_2048.npz"))
texts = []
for s in decision_data.ALL:
    for e in decision_data.load(s):
        texts += [e.state_text, *e.candidates]
todo = cache.missing(texts)
print(f"[embed-all] {a.model}: {len(set(texts))} unique texts, {len(todo)} to embed", flush=True)
if todo:
    rec = Recipe(a.model, 2048)
    be = HFBackend(a.model, a.dtype, a.device_map, a.token_budget, a.max_gpu_gib)
    for i in range(0, len(todo), a.chunk):
        part = todo[i:i + a.chunk]
        cache.add(part, be.embed([rec.text_ids(t, keep="tail") for t in part]))
        print(f"[embed-all] saved {min(i + a.chunk, len(todo))}/{len(todo)}", flush=True)
    print(f"[embed-all] totals {be.stats}", flush=True)
