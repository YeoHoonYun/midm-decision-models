#!/usr/bin/env python3
"""Run repo/train/finetune.py --task choice with a transformers last-token embedder.

The repo embeds with vLLM (Linux only). This wrapper leaves the repo untouched and
swaps in an in-process transformers backend that follows the same recipe: token ids
from ``embed_utils.Recipe`` (no special tokens, tail kept), final-layer hidden state
of the last token, L2-normalised. It also sizes the heads to the encoder's hidden
width so smaller Qwen3 encoders can be used.

    python run_choice_hf.py --embed-model Qwen/Qwen3-0.6B --dtype bfloat16 -- \
        --task choice --data data/typed-decisions --workflow all --out-dir runs/x
Everything after ``--`` goes to finetune.py unchanged; --embed-model is forwarded.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "preprocessing"), os.path.join(REPO, "train")]


class HFBackend:
    def __init__(self, model: str, dtype: str, device_map: str, token_budget: int, max_gpu_gib: float | None = None):
        from transformers import AutoConfig, AutoModel
        self.model_name = model
        self.dtype = getattr(torch, dtype)
        self.device_map = device_map
        self.token_budget = token_budget
        self.max_gpu_gib = max_gpu_gib
        self.hidden = AutoConfig.from_pretrained(model).hidden_size
        self._m = None
        self.stats = {"texts": 0, "tokens": 0, "seconds": 0.0}

    def _load(self):
        from transformers import AutoModel
        dm = self.device_map if self.device_map != "cpu" else None
        kw = {}
        if self.max_gpu_gib:  # cap GPU 0 and offload the remaining layers to CPU RAM
            kw["max_memory"] = {0: f"{self.max_gpu_gib}GiB", "cpu": "48GiB"}
        self._m = AutoModel.from_pretrained(self.model_name, dtype=self.dtype, device_map=dm, **kw)
        print(f"[hf-embed] device map: {sorted(set(map(str, getattr(self._m, 'hf_device_map', {}).values())))}",
              flush=True)
        self._m.eval()

    @torch.no_grad()
    def embed(self, id_lists: list[list[int]]) -> np.ndarray:
        if self._m is None:
            self._load()
        dev = next(self._m.parameters()).device
        order = sorted(range(len(id_lists)), key=lambda i: len(id_lists[i]))
        out = np.zeros((len(id_lists), self.hidden), dtype=np.float32)
        t0 = time.time()
        i = 0
        while i < len(order):
            # length-sorted batches under a padded-token budget
            j = i
            while j < len(order) and (j - i + 1) * len(id_lists[order[j]]) <= self.token_budget:
                j += 1
            j = max(j, i + 1)
            idx = order[i:j]
            L = max(len(id_lists[k]) for k in idx)
            ids = torch.zeros((len(idx), L), dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, k in enumerate(idx):  # right padding; pool at each row's own last token
                s = id_lists[k]
                ids[r, :len(s)] = torch.tensor(s)
                att[r, :len(s)] = 1
            h = self._m(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state
            last = att.sum(1) - 1
            v = h[torch.arange(len(idx), device=h.device), last.to(h.device)].float().cpu().numpy()
            if not np.isfinite(v).all():
                raise FloatingPointError(f"non-finite embeddings from {self.model_name} ({self.dtype})")
            out[idx] = v
            self.stats["tokens"] += int(att.sum())
            i = j
            if time.time() - getattr(self, "_last_log", 0) > 60:
                self._last_log = time.time()
                print(f"[hf-embed] progress {i}/{len(order)} texts, {time.time() - t0:.0f}s", flush=True)
        self.stats["texts"] += len(id_lists)
        self.stats["seconds"] += time.time() - t0
        print(f"[hf-embed] {self.model_name}: {len(id_lists)} texts, {self.stats['tokens']} tokens, "
              f"{self.stats['seconds']:.1f}s", flush=True)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="cuda:0", help="e.g. cuda:0, auto, cpu")
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--max-gpu-gib", type=float, default=None, help="with --device-map auto: GPU cap, rest on CPU")
    ours, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    import embed_utils
    import clm.heads as heads
    backend = HFBackend(ours.embed_model, ours.dtype, ours.device_map, ours.token_budget, ours.max_gpu_gib)
    embed_utils.make_backend = lambda *a, **k: backend
    heads.HIDDEN = backend.hidden
    heads.make_head = functools.partial(heads.make_head, hidden=backend.hidden)

    import finetune
    sys.argv = ["finetune.py", *rest, "--embed-model", ours.embed_model]
    finetune.main()
    print(f"[hf-embed] totals {backend.stats}", flush=True)


if __name__ == "__main__":
    main()
