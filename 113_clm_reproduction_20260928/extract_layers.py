#!/usr/bin/env python3
"""Per-layer features for every typed-decisions text (train + test, workflow 'all').

One forward pass per batch with hooks on each decoder layer; for every layer keeps the
last-token vector and the attention-masked mean over tokens. Same token recipe as
training (embed_utils.Recipe.text_ids keep='tail', max_len 2048), right padding.

Output (fp16): <out>/last.npy [N, L+1, H], <out>/mean.npy [N, L+1, H] and texts.json
(sha1 keys in row order). Index 0..L-1 = decoder layer outputs (pre final norm),
index L = final-norm output (= the last_hidden_state the heads were trained on).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "repo", "src"), os.path.join(HERE, "repo", "train")]
import adapters  # noqa: E402
import finetune  # noqa: E402
from embed_utils import Recipe  # noqa: E402


def all_texts():
    ex = []
    for split in ("train", "test"):
        ex += list(adapters.typed_decision_examples(
            finetune.load_typed_rows(os.path.join(HERE, "data", "typed-decisions"), split, "all", None)))
    return list(dict.fromkeys(t for e in ex for t in (e.state_text, *e.candidates)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-gpu-gib", type=float, default=7)
    ap.add_argument("--token-budget", type=int, default=8192)
    a = ap.parse_args()
    from transformers import AutoModel
    texts = all_texts()
    rec = Recipe(a.model, 2048)
    ids = [rec.text_ids(t, keep="tail") for t in texts]
    m = AutoModel.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto",
                                  max_memory={0: f"{a.max_gpu_gib}GiB", "cpu": "48GiB"}).eval()
    layers = m.layers
    L, H = len(layers), m.config.hidden_size
    os.makedirs(a.out, exist_ok=True)
    last = np.lib.format.open_memmap(os.path.join(a.out, "last.npy"), "w+", np.float16, (len(texts), L + 1, H))
    mean = np.lib.format.open_memmap(os.path.join(a.out, "mean.npy"), "w+", np.float16, (len(texts), L + 1, H))
    cur = {}

    def hook(i):
        def f(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            mask, lastpos, rows = cur["mask"].to(h.device), cur["last"].to(h.device), cur["rows"]
            hf = h.float()
            cur["last_v"][:, i] = hf[torch.arange(h.shape[0], device=h.device), lastpos].cpu()
            cur["mean_v"][:, i] = ((hf * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)).cpu()
        return f

    hs = [layers[i].register_forward_hook(hook(i)) for i in range(L)]
    order = sorted(range(len(ids)), key=lambda k: len(ids[k]))
    t0, i, last_log = time.time(), 0, 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * len(ids[order[j]]) <= a.token_budget:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        Lx = max(len(ids[k]) for k in idx)
        x = torch.zeros((len(idx), Lx), dtype=torch.long)
        mask = torch.zeros((len(idx), Lx))
        for r, k in enumerate(idx):
            x[r, :len(ids[k])] = torch.tensor(ids[k]); mask[r, :len(ids[k])] = 1
        cur.update(mask=mask, last=(mask.sum(1) - 1).long(), rows=idx,
                   last_v=torch.zeros(len(idx), L + 1, H), mean_v=torch.zeros(len(idx), L + 1, H))
        dev = next(m.parameters()).device
        out = m(input_ids=x.to(dev), attention_mask=mask.long().to(dev)).last_hidden_state.float()
        mk = mask.to(out.device)
        cur["last_v"][:, L] = out[torch.arange(len(idx), device=out.device), cur["last"].to(out.device)].cpu()
        cur["mean_v"][:, L] = ((out * mk[..., None]).sum(1) / mk.sum(1, keepdim=True)).cpu()
        if not (torch.isfinite(cur["last_v"]).all() and torch.isfinite(cur["mean_v"]).all()):
            raise FloatingPointError("non-finite features")
        last[idx] = cur["last_v"].half().numpy()
        mean[idx] = cur["mean_v"].half().numpy()
        i = j
        if time.time() - last_log > 60:
            last_log = time.time()
            print(f"[layers] {i}/{len(order)} texts, {time.time() - t0:.0f}s", flush=True)
    for h in hs:
        h.remove()
    last.flush(); mean.flush()
    json.dump({"model": a.model, "dtype": a.dtype, "layers": L, "hidden": H,
               "keys": [hashlib.sha1(t.encode()).hexdigest() for t in texts]},
              open(os.path.join(a.out, "texts.json"), "w"))
    print(f"[layers] done {len(texts)} texts x {L + 1} layers in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
