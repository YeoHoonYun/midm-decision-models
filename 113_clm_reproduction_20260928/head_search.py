#!/usr/bin/env python3
"""CLM head search on typed-decisions with row-grouped K-fold CV (test untouched).

Features: --features final (TextCache npz of the final-norm last token, as trained so far)
or --features LAYERDIR:POOL:LAYERS (extract_layers.py output; POOL last|mean, LAYERS like
"18" or "12,18,24" concatenated, or "avg12-24" averaged).

Default mode: K-fold CV over the 1200 train rows (grouped by row id), fixed epochs,
reports mean per-decision accuracy per epoch. --final: train on all 1200 rows for
--epochs (chosen from CV beforehand) with --seeds and evaluate the test split once,
also the seed-ensemble (mean of softmax probabilities).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "repo", "src"), os.path.join(HERE, "repo", "train")]
import adapters  # noqa: E402
import finetune  # noqa: E402
from clm.heads import make_head  # noqa: E402

DATA = os.path.join(HERE, "data", "typed-decisions")


def sha(t):
    return hashlib.sha1(t.encode()).hexdigest()


def load_features(spec):
    if spec == "final":
        z = np.load(os.path.join(HERE, "runs", "emb_cache", "choice_Qwen_Qwen3-8B_2048.npz"))
        return {k: i for i, k in enumerate(z["keys"].tolist())}, z["vecs"].astype(np.float32)
    d, pool, layers = spec.split(":")
    meta = json.load(open(os.path.join(d, "texts.json")))
    arr = np.load(os.path.join(d, f"{pool}.npy"), mmap_mode="r")
    if layers.startswith("avg"):
        lo, hi = map(int, layers[3:].split("-"))
        x = np.asarray(arr[:, lo:hi + 1, :], dtype=np.float32).mean(1)
    else:
        ls = [int(v) for v in layers.split(",")]
        x = np.concatenate([np.asarray(arr[:, l, :], dtype=np.float32) for l in ls], 1)
    return {k: i for i, k in enumerate(meta["keys"])}, x


def examples(split):
    return list(adapters.typed_decision_examples(finetune.load_typed_rows(DATA, split, "all", None)))


def tensors(ex, index):
    kmax = max(len(e.keys) for e in ex)
    st = torch.tensor([index[sha(e.state_text)] for e in ex])
    op = torch.tensor([[index[sha(t)] for t in e.candidates] + [-1] * (kmax - len(e.keys)) for e in ex])
    tg = torch.tensor([e.target + [0.0] * (kmax - len(e.keys)) for e in ex])
    lab = torch.tensor([e.label for e in ex])
    return st, op, tg, lab


def train_eval(X, tr, ev, a, seed, epochs, log_every=True):
    """-> (per-epoch eval acc list, final eval probs [n, kmax]) ; ev may be None."""
    torch.manual_seed(seed); random.seed(seed)
    dev = X.device
    st, op, tg, lab = (t.to(dev) for t in tr)
    if ev is not None:
        ev = tuple(t.to(dev) for t in ev)
    if a.targets == "hard":
        tg = F.one_hot(lab, op.shape[1]).float().to(dev)
    used = torch.unique(torch.cat([st, op[op >= 0]] + ([ev[0], ev[1][ev[1] >= 0]] if ev else [])))
    feats = X
    if a.norm == "zscore":   # statistics from training texts only
        tr_ids = torch.unique(torch.cat([st, op[op >= 0]]))
        mu = feats[tr_ids].mean(0, keepdim=True); sd = feats[tr_ids].std(0, keepdim=True) + 1e-4
        feats = (feats - mu) / sd
    elif a.norm == "l2":
        feats = F.normalize(feats, dim=-1)
    hidden = feats.shape[1]
    kw = dict(activation="gelu", layernorm=True, residual=False, hidden=hidden)
    sh = make_head(a.width, a.depth, a.proj, **kw).to(dev); ah = make_head(a.width, a.depth, a.proj, **kw).to(dev)
    ls = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07), device=dev))
    params = list(sh.parameters()) + list(ah.parameters()) + [ls]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    n = len(st); per = math.ceil(n / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=per * epochs, pct_start=0.1,
                                                anneal_strategy="cos")
    drop = torch.nn.Dropout(a.dropout)

    def scores(s_idx, o_idx, train):
        zs = F.normalize(sh(drop(feats[s_idx]) if train else feats[s_idx]), dim=-1)
        valid = o_idx >= 0
        zo = torch.zeros(*o_idx.shape, a.proj, device=dev)
        zo[valid] = F.normalize(ah(drop(feats[o_idx[valid]]) if train else feats[o_idx[valid]]), dim=-1)
        lg = ls.exp().clamp(max=100) * torch.einsum("bh,bkh->bk", zs, zo)
        return lg.masked_fill(~valid, -1e4)

    def infonce(idx):
        oi, t = op[idx], tg[idx]
        valid = oi >= 0
        pool, col = torch.unique(oi[valid], return_inverse=True)
        cols = torch.zeros_like(oi); cols[valid] = col
        T = torch.zeros(len(idx), len(pool), device=dev).scatter_add_(1, cols, t * valid)
        zq = F.normalize(sh(drop(feats[st[idx]])), dim=-1)
        zc = F.normalize(ah(drop(feats[pool])), dim=-1)
        lg = ls.exp().clamp(max=100) * zq @ zc.t()
        fwd = -(T * F.log_softmax(lg, 1)).sum(1).mean()
        w = T.t(); keep = w.sum(1) > 0; w = w[keep] / w[keep].sum(1, keepdim=True)
        bwd = -(w * F.log_softmax(lg.t()[keep], 1)).sum(1).mean()
        return (fwd + bwd) / 2

    @torch.no_grad()
    def evaluate():
        sh.eval(); ah.eval()
        lg = scores(ev[0], ev[1], False)
        sh.train(); ah.train()
        p = lg.softmax(-1)
        return (p.argmax(-1) == ev[3]).float().mean().item(), p

    g = torch.Generator().manual_seed(seed)
    accs = []
    p = None
    for ep in range(epochs):
        for idx in torch.randperm(n, generator=g).to(dev).split(a.batch):
            if a.loss == "infonce":
                loss = infonce(idx)
            elif a.loss == "softce":
                loss = -(tg[idx] * F.log_softmax(scores(st[idx], op[idx], True), -1)).sum(-1).mean()
            else:  # mix
                loss = 0.5 * infonce(idx) - 0.5 * (tg[idx] * F.log_softmax(scores(st[idx], op[idx], True), -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if ev is not None and (log_every or ep == epochs - 1):
            acc, p = evaluate(); accs.append(acc)
    return accs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="final")
    ap.add_argument("--norm", choices=["none", "l2", "zscore"], default="l2")
    ap.add_argument("--loss", choices=["infonce", "softce", "mix"], default="infonce")
    ap.add_argument("--targets", choices=["soft", "hard"], default="soft")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--proj", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--final", action="store_true", help="train on all train rows, evaluate TEST once")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    index, X = load_features(a.features)
    X = torch.from_numpy(X).to(a.device)
    train_ex = examples("train")
    res = {"args": vars(a)}
    if not a.final:
        rows = sorted({e.group for e in train_ex})
        random.Random(0).shuffle(rows)
        fold_of = {r: i % a.folds for i, r in enumerate(rows)}
        curves = []
        for k in range(a.folds):
            tr = [e for e in train_ex if fold_of[e.group] != k]
            ev = [e for e in train_ex if fold_of[e.group] == k]
            for s in a.seeds:
                accs, _ = train_eval(X, tensors(tr, index), tensors(ev, index), a, s, a.epochs)
                curves.append(accs)
        mean = np.mean(curves, 0).tolist()
        best = int(np.argmax(mean))
        res.update(cv_curve=mean, best_epoch=best + 1, best_cv_acc=mean[best], last_cv_acc=mean[-1],
                   fold_seed_curves=curves)
        print(f"[cv] {a.features} {a.norm} {a.loss} w{a.width} d{a.depth} lr{a.lr} do{a.dropout} "
              f"best ep {best + 1}: {mean[best]:.4f} last {mean[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    else:
        test_ex = examples("test")
        te = tensors(test_ex, index)
        probs, per_seed = [], []
        for s in a.seeds:
            accs, p = train_eval(X, tensors(train_ex, index), te, a, s, a.epochs, log_every=False)
            per_seed.append(accs[-1]); probs.append(p)
        ens = torch.stack(probs).mean(0)
        ens_acc = (ens.argmax(-1).cpu() == te[3]).float().mean().item()
        res.update(test_acc_per_seed=per_seed, test_acc_seed_mean=float(np.mean(per_seed)),
                   test_acc_ensemble=ens_acc, n_test=len(test_ex))
        np.save(a.out.replace(".json", "_probs.npy"), ens.cpu().numpy())
        print(f"[final] {a.features} per-seed {per_seed} mean {np.mean(per_seed):.4f} ensemble {ens_acc:.4f}",
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
