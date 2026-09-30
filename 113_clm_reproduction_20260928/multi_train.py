#!/usr/bin/env python3
"""Multi-benchmark CLM decision head on a frozen encoder (cached TextCache embeddings).

Selection mode (default): train on --train suites minus a 10% row-group holdout of td_train,
score on --select suites (kev_dev + the td holdout). Final mode (--final): train on the full
--train suites and report accuracy / Brier / ECE on --eval suites (each read once).

Brier = mean over decisions of sum_k (p_k - y_k)^2 (JevBench definition); ECE = top-label,
10 equal-width bins. With --save, the z-score is folded into the first Linear of each head
(W' = W / sd, b' = b - W' mu), so the checkpoint runs unchanged in clm.server on L2 embeddings.
"""
from __future__ import annotations

import argparse
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
sys.path[:0] = [HERE, os.path.join(HERE, "repo", "src"), os.path.join(HERE, "repo", "train")]
import decision_data  # noqa: E402
import finetune  # noqa: E402
from clm.heads import make_head  # noqa: E402
from head_search import sha  # noqa: E402


def load_cache(model):
    z = np.load(os.path.join(HERE, "runs", "emb_cache", f"choice_{finetune._slug(model)}_2048.npz"))
    return {k: i for i, k in enumerate(z["keys"].tolist())}, torch.from_numpy(z["vecs"].astype(np.float32))


def pack(ex, index, kmax):
    st = torch.tensor([index[sha(e.state_text)] for e in ex])
    op = torch.tensor([[index[sha(t)] for t in e.candidates] + [-1] * (kmax - len(e.keys)) for e in ex])
    tg = torch.tensor([e.target + [0.0] * (kmax - len(e.keys)) for e in ex])
    lab = torch.tensor([e.label for e in ex])
    return st, op, tg, lab


def metrics(p, lab, nopt):
    p = p.cpu(); lab = lab.cpu()
    y = F.one_hot(lab, p.shape[1]).float()
    conf, pred = p.max(1)
    correct = (pred == lab).float()
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.any():
            ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    return {"n": len(lab), "acc": correct.mean().item(), "brier": ((p - y) ** 2).sum(1).mean().item(), "ece": ece}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--train", nargs="+", default=["td_train", "kev_train"])
    ap.add_argument("--select", nargs="+", default=["kev_dev"])
    ap.add_argument("--eval", nargs="+", default=["td_test", "kev_dev", "kevT_dev", "kevT9_dev",
                                                   "jb_original", "jb_easy", "jb_hard"])
    ap.add_argument("--norm", choices=["zscore", "l2"], default="zscore")
    ap.add_argument("--loss", choices=["infonce", "softce", "mix"], default="infonce")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--proj", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--td-weight", type=float, default=1.0, help="sampling weight of td_train vs other suites")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--save", default=None, help="final mode: write a clm.server checkpoint here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    t0 = time.time()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    index, X = load_cache(a.encoder)
    X = X.to(dev)

    train_ex = []
    for s in a.train:
        ex = decision_data.load(s)
        for e in ex:
            e.meta["suite"] = s
        train_ex += ex
    sel = {}
    if not a.final:   # hold out 10% of td_train row groups for selection
        groups = sorted({e.group for e in train_ex if e.meta["suite"] == "td_train"})
        random.Random(0).shuffle(groups)
        hold = set(groups[:len(groups) // 10])
        sel["td_holdout"] = [e for e in train_ex if e.group in hold]
        train_ex = [e for e in train_ex if e.group not in hold]
        for s in a.select:
            sel[s] = decision_data.load(s)
    else:
        for s in a.eval:
            sel[s] = decision_data.load(s)
    kmax = max(len(e.keys) for e in train_ex + [e for v in sel.values() for e in v])
    st, op, tg, lab = (t.to(dev) for t in pack(train_ex, index, kmax))
    evals = {k: tuple(t.to(dev) for t in pack(v, index, kmax)) for k, v in sel.items()}
    w = torch.tensor([a.td_weight if e.meta["suite"] == "td_train" else 1.0 for e in train_ex])

    tr_ids = torch.unique(torch.cat([st, op[op >= 0]]))
    if a.norm == "zscore":
        mu = X[tr_ids].mean(0); sd = X[tr_ids].std(0) + 1e-4
    else:
        mu = torch.zeros(X.shape[1], device=dev); sd = torch.ones(X.shape[1], device=dev)
    feats = (X - mu) / sd
    kw = dict(activation="gelu", layernorm=True, residual=False, hidden=X.shape[1])
    sh = make_head(a.width, a.depth, a.proj, **kw).to(dev); ah = make_head(a.width, a.depth, a.proj, **kw).to(dev)
    ls = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07), device=dev))
    params = list(sh.parameters()) + list(ah.parameters()) + [ls]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    n = len(st); per = math.ceil(n / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=per * a.epochs, pct_start=0.1,
                                                anneal_strategy="cos")
    drop = torch.nn.Dropout(a.dropout)

    def logits(s_idx, o_idx, train):
        zs = F.normalize(sh(drop(feats[s_idx]) if train else feats[s_idx]), dim=-1)
        valid = o_idx >= 0
        zo = torch.zeros(*o_idx.shape, a.proj, device=dev)
        zo[valid] = F.normalize(ah(drop(feats[o_idx[valid]]) if train else feats[o_idx[valid]]), dim=-1)
        return (ls.exp().clamp(max=100) * torch.einsum("bh,bkh->bk", zs, zo)).masked_fill(~valid, -1e4)

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
        wt = T.t(); keep = wt.sum(1) > 0; wt = wt[keep] / wt[keep].sum(1, keepdim=True)
        return (fwd - (wt * F.log_softmax(lg.t()[keep], 1)).sum(1).mean()) / 2

    @torch.no_grad()
    def evaluate():
        sh.eval(); ah.eval()
        out = {k: metrics(logits(v[0], v[1], False).softmax(-1), v[3], None) for k, v in evals.items()}
        sh.train(); ah.train()
        return out

    g = torch.Generator().manual_seed(a.seed)
    history = []
    for ep in range(1, a.epochs + 1):
        order = torch.multinomial(w, n, replacement=a.td_weight != 1.0, generator=g) if a.td_weight != 1.0 \
            else torch.randperm(n, generator=g)
        for idx in order.to(dev).split(a.batch):
            if a.loss == "infonce":
                loss = infonce(idx)
            elif a.loss == "softce":
                loss = -(tg[idx] * F.log_softmax(logits(st[idx], op[idx], True), -1)).sum(-1).mean()
            else:
                loss = 0.5 * infonce(idx) - 0.5 * (tg[idx] * F.log_softmax(logits(st[idx], op[idx], True), -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if not a.final and (ep % 5 == 0 or ep == a.epochs):
            m = evaluate()
            history.append({"epoch": ep, **{k: v["acc"] for k, v in m.items()}})
    res = {"args": vars(a), "minutes": None}
    if a.final:
        res["eval"] = evaluate()
        line = " ".join(f"{k}={v['acc']:.4f}" for k, v in res["eval"].items())
    else:
        res["history"] = history
        score = lambda h: np.mean([h[k] for k in sel])
        best = max(history, key=score)
        res.update(select_score=score(history[-1]), best_epoch=best["epoch"], best_select_score=score(best),
                   last=history[-1])
        line = f"select(last)={score(history[-1]):.4f} " + " ".join(f"{k}={history[-1][k]:.4f}" for k in sel) + \
               f" | best ep {best['epoch']} {score(best):.4f}"
    res["minutes"] = round((time.time() - t0) / 60, 2)
    print(f"[multi] {a.encoder} {a.norm} {a.loss} ep{a.epochs} w{a.width} tdw{a.td_weight} | {line} "
          f"({res['minutes']} min)", flush=True)
    if a.save and a.final:
        from clm.heads import HIDDEN  # noqa: F401
        blob = {}
        for name, head in (("state_head", sh), ("action_head", ah)):
            sd_ = {k: v.detach().clone().cpu() for k, v in head.state_dict().items()}
            W, b = sd_["inp.weight"], sd_["inp.bias"]
            W2 = W / sd.cpu()[None, :]
            sd_["inp.weight"], sd_["inp.bias"] = W2, b - W2 @ mu.cpu()
            blob[name] = sd_
        blob.update(logit_scale=ls.detach().cpu(), cfg={"width": a.width, "depth": a.depth, "projection_dim": a.proj,
                    "activation": "gelu", "layernorm": True, "residual": False, "hidden_size": X.shape[1],
                    "embed_model": a.encoder, "train": a.train, "epochs": a.epochs, "norm_folded": a.norm})
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        torch.save(blob, a.save)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
