#!/usr/bin/env python3
"""Automatic unknown-domain (OOD) gate for decider-auto (DECISIONS.md D1a follow-up). CPU only.

Per question: OOD score from the state-text embedding (cached, exp 113 runs/emb_cache) against the
training reference set = the pointer models' training questions (td_train minus td_holdout groups,
plus kev_train). Gate: score <= threshold -> cascade (D1 thresholds), else -> top tier directly.
Threshold = a quantile of the score on in-distribution DEV only (kev_dev + td_holdout). Nothing is
tuned on td_test / kevT_* / jb_*; those are read only to report.

    $env:PYTHONPATH=''; $env:CUDA_VISIBLE_DEVICES=''
    ..\\..\\113_clm_reproduction_20260928\\.venv\\Scripts\\python.exe ood_gate.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E113 = os.path.normpath(os.path.join(HERE, "..", "..", "113_clm_reproduction_20260928"))
sys.path.insert(0, E113)
import decision_data  # noqa: E402

PROBS = os.path.join(E113, "results", "probs")
CACHE = os.path.join(E113, "runs", "emb_cache")
ENCODERS = {"qwen3-emb-0.6b": "Qwen_Qwen3-Embedding-0.6B", "qwen3-0.6b": "Qwen_Qwen3-0.6B",
            "qwen3-1.7b": "Qwen_Qwen3-1.7B", "qwen3-4b": "Qwen_Qwen3-4B"}
ENC_COST = {"qwen3-emb-0.6b": 1, "qwen3-0.6b": 1, "qwen3-1.7b": 2, "qwen3-4b": 6}   # provisional, like tiers
COST = {"qwen3-0.6b": 1, "qwen3-1.7b": 2, "qwen3-4b": 6, "qwen3-8b": 12}
CASCADES = [("qwen3-1.7b", "qwen3-8b", 0.58), ("qwen3-0.6b", "qwen3-4b", 0.62)]   # D1, not re-tuned
DEV = ["kev_dev", "td_holdout"]
EVAL = ["td_test", "kevT_dev", "kevT_test", "kevT9_dev", "jb_original", "jb_easy", "jb_hard"]
ID_EVAL, OOD_EVAL = ["td_test"], [s for s in EVAL if s != "td_test"]
KS = [1, 5, 10]
QUANTILES = [0.90, 0.95, 0.99]
PCA_DIM = 64


def sha(t):
    return hashlib.sha1(t.encode()).hexdigest()


def load_examples():
    ex = {s: decision_data.load(s) for s in ["td_train", "kev_train", "kev_dev", "td_test", "kevT_dev", "kevT_test",
                                             "kevT9_dev", "jb_original", "jb_easy", "jb_hard"]}
    hold = decision_data.load("td_holdout")
    hg = {e.group for e in hold}
    ex["td_holdout"] = hold
    ex["td_train_fit"] = [e for e in ex["td_train"] if e.group not in hg]   # what the pointers trained on
    return ex


def load_probs(ex):
    """Dumps are written in decision_data.load order per suite -> align by position and verify."""
    out = {}
    for tier in COST:
        path = os.path.join(PROBS, f"{tier}.jsonl")
        if not os.path.exists(path):
            continue
        by = {}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            by.setdefault(r["suite"], []).append(r)
        for s, rows in by.items():
            if s not in ex:
                continue
            assert len(rows) == len(ex[s]), (tier, s, len(rows), len(ex[s]))
            for r, e in zip(rows, ex[s]):
                assert r["qid"] == e.qid and r["label"] == e.label, (tier, s, r["qid"], e.qid)
        out[tier] = {s: (np.array([max(r["probs"]) for r in rows]),
                         np.array([int(np.argmax(r["probs"])) == r["label"] for r in rows]))
                     for s, rows in by.items()}
    return out


def knn_dist(Q, R, k_max, chunk=512):
    """cosine distance (1 - cos) to the k nearest reference vectors, sorted ascending."""
    out = np.empty((len(Q), k_max), np.float32)
    for i in range(0, len(Q), chunk):
        s = Q[i:i + chunk] @ R.T
        top = np.partition(-s, k_max - 1, axis=1)[:, :k_max]
        out[i:i + chunk] = np.sort(1 + top, axis=1)
    return out


def auroc(neg, pos):
    """P(score_pos > score_neg), ties 0.5."""
    x = np.concatenate([neg, pos])
    r = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float)
    # average ranks for ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv] + 1
    rp = r[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def encoder_scores(enc, ex):
    t0 = time.time()
    z = np.load(os.path.join(CACHE, f"choice_{ENCODERS[enc]}_2048.npz"))
    idx = {k: i for i, k in enumerate(z["keys"].tolist())}
    vecs = z["vecs"]

    def mat(texts):
        miss = [t for t in texts if sha(t) not in idx]
        if miss:
            raise KeyError(f"{enc}: {len(miss)} texts missing from cache")
        m = vecs[[idx[sha(t)] for t in texts]].astype(np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    ref_texts = list(dict.fromkeys(e.state_text for s in ("td_train_fit", "kev_train") for e in ex[s]))
    R = mat(ref_texts)
    mu = R.mean(0)
    w, V = np.linalg.eigh(np.cov(R - mu, rowvar=False))
    P, var = V[:, ::-1][:, :PCA_DIM].T, w[::-1][:PCA_DIM]
    resid_var = float(((R - mu) ** 2).sum(1).mean() - var.sum()) / (R.shape[1] - PCA_DIM)
    scores = {}
    for s in DEV + EVAL:
        texts = [e.state_text for e in ex[s]]
        uniq = list(dict.fromkeys(texts))
        pos = {t: i for i, t in enumerate(uniq)}
        Q = mat(uniq)
        d = knn_dist(Q, R, max(KS))
        c = (Q - mu) @ P.T
        maha = (c ** 2 / var).sum(1)
        resid = ((Q - mu) ** 2).sum(1) - (c ** 2).sum(1)
        per = {f"knn{k}": d[:, :k].mean(1) for k in KS}
        per["maha_pca"] = maha
        per["maha_pca+resid"] = maha + resid / resid_var   # PCA Mahalanobis + isotropic residual term
        ii = np.array([pos[t] for t in texts])
        scores[s] = {n: v[ii] for n, v in per.items()}
    print(f"[ood] {enc}: ref {len(R)} unique states, dim {R.shape[1]}, {time.time() - t0:.0f}s", flush=True)
    return scores, len(R)


def policy(gate_in, probs, small, top, thr, s):
    """gate_in: bool array (True -> cascade). Returns acc, share to cascade, mean cost (tier units)."""
    pm, cs = probs[small][s]
    _, ct = probs[top][s]
    stay = gate_in & (pm >= thr)
    correct = np.where(stay, cs, ct)
    cost = np.where(gate_in, COST[small] + np.where(pm >= thr, 0, COST[top]), COST[top])
    return float(correct.mean()), float(gate_in.mean()), float(cost.mean())


def main():
    ex = load_examples()
    probs = load_probs(ex)
    print("[ood] probs for", sorted(probs), flush=True)
    res = {"reference": "td_train minus td_holdout groups + kev_train (unique state texts)",
           "dev": DEV, "eval": EVAL, "cascades": CASCADES, "tier_cost": COST, "encoder_cost": ENC_COST,
           "n": {s: len(ex[s]) for s in DEV + EVAL}, "baselines": {}, "auroc": {}, "gates": []}

    # baselines per cascade: top alone, cascade always, domain-tag oracle (td_* known; kevT/jb unknown)
    for small, top, thr in CASCADES:
        if small not in probs or top not in probs:
            continue
        name = f"{small}->{top}@{thr}"
        b = {}
        for s in DEV + EVAL:
            n = len(ex[s])
            known = s in ("td_test", "kev_dev", "td_holdout")
            b[s] = {"top_alone": policy(np.zeros(n, bool), probs, small, top, thr, s),
                    "cascade_always": policy(np.ones(n, bool), probs, small, top, thr, s),
                    "oracle_tag": policy(np.full(n, known), probs, small, top, thr, s),
                    "small_alone_acc": float(probs[small][s][1].mean())}
        res["baselines"][name] = b

    for enc in ENCODERS:
        try:
            sc, nref = encoder_scores(enc, ex)
        except (KeyError, FileNotFoundError) as e:
            print("[ood] skip", enc, e, flush=True)
            continue
        names = list(next(iter(sc.values())))
        for small, top, thr in CASCADES:
            if small not in probs or top not in probs:
                continue
            # combined score: embedding distance (dev z-score) + small tier's uncertainty (dev z-score)
            for n0 in ("knn10", "maha_pca+resid"):
                dz = np.concatenate([sc[s][n0] for s in DEV])
                du = np.concatenate([1 - probs[small][s][0] for s in DEV])
                for s in DEV + EVAL:
                    sc[s][f"{n0}+unc[{small}]"] = ((sc[s][n0] - dz.mean()) / dz.std()
                                                   + (1 - probs[small][s][0] - du.mean()) / du.std())
        all_names = list(next(iter(sc.values())))
        for n in all_names:
            neg = np.concatenate([sc[s][n] for s in ID_EVAL])
            res["auroc"][f"{enc}/{n}"] = {"td_test_vs_all_ood": auroc(neg, np.concatenate([sc[s][n] for s in OOD_EVAL])),
                                         **{f"td_test_vs_{s}": auroc(neg, sc[s][n]) for s in OOD_EVAL},
                                         "dev_vs_td_test": auroc(np.concatenate([sc[s][n] for s in DEV]), neg)}
        for small, top, thr in CASCADES:
            if small not in probs or top not in probs:
                continue
            for n in all_names:
                if "+unc[" in n and f"[{small}]" not in n:
                    continue
                dev_sc = np.concatenate([sc[s][n] for s in DEV])
                for q in QUANTILES:
                    t = float(np.quantile(dev_sc, q))
                    row = {"encoder": enc, "score": n, "cascade": f"{small}->{top}@{thr}", "quantile": q,
                           "threshold": t, "ref_n": nref, "suites": {}}
                    for s in DEV + EVAL:
                        acc, share, cost = policy(sc[s][n] <= t, probs, small, top, thr, s)
                        row["suites"][s] = {"acc": acc, "to_cascade": share, "cost": cost,
                                            "rel_cost": cost / COST[top],
                                            "rel_cost_incl_gate": (cost + ENC_COST[enc]) / COST[top]}
                    res["gates"].append(row)
    def pooled(get, suites):
        w = np.array([len(ex[s]) for s in suites], float)
        return float(np.dot([get(s) for s in suites], w) / w.sum())

    # question-weighted pooled summary: dev, td_test (known domain), OOD = kevT_* + jb_*
    res["summary"] = {"baselines": {}, "gates": []}
    for cname, b in res["baselines"].items():
        res["summary"]["baselines"][cname] = {
            k: {f"{grp}_{m}": pooled(lambda s: b[s][k][j], ss) for grp, ss in (("dev", DEV), ("td_test", ID_EVAL),
                                                                                ("ood", OOD_EVAL))
                for j, m in ((0, "acc"), (1, "to_cascade"), (2, "cost"))}
            for k in ("top_alone", "cascade_always", "oracle_tag")}
    for g in res["gates"]:
        S = g["suites"]
        res["summary"]["gates"].append({"encoder": g["encoder"], "score": g["score"], "cascade": g["cascade"],
                                        "quantile": g["quantile"], **{
            f"{grp}_{m}": pooled(lambda s: S[s][m], ss) for grp, ss in (("dev", DEV), ("td_test", ID_EVAL),
                                                                          ("ood", OOD_EVAL))
            for m in ("acc", "to_cascade", "cost")}})
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    for cname, b in res["baselines"].items():
        print(f"\n== {cname}")
        for k in ("top_alone", "cascade_always", "oracle_tag"):
            print(f"  {k:16s} td_test {b['td_test'][k][0]:.3f}  OOD pooled {pooled(lambda s: b[s][k][0], OOD_EVAL):.3f}"
                  f"  cost td {b['td_test'][k][2]:.2f} ood {pooled(lambda s: b[s][k][2], OOD_EVAL):.2f}")
        rows = [g for g in res["gates"] if g["cascade"] == cname]
        rows.sort(key=lambda g: -pooled(lambda s: g["suites"][s]["acc"], OOD_EVAL))
        for g in rows[:12]:
            S = g["suites"]
            print(f"  {g['encoder']:15s} {g['score']:28s} q{g['quantile']:.2f} dev acc {pooled(lambda s: S[s]['acc'], DEV):.3f}"
                  f" td_test {S['td_test']['acc']:.3f} ({S['td_test']['to_cascade']:.2f} casc, cost {S['td_test']['cost']:.2f})"
                  f" OOD {pooled(lambda s: S[s]['acc'], OOD_EVAL):.3f} ({pooled(lambda s: S[s]['to_cascade'], OOD_EVAL):.2f} casc,"
                  f" cost {pooled(lambda s: S[s]['cost'], OOD_EVAL):.2f})")
    print("\n== AUROC td_test vs OOD (top 12)")
    for k, v in sorted(res["auroc"].items(), key=lambda kv: -kv[1]["td_test_vs_all_ood"])[:12]:
        print(f"  {k:45s} {v['td_test_vs_all_ood']:.3f}  dev_vs_td_test {v['dev_vs_td_test']:.3f}")


if __name__ == "__main__":
    main()
