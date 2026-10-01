#!/usr/bin/env python3
"""Statistical tests for the paper (CPU only). Writes results/stats/stats.json and stats.md.

1. DeepSWE held-out Bo4 selector vs random: exact one-sided test. Under H0 (selection uniform over
   the 4 candidates) task i succeeds with p_i = passes_i / n_i. The number of successes is then a sum of
   independent Bernoullis, and its exact distribution is obtained by convolution. P(X >= observed).
2. Paired, row-clustered bootstrap of accuracy differences between models on every eval suite,
   using the per-question probability dumps (same question order across models). Rows (cases) are
   resampled, not questions, because questions of one row share a state.
"""
import json
import os
import random
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "stats")


def deepswe():
    meta = json.load(open(os.path.join(HERE, "data", "emb_cache", "data_deepswe_eval_parquet", "metadata.json")))
    held = set(json.load(open(os.path.join(HERE, "heads", "deepswe", "heldout_tasks.json"))))
    traj = {}
    for s in meta["samples"]:
        if s["task_id"] in held:
            traj[s["trajectory_id"]] = (s["task_id"], float(s.get("reward") or 0) > 0.5)
    by_task = defaultdict(dict)
    for t, (task, ok) in traj.items():
        by_task[task][t] = ok
    res = {}
    runs = [("released_head", "r1_deepswe_heldout38_released_head.json"),
            ("base_head_zeroshot", "r2_deepswe_heldout38_zeroshot_v0.1_head.json")]
    lto = os.path.join(HERE, "results", "deepswe_lto")
    if os.path.isdir(lto):   # our retrained heads (held-out tasks excluded), one per seed
        runs += [(f"retrained_{os.path.splitext(f)[0]}", os.path.join("deepswe_lto", f))
                 for f in sorted(os.listdir(lto)) if f[0] == "s" and f[1:-5].isdigit() and f.endswith(".json")]
    for name, f in runs:
        r = json.load(open(os.path.join(HERE, "results", f)))
        picks = next(iter(r["selectors"].values()))["picks"]
        ps, obs_dec, n_dec = [], 0.0, 0
        for task, trials in by_task.items():
            p = sum(trials.values()) / len(trials)
            chosen = picks.get(task, [])
            succ = np.mean([trials[t] for t in chosen]) if chosen else p   # exact ties -> expectation
            if 0 < p < 1:
                ps.append(p); obs_dec += succ; n_dec += 1
        dist = np.array([1.0])
        for p in ps:   # convolution of Bernoulli(p)
            dist = np.convolve(dist, [1 - p, p])
        k = int(np.floor(obs_dec + 1e-9))
        p_ge = float(dist[k:].sum())
        p_le = float(dist[:k + 1].sum())
        res[name] = {"decidable_tasks": n_dec, "observed_successes_decidable": round(obs_dec, 3),
                     "random_expectation_decidable": round(float(sum(ps)), 3),
                     "p_one_sided_ge": p_ge, "p_one_sided_le": p_le}
    return res


def load_probs(path):
    rows, pos = {}, defaultdict(int)
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        rows[(r["suite"], pos[r["suite"]])] = r
        pos[r["suite"]] += 1
    return rows


def correct(r):
    p = r["probs"]
    return int(max(range(len(p)), key=p.__getitem__) == r["label"])


def paired(a_path, b_path, B=4000, seed=0):
    A, Bm = load_probs(a_path), load_probs(b_path)
    out = {}
    suites = sorted({k[0] for k in A} & {k[0] for k in Bm})
    rng = np.random.default_rng(seed)
    for s in suites:
        keys = [k for k in A if k[0] == s and k in Bm]
        for k in keys:
            if A[k]["qid"] != Bm[k]["qid"] or A[k]["label"] != Bm[k]["label"]:
                raise ValueError(f"misaligned {k}")
        groups = defaultdict(list)
        for k in keys:
            groups[A[k]["group"]].append(k)
        g = list(groups)
        da = np.array([sum(correct(A[k]) for k in groups[x]) for x in g], float)
        db = np.array([sum(correct(Bm[k]) for k in groups[x]) for x in g], float)
        n = np.array([len(groups[x]) for x in g], float)
        diff = (db.sum() - da.sum()) / n.sum()
        idx = rng.integers(0, len(g), size=(B, len(g)))
        boots = (db[idx].sum(1) - da[idx].sum(1)) / n[idx].sum(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p_two = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
        out[s] = {"n_questions": int(n.sum()), "n_rows": len(g), "acc_a": float(da.sum() / n.sum()),
                  "acc_b": float(db.sum() / n.sum()), "diff_b_minus_a": float(diff),
                  "ci95": [float(lo), float(hi)], "p_bootstrap_two_sided": min(1.0, p_two)}
    return out


PAIRS = [  # (label, model A, model B) -> B minus A
    ("generation: Qwen3-4B e1 -> Qwen3.5-4B e1", "qwen3-4b", "B_qwen35-4b-1ep"),
    ("size: Qwen3-4B e2 -> Qwen3-8B e2", "B_qwen3-4b-2ep", "B_qwen3-8b-2ep"),
    ("half-cost parity: Qwen3-8B e2 -> Qwen3.5-4B e1", "B_qwen3-8b-2ep", "B_qwen35-4b-1ep"),
    ("epoch: Qwen3.5-4B e1 -> e2 (bf16, clean)", "B_qwen35-4b-1ep", "B_qwen35-4b-2ep-bf16"),
    ("epoch: Qwen3-8B e1 -> e2", "qwen3-8b", "B_qwen3-8b-2ep"),
    ("breadth: Qwen3-4B e1 -> +arc/obqa/csqa", "qwen3-4b", "D_qwen3-4b-1ep-broad"),
    ("balanced breadth: Qwen3.5-4B e1 -> +breadth_v1", "B_qwen35-4b-1ep", "E_q35_4b_e1_breadth"),
    ("balanced breadth + MC: Qwen3.5-4B e1 -> +breadth_v1+arc/obqa/csqa", "B_qwen35-4b-1ep", "E_q35_4b_e1_breadth_mc"),
    ("TTA: Qwen3.5-4B e1, 1 order -> 4 orders", "B_qwen35-4b-1ep", "T4_q35_4b_e1"),
    ("seed: Qwen3-4B e1 seed0 -> seed1 (noise reference)", "qwen3-4b", "S_q3_4b_e1_s1"),
    ("size (Qwen3.5): 0.8B e1 -> 2B e1", "R_q35_0p8b_e1", "R_q35_2b_e1"),
    ("size (Qwen3.5): 2B e1 -> 4B e1", "R_q35_2b_e1", "B_qwen35-4b-1ep"),
    ("generation at small size: Qwen3-1.7B e1 -> Qwen3.5-2B e1", "qwen3-1.7b", "R_q35_2b_e1"),
    ("generation at small size: Qwen3-0.6B e1 -> Qwen3.5-0.8B e1", "qwen3-0.6b", "R_q35_0p8b_e1"),
    ("-bx at 2B: Qwen3.5-2B e1 -> +breadth_v1+arc/obqa/csqa", "R_q35_2b_e1", "R_q35_2b_e1_bx"),
    ("-bx at 0.8B: Qwen3.5-0.8B e1 -> +breadth_v1+arc/obqa/csqa", "R_q35_0p8b_e1", "R_q35_0p8b_e1_bx"),
    ("precision: Qwen3-4B e1 fp16 -> bf16", "qwen3-4b", "P_q3_4b_e1_bf16"),
    ("generation, both bf16: Qwen3-4B e1 -> Qwen3.5-4B e1", "P_q3_4b_e1_bf16", "B_qwen35-4b-1ep"),
    ("precision: Qwen3.5-2B e1 fp16 -> bf16", "R_q35_2b_e1", "P_q35_2b_e1_bf16"),
    ("size (Qwen3.5), both bf16: 2B e1 -> 4B e1", "P_q35_2b_e1_bf16", "B_qwen35-4b-1ep"),
    ("ablation: + LoRA (joint options fixed; Qwen3-4B bf16)", "H_q3_4b_headonly", "P_q3_4b_e1_bf16"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"deepswe": deepswe(), "paired": {}}
    probs = os.path.join(HERE, "results", "probs")
    for label, a, b in PAIRS:
        pa, pb = os.path.join(probs, f"{a}.jsonl"), os.path.join(probs, f"{b}.jsonl")
        if os.path.exists(pa) and os.path.exists(pb):
            res["paired"][label] = {"a": a, "b": b, "suites": paired(pa, pb)}
    json.dump(res, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
    lines = ["# Statistical tests (auto-generated by stats.py)", "", "## DeepSWE held-out Bo4 vs random (decidable tasks)", "",
             "| selector | decidable | observed | random expectation | P(X >= obs) | P(X <= obs) |", "|---|---|---|---|---|---|"]
    for k, v in res["deepswe"].items():
        lines.append(f"| {k} | {v['decidable_tasks']} | {v['observed_successes_decidable']} | "
                     f"{v['random_expectation_decidable']} | {v['p_one_sided_ge']:.4f} | {v['p_one_sided_le']:.4f} |")
    lines += ["", "## Paired row-clustered bootstrap (B minus A, accuracy pp, 95% CI, 4000 resamples)", ""]
    for label, v in res["paired"].items():
        lines += [f"### {label}", "", "| suite | n | A | B | diff | 95% CI | p |", "|---|---|---|---|---|---|---|"]
        for s, d in v["suites"].items():
            lines.append(f"| {s} | {d['n_questions']} | {d['acc_a']:.3f} | {d['acc_b']:.3f} | {100 * d['diff_b_minus_a']:+.1f} | "
                         f"[{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] | {d['p_bootstrap_two_sided']:.3f} |")
        lines.append("")
    open(os.path.join(OUT, "stats.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
