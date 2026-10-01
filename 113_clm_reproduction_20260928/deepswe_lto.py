#!/usr/bin/env python3
"""DeepSWE audit, step 2: retrain the CLM DeepSWE head ourselves, held-out tasks removed, over several seeds.

Questions:
  (a) Leakage check. Do any held-out task ids occur in the public train-embedding pool? CLM only filters them
      by a task list; we count them, train with the same list, and assert that no held-out task is in train/val.
  (b) Seed sensitivity. Is 31/38 = 81.6% typical for the official recipe, or one lucky draw?
      The recipe: warm start from CLM-v0.1-8B, batch 512, default lr rule, held-out list.

    python deepswe_lto.py --seeds 42 1 2 3 4        # runs on one GPU via the queue (gpu1)

Writes runs/deepswe_lto/s<seed>/ and results/deepswe_lto/{s<seed>.json, summary.json, summary.md}.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HELD = os.path.join(HERE, "heads", "deepswe", "heldout_tasks.json")
INIT = os.path.join(HERE, "heads", "clm_v0.1_8b", "CLM_v0.1-8B.pt")
EVAL_EMB = os.path.join(HERE, "data", "emb_cache", "data_deepswe_eval_parquet")
OUT = os.path.join(HERE, "results", "deepswe_lto")


def find_train_dir(root):
    """Directory holding the train pool's metadata.json (the HF snapshot may nest it)."""
    hits = sorted(glob.glob(os.path.join(root, "**", "metadata.json"), recursive=True), key=len)
    if not hits:
        raise SystemExit(f"no metadata.json under {root}; is the dataset downloaded?")
    return os.path.dirname(hits[0])


def leak_check(train_dir, held):
    meta = json.load(open(os.path.join(train_dir, "metadata.json")))
    tasks = defaultdict(int)
    for s in meta["samples"]:
        tasks[s["task_id"]] += 1
    overlap = {t: tasks[t] for t in held if t in tasks}
    return {"pool_tasks": len(tasks), "pool_samples": len(meta["samples"]), "heldout_tasks": len(held),
            "heldout_tasks_in_pool": len(overlap), "heldout_samples_in_pool": sum(overlap.values())}


def evaluate(ckpt, out_json):
    cmd = [sys.executable, os.path.join(HERE, "repo", "evaluation", "bon_eval.py"), "--embeddings-dir", EVAL_EMB,
           "--checkpoint", ckpt, "--tasks-file", HELD, "--n", "4", "--window", "12", "--output", out_json]
    subprocess.run(cmd, check=True, cwd=os.path.join(HERE, "repo"))
    return json.load(open(out_json))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.join(HERE, "data", "deepswe_train_emb"),
                    help="HF snapshot of Contrastive-LM/deepswe-clm-train-embeddings-8k (parquet)")
    ap.add_argument("--train-root", default=os.path.join(HERE, "data", "deepswe_train_dir"),
                    help="embedding dir used for training (built from --parquet if missing)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    held = json.load(open(HELD))
    held = held.get("tasks", held.get("heldout_tasks")) if isinstance(held, dict) else held
    if not glob.glob(os.path.join(a.train_root, "**", "metadata.json"), recursive=True):
        # HF parquet snapshot -> the repo's embedding-dir format (one-time conversion, CPU)
        sys.path.insert(0, os.path.join(HERE, "repo", "preprocessing"))
        import hf_embeddings  # noqa: E402
        print(f"[lto] converting {a.parquet} -> {a.train_root}", flush=True)
        hf_embeddings.download(a.parquet, a.train_root)
    train_dir = find_train_dir(a.train_root)
    leak = leak_check(train_dir, set(held))
    print("[lto] leak check:", leak, flush=True)

    summary = {"leak_check": leak, "recipe": {"init": "CLM_v0.1-8B.pt", "batch": 512, "holdout": "heldout_tasks.json",
                                              "eval": "bon_eval --n 4 --window 12"}, "runs": {}}
    for seed in a.seeds:
        rd = os.path.join(HERE, "runs", "deepswe_lto", f"s{seed}")
        os.makedirs(rd, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, "repo", "train", "finetune.py"), "--task", "clm", "--emb-dir", train_dir,
               "--init-ckpt", INIT, "--out-dir", rd, "--holdout-tasks", HELD, "--batch", "512", "--seed", str(seed)]
        print("[lto] train:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=os.path.join(HERE, "repo"))
        split = json.load(open(os.path.join(rd, "task_split.json")))
        assert not set(split["train_tasks"]) & set(held) and not set(split["val_tasks"]) & set(held)
        ckpt = os.path.join(rd, "best_head.pt")
        res = evaluate(ckpt, os.path.join(OUT, f"s{seed}.json"))
        sel = next(iter(res["selectors"].values()))
        summary["runs"][seed] = {"accuracy": sel.get("rate"), "solved": sel.get("resolved"),
                                 "random_pick": res.get("random_pick"), "oracle": res.get("oracle_any")}
        print(f"[lto] seed {seed}: {summary['runs'][seed]}", flush=True)
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

    accs = [r["accuracy"] for r in summary["runs"].values() if r["accuracy"] is not None]
    summary["accuracy_mean"], summary["accuracy_sd"] = float(np.mean(accs)), float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    md = ["# DeepSWE held-out 38: CLM head retrained by us (held-out tasks excluded)", "",
          f"Leak check: {leak}", "", "| seed | accuracy | solved/38 |", "|---|---|---|"]
    md += [f"| {s} | {r['accuracy']} | {r['solved']} |" for s, r in summary["runs"].items()]
    md += ["", f"mean {summary['accuracy_mean']:.4f} ± {summary['accuracy_sd']:.4f} (sd over seeds); released head 0.8158; "
           f"random 0.7368; oracle 0.8947"]
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
