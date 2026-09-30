"""Collect runs/choice_*/finetune_summary.json into results/choice_summary.{json,md}."""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "runs", "choice_*", "finetune_summary.json"))):
    s = json.load(open(f))
    a = s["args"]
    rows.append({"run": os.path.basename(os.path.dirname(f)), "encoder": a["embed_model"],
                 "init": os.path.basename(a["init_ckpt"]) if a.get("init_ckpt") else "scratch",
                 "patience": a["patience"], "seed": a["seed"], "best_epoch": s["best_epoch"],
                 "val_acc": s["val_acc"], "test_acc": s["test"]["acc"], "init_test_acc": s["init_test"]["acc"],
                 "test_soft_ce": s["test"]["soft_ce"], "majority_test": s["majority_test"], "minutes": s["minutes"]})

groups = {}
for r in rows:
    groups.setdefault((r["encoder"], r["init"], r["patience"]), []).append(r)
lines = ["| encoder | init | patience | seeds | test acc per seed | mean | init (epoch-0) test acc |",
         "|---|---|---|---|---|---|---|"]
for (enc, init, pat), rs in sorted(groups.items()):
    accs = [r["test_acc"] for r in sorted(rs, key=lambda r: r["seed"])]
    lines.append(f"| {enc} | {init} | {pat} | {len(rs)} | {', '.join(f'{x:.4f}' for x in accs)} | "
                 f"{st.mean(accs):.4f} | {', '.join(f'{r['init_test_acc']:.4f}' for r in sorted(rs, key=lambda r: r['seed']))} |")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(rows, open(os.path.join(HERE, "results", "choice_summary.json"), "w"), indent=1)
open(os.path.join(HERE, "results", "choice_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
