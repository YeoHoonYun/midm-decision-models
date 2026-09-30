"""Per-workflow sweep -> results/sweep_summary.{json,md}.

'pooled' combines the four per-workflow specialists into one accuracy over all 2000
test decisions (decision-weighted), comparable to the single 'all' head.
"""
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "repo", "src"), os.path.join(HERE, "repo", "train")]
import adapters  # noqa: E402
import finetune  # noqa: E402

WFS = ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]
n_test = {wf: len(list(adapters.typed_decision_examples(
    finetune.load_typed_rows(os.path.join(HERE, "data", "typed-decisions"), "test", wf, None))))
    for wf in WFS}

acc = defaultdict(dict)   # (enc, pat) -> {(wf, seed): test acc}
for f in glob.glob(os.path.join(HERE, "runs", "sweep", "*", "finetune_summary.json")):
    s = json.load(open(f)); a = s["args"]
    acc[(a["embed_model"].split("/")[-1], a["patience"])][(a["workflow"], a["seed"])] = s["test"]["acc"]

lines = ["| encoder | patience | " + " | ".join(WFS) + " | pooled (per seed) | pooled mean |",
         "|---|---|" + "---|" * len(WFS) + "---|---|"]
rows = []
for (enc, pat), d in sorted(acc.items()):
    seeds = sorted({s for _, s in d})
    per_wf = {wf: st.mean(d[(wf, s)] for s in seeds) for wf in WFS}
    pooled = [sum(d[(wf, s)] * n_test[wf] for wf in WFS) / sum(n_test.values()) for s in seeds]
    rows.append({"encoder": enc, "patience": pat, "per_workflow_mean": per_wf, "pooled": pooled})
    lines.append(f"| {enc} | {pat} | " + " | ".join(f"{per_wf[wf]:.3f}" for wf in WFS) +
                 f" | {', '.join(f'{p:.4f}' for p in pooled)} | {st.mean(pooled):.4f} |")
json.dump({"n_test_decisions": n_test, "rows": rows},
          open(os.path.join(HERE, "results", "sweep_summary.json"), "w"), indent=1)
open(os.path.join(HERE, "results", "sweep_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(n_test)
print("\n".join(lines))
