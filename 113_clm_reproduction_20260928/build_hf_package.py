#!/usr/bin/env python3
"""Assemble a Hugging Face-ready MiDM release folder (no upload).

    python build_hf_package.py --adapter runs/E/q35_4b_e1_breadth_mc/final --name MiDM-4B-q35-e1-bx \
        --base Qwen/Qwen3.5-4B-Base --final results/pointer/FINAL_E_q35_4b_e1_breadth_mc.json \
        --baseline results/pointer/FINAL_B_qwen35-4b-1ep.json --baseline-name MiDM-4B-q35-e1 \
        --train-suites td_train kev_train breadth_v1 pp6_new

Output: <out-root>/<name>/ (default: the research folder's huggingface/) with adapter_config.json, adapter_model.safetensors, pointer_head.safetensors,
midm_config.json, midm.py, requirements.txt, README.md (model card).
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = [("td_test", "typed-decisions test (2000)"), ("kev_dev", "Kev decision-v7 dev (1468)"),
          ("kevT_dev", "Kev transfer-v4 dev (764)"), ("kevT_test", "Kev transfer-v4 test (764)"),
          ("kevT9_dev", "Kev transfer-v9 dev (1264)"), ("jb_public", "JevBench public (231)")]
DATA = {
    "td_train": "LocalLLaMA/typed-decisions `all` train (Apache-2.0; synthetic, teacher-labelled)",
    "kev_train": "jaredpalmer/kev-suites decision-v7 train (derived from agnews, yelp, dbpedia14, banking77, trec, "
                 "mnli, imdb, sst5, boolq, amazon reviews, plus generated policy data; each source keeps its own terms)",
    "breadth_v1": "built for this release from glue/mrpc, glue/qqp, glue/rte, go_emotions, civil_comments, "
                  "winogrande (train splits; each keeps its own terms)",
    "pp6_new": "jaredpalmer/kev-suites public-pool-v6: arc, openbookqa, csqa rows",
}


def scores(path):
    d = json.load(open(path))["eval"]
    out = {k: v["acc"] for k, v in d.items()}
    if all(k in d for k in ("jb_original", "jb_easy", "jb_hard")):
        out["jb_public"] = sum(d[k]["acc"] * d[k]["n"] for k in ("jb_original", "jb_easy", "jb_hard")) / 231
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--baseline-name", default=None)
    ap.add_argument("--train-suites", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--repo-id", default=None, help="Hugging Face repo id shown in the model card usage example")
    ap.add_argument("--out-root", default=os.path.join(HERE, "..", "..", "research_topics",
                    "clm_decision_heads_novelty_20260928", "huggingface"))
    a = ap.parse_args()
    out = os.path.abspath(os.path.join(a.out_root, a.name))
    os.makedirs(out, exist_ok=True)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(os.path.join(HERE, a.adapter, f), os.path.join(out, f))
    cfgj = json.load(open(os.path.join(out, "adapter_config.json")))
    if cfgj.get("base_model_name_or_path") != a.base:
        cfgj["base_model_name_or_path"] = a.base
        json.dump(cfgj, open(os.path.join(out, "adapter_config.json"), "w"), indent=2)
    head = torch.load(os.path.join(HERE, a.adapter, "pointer_head.pt"), map_location="cpu")
    save_file({k: v.contiguous() for k, v in head.items()}, os.path.join(out, "pointer_head.safetensors"))
    hidden = int(head["weight"].shape[1])
    json.dump({"architecture": "MiDM", "base_model": a.base, "hidden_size": hidden, "max_len": 1024,
               "option_line_format": "\\n- {key}: {text}", "options_header": "\\n\\nOptions:",
               "quantization_at_training": "bitsandbytes nf4 double-quant", "lora": {"r": cfgj.get("r"),
               "alpha": cfgj.get("lora_alpha")}, "epochs": a.epochs, "train_suites": a.train_suites},
              open(os.path.join(out, "midm_config.json"), "w"), indent=2)
    shutil.copy2(os.path.join(HERE, "hf_template", "midm.py"), os.path.join(out, "midm.py"))
    open(os.path.join(out, "requirements.txt"), "w").write(
        "torch>=2.4\ntransformers>=5.17\npeft>=0.21\nbitsandbytes>=0.50\nsafetensors\nhuggingface_hub\naccelerate\n")

    sc = scores(os.path.join(HERE, a.final))
    base_sc = scores(os.path.join(HERE, a.baseline)) if a.baseline else None
    rows = []
    for k, label in SUITES:
        if k in sc:
            b = f" | {base_sc[k]:.3f} | {100 * (sc[k] - base_sc[k]):+.1f}" if base_sc and k in base_sc else ""
            rows.append(f"| {label} | **{sc[k]:.3f}**{b} |")
    hdr = f"| suite | {a.name} | {a.baseline_name} | change (pp) |\n|---|---|---|---|" if base_sc else \
          f"| suite | {a.name} |\n|---|---|"
    data_lines = "\n".join(f"- `{s}`: {DATA.get(s, s)}" for s in dict.fromkeys(a.train_suites))
    card = f"""---
license: apache-2.0
base_model: {a.base}
library_name: peft
pipeline_tag: text-classification
tags: [midm, decision-model, typed-decisions, pointer-head, lora, qlora]
---

# {a.name}

**MiDM (Minimal Decision Model)**: a typed-decision model. The input is one state plus typed questions
(`choice`, `noul` yes/no, `score`), and the output is a probability distribution over each question's options.
It is a LoRA adapter plus a linear **pointer head** on `{a.base}`. The model reads the state, the question and
every option line in one pass. It scores each option from the final hidden state at the end of that option's
line.

## Use
```python
from midm import MiDM          # midm.py ships in this repo
m = MiDM.from_pretrained("{a.repo_id or '<this repo id or local folder>'}")   # 4-bit NF4 base by default (as in training)
m.predict(state={{"order_total": 1240, "po_total": 1420}},
          questions={{"action": {{"type": "choice", "instructions": "What should happen to this invoice?",
                                  "criteria": {{"approve": "Approve and pay.", "hold": "Hold for review.",
                                               "reject": "Reject as fraud."}}}}}})
```
Requests follow the "System One" typed-decision shape. `requirements.txt` lists the dependencies.
A GPU with about 9 GB free is needed for the 4B model in 4-bit.

## Evaluation (each suite evaluated once; accuracy = argmax over options)
{hdr}
{chr(10).join(rows)}

Kev transfer suites contain sources never seen in training (e.g. MMLU, SciQ, emotion, PAWS, QNLI).
JevBench "public" is the 231 public items only; sealed-set performance is typically much lower on that board.
For reference, numbers reported by others on the same suites are not paired with ours and come from
different protocols. They include TypeSafe Jev 0.727 on typed-decisions (zero-shot) and Kev-9B 0.852 on
Kev transfer-v4 test.

## Training
- Base: `{a.base}`, frozen, 4-bit NF4. LoRA r={cfgj.get('r')}, alpha={cfgj.get('lora_alpha')} on all attention/MLP
  (and linear-attention) projections. Pointer head: Linear({hidden}, 1).
- Recipe: soft-target cross-entropy over options, options shuffled, {a.epochs} epoch(s), lr 2e-4, 4096-token batches.
- Data:
{data_lines}
- Held-out evaluation suites were checked for zero text overlap with all training rows.

## Limitations
- typed-decisions is synthetic and teacher-labelled, with public test labels.
- Gains on unseen sources are partly near-transfer. The largest gain is on knowledge multiple-choice (MMLU).
- Accuracy on the in-distribution typed-decisions test is lower than the narrower-data model.
- The model scores only the options you give it. It cannot answer open questions and does not abstain.
- Training data include datasets with their own licences or terms (e.g. Yelp, Amazon reviews). Review them
  before commercial use. The Apache-2.0 licence applies to this adapter and code only.
"""
    open(os.path.join(out, "README.md"), "w", encoding="utf-8").write(card)
    print(f"wrote {out}")
    for f in sorted(os.listdir(out)):
        print(f"  {f:28s} {os.path.getsize(os.path.join(out, f)):>12,d}")


if __name__ == "__main__":
    main()
