# MiDM: Minimal Decision Model (naming, 2026-10-01)

**MiDM** is the name of both the architecture and the model family.

- **Architecture.** A causal LLM reads the state, the question instructions and every option line
  (`- key: text`) in one sequence. A linear **pointer head** on the final hidden state at the end of
  each option line gives one score per option, and a softmax over options gives the answer
  distribution.
- **Training.** The base is 4-bit NF4 and frozen. LoRA r16 (α32) is applied to every attention, MLP
  and linear-attention projection. Loss is soft-target cross-entropy, options are shuffled, and
  compute is bf16 on the RTX 3090 or fp16 on the RTX 2080 Ti.
- **Code.** `experiments/113_clm_reproduction_20260928/pointer_lora.py` (train/eval) and `serve_pointer.py`
  (System One API).
- **Distinct from** CLM (frozen bi-encoder with contrastive heads), Kev and Jev. MiDM is our own
  recipe and makes no claim to be any of them.

Name pattern: `MiDM-{size}-{base}-e{epochs}[-{variant}]`, with base `q3` = Qwen3 and `q35` = Qwen3.5-Base.

| model | base | epochs | adapter dir (exp 113) | typed-dec | Kev dev | Kev T test | JevBench pub. | status |
|---|---|---|---|---|---|---|---|---|
| MiDM-0.6B-q3-e1 | Qwen3-0.6B | 1 | runs/ptr0p6b/final | 0.736 | 0.785 | 0.585 | 0.576 | router tier 1 |
| MiDM-1.7B-q3-e1 | Qwen3-1.7B | 1 | runs/ptr1p7b/final | 0.782 | 0.817 | 0.641 | 0.632 | |
| MiDM-4B-q3-e1 | Qwen3-4B | 1 | runs/ptr4b/final | 0.789 | 0.854 | 0.716 | 0.688 | |
| MiDM-4B-q3-e2 | Qwen3-4B | 2 | runs/B/qwen3_4b_2ep/final | 0.797 | 0.848 | 0.725 | 0.684 | |
| MiDM-4B-q3-e1-broad | Qwen3-4B | 1 | runs/D/qwen3_4b_1ep_broad/final | 0.787 | 0.847 | 0.746 | 0.693 | experiment: +arc/obqa/csqa |
| MiDM-8B-q3-e1 | Qwen3-8B | 1 | runs/ptr8b/final | 0.794 | 0.857 | 0.764 | 0.697 | |
| MiDM-8B-q3-e2 | Qwen3-8B | 2 | runs/B/qwen3_8b_2ep/final | **0.805** | **0.864** | **0.772** | 0.706 | best overall (Qwen3) |
| **MiDM-4B-q35-e1** | Qwen3.5-4B-Base | 1 | runs/B/qwen35_4b_1ep/final | **0.805** | 0.861 | 0.764 | **0.710** | **flagship**: 8B-level at half cost; router tier 2 |
| MiDM-4B-q35-e2 | Qwen3.5-4B-Base | 2 | runs/B/qwen35_4b_2ep_bf16/final | 0.800 | 0.862 | 0.736 | 0.684 | not used: 2nd epoch hurts transfer |
| MiDM-0.8B-q35-e1 | Qwen3.5-0.8B-Base | 1 | runs/R/q35_0p8b_e1/final | 0.770 | 0.805 | 0.592 | 0.615 | better than 0.6B-q3, but the D1 rule keeps 0.6B as tier 1 |
| MiDM-4B-q35-e1-b1 | Qwen3.5-4B-Base | 1 | runs/E/q35_4b_e1_breadth/final | 0.791 | 0.857 | 0.779 | 0.697 | +breadth_v1 only: null (all p > 0.1) |
| **MiDM-4B-q35-e1-bx** | Qwen3.5-4B-Base | 1 | runs/E/q35_4b_e1_breadth_mc/final | 0.788 | 0.856 | **0.798** | 0.710 | **best off-distribution**: +breadth_v1 + arc/obqa/csqa; Kev T +3.4 (p = 0.042), Kev T dev +3.5 (p = 0.011), T-v9 +3.0 (p = 0.009); typed-dec −1.7 |

`-bx` training data: td_train + Kev decision-v7 + breadth_v1 (mrpc, qqp, rte, go_emotions,
civil_comments, winogrande) + public-pool-v6 (arc, openbookqa, csqa). There is zero text overlap
with any eval suite. This set is broader than Kev's training sources, so -bx vs Kev is not
like-for-like.

Each score is the one-time final evaluation (`results/pointer/FINAL_*.json`). Kev T test is Kev
transfer-v4 test, which covers never-trained sources.

**Router: `MiDM-Auto`.** A per-question cascade MiDM-0.6B-q3-e1 → MiDM-4B-q35-e1 → MiDM-8B-q3-e2 at
thresholds 0.66 / 0.30, used for known domains. Unknown domains go directly to MiDM-4B-q35-e1.
Decisions D1/D1a are in DECISIONS.md.
