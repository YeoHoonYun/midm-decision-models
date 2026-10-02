# MiDM: Minimal Decision Models, and an audit of CLM-8B

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23084584.svg)](https://doi.org/10.5281/zenodo.23084584)

Code, results and write-up for a controlled study of small (≤8B) typed-decision models on two consumer GPUs
(RTX 3090 + RTX 2080 Ti, Windows, no vLLM).

- **MiDM (Minimal Decision Model).** A causal LM reads the state, the question and every option line in one
  pass. A linear **pointer head** on the final hidden state at the end of each option line scores that
  option, and a softmax over options gives the answer. The base model is frozen in 4-bit NF4 and trained
  with LoRA r16 on every projection. Answers use the typed-decision ("System One") wire format:
  `choice`, yes/no `noul`, and `score`.
- **Audit of CLM-8B**, the Stanford/NVIDIA contrastive decision model (github.com/Contrastive-LM/CLM). We
  reproduce its DeepSWE best-of-4 verifier result exactly: 31/38 = 81.6%. Only 13 of the 38 tasks can be
  changed by the selector. On those tasks the result is 10/13 against a random expectation of 7.0, which
  gives an exact one-sided **p = 0.062**. The base head before task fine-tuning is below random.

## Headline results (each evaluation suite read once; `113_clm_reproduction_20260928/results/`)

| model | typed-decisions test | Kev transfer-v4 test | Kev transfer-v4 dev | JevBench public |
|---|---|---|---|---|
| CLM-style frozen Qwen3-4B + heads | 0.759 | 0.411 | 0.366 | 0.411 |
| MiDM-4B-q3-e1 (Qwen3-4B) | 0.789 | 0.716 | 0.700 | 0.688 |
| MiDM-8B-q3-e2 (Qwen3-8B) | **0.805** | 0.772 | 0.729 | 0.706 |
| MiDM-4B-q35-e1 (Qwen3.5-4B-Base) | **0.805** | 0.764 | 0.755 | 0.710 |
| **MiDM-4B-q35-e1-bx** (broadened data) | 0.788 | **0.798** | **0.791** | 0.710 |

Findings, with paired row-clustered bootstrap results in `results/stats/stats.md`:
- **Reading the options jointly is what transfers.** Ablation on Qwen3-4B with the same data:
  - a frozen bi-encoder head scores 0.411 on Kev transfer test;
  - joint options with a frozen base and only the pointer head trained scores 0.630;
  - adding LoRA scores 0.715.
  About 70% of the transfer gain comes from joint reading. LoRA adds most of the in-domain gain.
- **Data mix (`-bx`): +3.3 pp on Kev transfer test over 3 seeds** (0.763 ± 0.005 → 0.796 ± 0.002; per seed +3.4 / +3.8 / +2.7).
  There is no in-domain cost (typed-decisions −0.5 pp, within noise). The effect holds at 2B (+5.8 pp) and is gone at 0.8B.
- **Base generation and size.** These figures are precision-matched (bf16 vs bf16; fp16 vs bf16 alone was not significant).
  - Qwen3 → Qwen3.5 at 4B gives +3.4 to +5.9 pp on transfer suites.
  - Qwen3.5 2B → 4B gives +10.2 pp.
  - Neither generation nor data mix helps at 2B or below, so about 4B is the practical floor.
  - Qwen3.5-4B equals Qwen3-8B (p = 0.69).
- **No gain from:** DoRA, rsLoRA, rank 64, attention-only LoRA, per-domain adapters (routed or merged), TTA over
  option orders, temperature scaling, and ensembles. A second epoch hurts transfer (−2 to −3 pp).
- **In-distribution dev does not select for transfer.** This held for epochs, data mix and cascade thresholds.
- **Gap to Jev, per item on the 231 public JevBench items.**
  - The gap is almost entirely the hard tier: 0.450 vs 0.730. Jev alone is right on 32 items, MiDM alone on 1.
  - Longer inputs (+0.9 pp) and long-document training (`-bxL`) did not close it. Single-pass option scoring is
    weak at multi-step reasoning.
- **Cascade cost.** Measured batch-1 latency on an RTX 3090 (bf16):

  | tier | latency |
  |---|---|
  | 0.6B | 151 ms |
  | 4B-q35 | 309 ms |
  | 8B | 196 ms |

  In this stack a cascade is not faster than the 8B alone. `results/speed/` has the details.
- **As a best-of-N selector for SQL and code (zero-shot, offline; `115_midm_selector_sql_code_20261001/`):**
  - code (LiveCodeBench hard, 80 problems): +10 pp over majority vote (p = 0.004);
  - SQL with retrieved examples: +1.1 pp on Spider test and +1.7 pp on a private holdout over majority vote.
  It does not beat the Solar Pro4 direct baseline significantly.

## Comparison with CLM-8B and TypeSafe Jev

| system | typed-decisions test | Kev transfer-v4 dev | Kev transfer-v4 test | JevBench public | DeepSWE held-out 38 (Bo4) |
|---|---|---|---|---|---|
| TypeSafe Jev (commercial, zero-shot; reported) | 0.727 | 0.857 | – | **0.866** | 71.1% (reported) |
| CLM-8B (reported: typed-dec. from CLM PR #2, JevBench board) | 0.685 | – | – | 0.407 | 81.6% (claim) |
| CLM-8B released DeepSWE head, run by us | – | – | – | – | 31/38 = 81.6% (reproduced) |
| CLM-v0.1-8B base head, zero-shot, run by us | – | – | – | – | 27/38 = 71.1% (< random 73.7%) |
| CLM recipe re-trained by us (frozen Qwen3-8B, z-score, 3 seeds) | 0.766 | – | – | – | – |
| CLM-style frozen Qwen3-4B + heads (ours, same data as MiDM) | 0.759 | 0.366 | 0.411 | 0.411 | – |
| Kev-9B (reported) | – | 0.822 | **0.852** | – | – |
| MiDM-8B-q3-e2 | **0.805** | 0.729 | 0.772 | 0.706 | – |
| MiDM-4B-q35-e1 | **0.805** | 0.755 | 0.764 | 0.710 | – |
| **MiDM-4B-q35-e1-bx** | 0.788 | 0.791 | 0.798 | 0.710 | – |

How to read this:
- **Reported rows are not paired with ours.** They come from their authors' cards and boards and use different
  protocols. Jev is zero-shot. MiDM was trained on typed-decisions train, so the typed-decisions column favours MiDM.
- **Against CLM.** On the same data, the frozen-encoder design scores about 0.76 in-distribution but falls to
  about 0.4 on unseen sources and JevBench. MiDM keeps 0.76–0.80. CLM's DeepSWE headline reproduces exactly,
  but the edge over random is 3 tasks out of 13 decidable ones (p = 0.062). The base head is below random.
- **Against Jev and Kev.** MiDM (≤8B, trained locally on two consumer GPUs) is still behind Jev on Kev transfer
  dev (−6.6 pp) and JevBench public (−15.6 pp), and behind Kev-9B on Kev transfer test (−5.4 pp).
- **Where MiDM wins.** It runs locally in 4-bit on one GPU, and a 4B model matches 8B.

## Release
- Hugging Face: [yunicro/MiDM-4B-q35-e1-bx](https://huggingface.co/yunicro/MiDM-4B-q35-e1-bx) (public).
  It contains the LoRA adapter, pointer head, `midm.py` and the model card.

## Layout
- `113_clm_reproduction_20260928/`: CLM reproduction, frozen-head study and MiDM training/eval. See its `README.md`.
  - `pointer_lora.py`: MiDM train/eval (QLoRA, pointer head, `--tta`, `--save-probs`).
  - `decision_data.py`: all suites in one format (typed-decisions, Kev suites, JevBench, breadth sets).
  - `stats.py`, `build_ledger.py`: bootstrap tests and the test-read ledger (SHA-256 per result file).
  - `build_hf_package.py`, `hf_template/midm.py`, `verify_hf_package.py`: the Hugging Face release.
  - `gpuq.py`: a small Huey/SQLite GPU job queue (one worker per GPU, VRAM/utilisation start gate).
  - `data/breadth_v1/build_breadth_v1.py`: rebuilds the balanced-breadth set from public HF datasets.
- `114_local_model_router_20260928/`: local gateway (LiteLLM) plus a decision router (`MiDM-Auto` cascade),
  `MODELS.md` (catalogue), `DECISIONS.md` (every selection rule and its data).
- `docs/research/`: the report draft, the novelty and literature review, and references.

## Reproduce (outline)
1. Python 3.12. Install torch (CUDA) plus `transformers>=5.17 peft>=0.21 bitsandbytes>=0.50 pyarrow huggingface_hub`.
2. Clone the CLM repo at `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7` into `113_clm_reproduction_20260928/repo/`.
   It is not vendored; the schema helpers are used under Apache-2.0.
3. Download the public data into `113_clm_reproduction_20260928/data/`: `LocalLLaMA/typed-decisions`,
   `jaredpalmer/kev-suites` and the JevBench public items. Then run `data/breadth_v1/build_breadth_v1.py`.
4. Train and evaluate, e.g.
   `PTR_BF16=1 python pointer_lora.py train --model Qwen/Qwen3.5-4B-Base --train td_train kev_train breadth_v1 pp6_new --epochs 1 --out runs/E/bx`,
   then `pointer_lora.py eval ... --save-probs`.

Datasets and model weights are not redistributed here. Datasets keep their own licences. The adapters are
released separately on Hugging Face: https://huggingface.co/yunicro/MiDM-4B-q35-e1-bx (Zenodo DOI 10.5281/zenodo.23084584)

## Citation
Authors: YeoHoon Yoon and Kyung-Sung Kim (Graduate School of AI, aSSIST University, Seoul, Republic of Korea).

```bibtex
@software{yoon_kim_2026_midm,
  author    = {Yoon, YeoHoon and Kim, Kyung-Sung},
  title     = {MiDM: Minimal Decision Models and an audit of CLM-8B},
  year      = {2026},
  version   = {0.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23084584},
  url       = {https://github.com/YeoHoonYun/midm-decision-models}
}
```
See `CITATION.cff`. Archived on Zenodo: https://doi.org/10.5281/zenodo.23084584 (all versions: https://doi.org/10.5281/zenodo.23084583)
## License
Apache-2.0 (`LICENSE`); third-party notices are in `NOTICE`.
