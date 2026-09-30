# MiDM: Minimal Decision Models, and an audit of CLM-8B

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
- Moving from the frozen bi-encoder heads to the pointer cross-encoder with LoRA adds about 30 pp on unseen sources.
- The base-model generation matters most. Qwen3 → Qwen3.5 at 4B gives +4–5 pp on transfer (p < 0.005).
  Qwen3.5-4B equals Qwen3-8B at half the cost (p = 0.69).
- These changes did not beat LoRA r16 on all projections: DoRA, rsLoRA, rank 64, attention-only LoRA,
  per-domain adapters (routed or merged), TTA over option orders, temperature scaling and ensembles.
- A second epoch hurts transfer for Qwen3.5 (−2 to −3 pp).
- Breadth only helps when it is balanced across task types:
  - Knowledge-MC alone helps MMLU and dilutes sentence-pair tasks.
  - Balanced sentence-pair, affect, toxicity and commonsense data alone does nothing.
  - Both together give +3.0 to +3.5 pp on transfer (p = 0.009 to 0.042). This is the `-bx` model.
- In-distribution dev selection does not pick for transfer. We saw this for cascade thresholds, epochs and data mix.
- A confidence cascade (0.6B → 4B → 8B) keeps in-domain accuracy at about 43% of the cost. It loses about 3 pp
  off-distribution because small tiers are overconfident there. The router therefore sends unknown domains
  straight to the strongest model.

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
released separately on Hugging Face (link: TODO).

## Citation
See `CITATION.cff`. A DOI will be added after the Zenodo release.

## License
Apache-2.0 (`LICENSE`); third-party notices are in `NOTICE`.
