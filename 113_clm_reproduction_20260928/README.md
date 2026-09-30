# 113 CLM-8B reproduction and smaller-encoder cycle (2026-09-28)

> **Naming (2026-10-01):** the QLoRA pointer decision models built here are **MiDM (Minimal Decision
> Model)**. MiDM names both the architecture and the model family, e.g. `MiDM-4B-q35-e1` =
> Qwen3.5-4B-Base, 1 epoch. The catalogue and the mapping to `runs/` directories are in
> `../114_local_model_router_20260928/MODELS.md`, and the router is `MiDM-Auto`.

Source: AI타임스 2026-09-27 (idxno=215683) on Stanford/NVIDIA CLM-8B vs TypeSafe Jev.
Code: github.com/Contrastive-LM/CLM @ bb42c6c5bf914fd449bed2f6ca65be80602cb1f7 (repo/, unmodified).

## Scope and constraints

- Uses only spare resources: RTX 2080 Ti (GPU 1) and CPU. The RTX 3090 was running
  another Ollama job (llama-server, ~17.6 GB) and was not touched.
- No data from other projects and no external model APIs were used.
  Only Hugging Face downloads (models, public datasets).
- Isolated env: `.venv` (Python 3.12, torch 2.14.0+cu126, transformers 5.17.0), created with
  `PYTHONPATH` cleared because the global `PYTHONPATH` points at another project's site-packages.
- vLLM is Linux-only, so `run_choice_hf.py` wraps `repo/train/finetune.py` with an in-process
  transformers embedder (same token recipe: no special tokens, tail kept; final-layer last-token
  state; L2-normalised) and sizes the heads to the encoder width. The repo itself is unchanged.

## Cycle 1: DeepSWE held-out-38 reproduction (article: CLM 81.6%)

The README's eval dataset `Contrastive-LM/deepswe-clm-embeddings-8k` returns 401 (not public).
The same data is public under its earlier name `tarsur385/deepswe-prm-embeddings-8k`
(Opus-5 rollouts, 4 per task, Qwen3-8B embeddings). Head and held-out list SHA-256 match
`verification.json`.

| selector | held-out 38 (Bo4, last-12 mean) | decidable 13 tasks |
|---|---|---|
| released DeepSWE head (`deepswe-clm-heads-8k`) | **31/38 = 81.579%** (exact match) | 10/13 |
| CLM-v0.1-8B base head, zero-shot | 27/38 = 71.053% | 6/13 |
| random pick (expectation) | 73.7% | ~7/13 |
| oracle (any candidate passes) | 89.5% | 13/13 |

21 tasks pass with every candidate and 4 fail with every candidate, so only 13 tasks can be
changed by the selector. The headline 81.6% vs random 73.7% is a 3-task difference on a
13-task decidable set. The base head without DeepSWE fine-tuning is below random.
Results: `results/r1_*.json`, `results/r2_*.json`.

Not reproduced: Terminal-Bench 2.1 (no script/data released), BFCL v4 and WikiRacing (no
scripts), T-Rex latency (needs a vLLM encoder server), Jev numbers (commercial API).

## Cycle 2: smaller open encoders on typed-decisions

Benchmark: `LocalLLaMA/typed-decisions` (`all`: 1200 train rows -> 5400 train / 600 val
questions, 400 test rows -> 2000 test decisions). Synthetic third-party benchmark; its README
leaderboard lists TypeSafe Jev 0.727, ModernBERT-base (fitted) 0.646, prior 0.470.

Recipe: `finetune.py --task choice` defaults (InfoNCE, soft targets, 20 epochs, width 1536,
depth 3, lr 5e-4, batch 256), frozen Qwen3 encoder in fp16 (Turing has no fast bf16; bf16
measured 4.5x slower), best epoch picked on val only, test read once per run.
Seeds 1234/1/2 (the seed also changes the val split). Patience 5 is the repo default;
patience 20 (= no early stop in 20 epochs) was added after seeing seed-2 1.7B stop at
epoch 7, and applied uniformly. p20 runs trained heads on CPU, p5 on GPU, so identical
configs can differ slightly.

See `results/choice_summary.md` for the per-seed table.

| encoder (single `all` head) | patience 5 | patience 20 |
|---|---|---|
| Qwen3-0.6B | 0.643 | 0.651 |
| Qwen3-1.7B | 0.594 (one seed early-stopped at 0.476) | 0.639 |
| Qwen3-4B | 0.694 | 0.691 |
| Qwen3-8B (scratch head) | – | 0.697 |
| Qwen3-8B, warm start from released CLM-v0.1-8B head | – | 0.679 |
| Qwen3-8B, released CLM-v0.1-8B head zero-shot (epoch 0) | – | 0.358 |

Qwen3-8B was embedded on the 2080 Ti with 7 GiB of weights on GPU and the rest offloaded
to CPU (23 min; batches that overflowed 11 GB spilled into shared memory and stalled, so
two earlier attempts were stopped). Caveat for the zero-shot/warm rows: the released head
was trained on vLLM bf16 embeddings; these are transformers fp16, which may cost some accuracy.

### Parallel per-workflow sweep (`parallel_sweep.py`)

72 CPU jobs (3 encoders x 4 workflows x 3 seeds x patience 5/20), 5 workers x 2 threads,
cached embeddings, 4.2 min total. Per-workflow specialists (300 train rows each) pooled over
the same 2000 test decisions are worse than one head trained on all workflows
(4B: 0.631 vs 0.691; 0.6B: 0.582 vs 0.651). `results/sweep_summary.md`.

## Cycle 3: head-end fine-tuning on frozen Qwen3-8B until it beats Jev (typed-decisions)

Protocol: every choice made by 5-fold CV over the 1200 train rows, grouped by row
(`head_search.py`); test evaluated once for the frozen winner. Jobs ran in parallel on the
3090's spare memory (Ollama untouched) and the 2080 Ti (`par.py`).

| round | change (CV accuracy, 5 folds) |
|---|---|
| baseline | repo recipe, 20 epochs: 0.700 |
| r1 | 40 epochs 0.738; softce 0.708; hard targets 0.734; width 3072/depth 2 0.741; **z-score inputs 0.774** |
| r2 (z-score, 80 epochs) | 0.766–0.773 for lr, width, dropout, wd, batch, mix loss; hard targets 0.744 |
| r3 (per-layer features, `extract_layers.py`, 34 min) | last-token layer 12 0.720, 18–35 0.770–0.774; mean-pool 0.750–0.768; avg of layers 16–28 0.777 |

Frozen config: final-layer last token, z-score (train statistics), InfoNCE, soft targets,
width 1536/depth 3, lr 5e-4, 40 epochs, trained on all 1200 train rows.
**TEST (once): seeds 0.7690 / 0.7675 / 0.7620, mean 0.7662, 3-seed ensemble 0.7685**
(Jev 0.727, meraGPT Decider 1 0.768 per the dataset leaderboard, which we cannot
re-run or pair per item). `results/search/FINAL_test_r1_08.json`.

All test evaluations made in this experiment: the default-recipe runs in cycle 2
(0.6B/1.7B/4B/8B, warm start) and this single final one. No search step read test.

## Cycle 4: one multi-benchmark head (<=8B frozen encoder) vs the Jev-class field

`decision_data.py` maps every suite to the same `build_pairs` texts the server embeds;
`embed_all.py` caches encoders (0.6B, 1.7B, 4B, Qwen3-Embedding-0.6B; Qwen3-Embedding-4B
download stalled and was dropped; 8B skipped under memory pressure from other jobs on the host).
`multi_train.py`: train td_train + Kev decision-v7 train; select on Kev decision-v7 dev + a 10%
td_train row holdout (`par_multi.py`, 2 concurrent on the 3090). Chosen: Qwen3-4B, z-score,
InfoNCE, 40 epochs (m1_06, select 0.753). **Correction:** this was not the top select score. m1_09
(td-weight 2, select 0.755) was 0.002 higher, but the simpler default was taken without that being
recorded at the time. The difference is within noise; the entry is corrected for accuracy. Final, each suite read once (`results/multi/FINAL_qwen3-4b_e40.json`):

| suite | n | acc | Brier | ECE | reference |
|---|---|---|---|---|---|
| typed-decisions test | 2000 | 0.759 | 0.368 | 0.136 | Jev 0.727, od1 0.796 (full FT) |
| Kev decision-v7 dev (in-domain) | 1468 | 0.715 | 0.445 | 0.174 | Kev-9B 0.872 |
| Kev transfer-v4 dev | 764 | 0.367 | 0.930 | 0.384 | Kev-9B 0.822, Jev 0.857 |
| Kev transfer-v4 test (read once) | 764 | 0.411 | 0.877 | 0.328 | Kev-9B 0.852 |
| Kev transfer-v9 dev | 1264 | 0.370 | 0.910 | 0.346 | – |
| JevBench public (72+48+111) | 231 | 0.411 | – | – | Jev 0.866, CLM-8B 0.407, Kev 8B 0.714 |

The frozen-encoder head is competitive only in-distribution; it collapses on never-trained
sources (MMLU-style items etc.). Note: a smoke run of the 0.6B multi head also read td_test
(0.7225) before this final; it did not inform any choice.

### Chrome T-Rex (repo `examples/t_rex/run.py`, 5 seeds x 60 s, real time, shield on)

Served the 4B multi head (encoder on the 2080 Ti): latency p50 16.5-16.7 ms, model p50 2.3-3.3 ms,
but 0/5 survived, 16 deaths, agreement with the planner 0.014-0.030 (`results/trex/`).
Probe: the head puts 0.99 on the two "Unsafe ... Collision." options; `clm-raw` (no head)
picks "jump". Published: CLM-8B 5/5, agreement 0.658, 4883 shield interventions, 16.5 ms;
Jev 5/5, agreement 0.987, 28 interventions, 149.8 ms. The 16.5 ms p50 appears to track the
60 FPS frame (16.7 ms) rather than model speed.

Literature/novelty review: `research_topics/clm_decision_heads_novelty_20260928/README.md`.

## Cycle 5: QLoRA pointer model (Kev-style cross-encoder), <=8B, both GPUs

`pointer_lora.py`: one sequence per question (state + instructions, then `- key: text` option
lines); final hidden state at the end of each option line -> linear pointer head -> softmax;
soft-target CE; options shuffled in training. Qwen3 base in 4-bit NF4, fp16 compute, LoRA r16 on
all attention/MLP projections, gradient checkpointing, 4096-token batches, 1 epoch over
td_train (minus 10% holdout) + Kev decision-v7 train (20,976 questions, 4.47M tokens), lr 2e-4.
The checkpoint rule was fixed in advance: `final`. Each evaluation suite was read once (`results/pointer/`).

Run notes: the 4B run stalled at step 430 after an eval with a 2x batch spilled the 2080 Ti past
11 GB; it resumed from `step400` with the same seed and batch order but a fresh optimizer state. A
queue bug briefly ran 1.7B next to it, and that run was stopped. 8B ran on the 3090 alongside the
other experiment's Ollama at ~330 tok/s. 4B on the 2080 Ti ran at ~650 tok/s, 6.4 GiB.

| suite | frozen 4B + head | **4B pointer QLoRA** | reference |
|---|---|---|---|
| typed-decisions test | 0.759 | **0.789** (Brier 0.331) | Jev 0.727, meraGPT 0.768, od1 0.796 |
| Kev decision-v7 dev | 0.715 | **0.854** (0.214) | Kev-9B 0.872 |
| Kev transfer-v4 dev | 0.367 | **0.700** | Kev-9B 0.822, Jev 0.857 |
| Kev transfer-v4 test (once) | 0.411 | **0.716** (0.365) | Kev-9B 0.852, Kev-4B 0.838 |
| Kev transfer-v9 dev | 0.370 | **0.634** | – |
| JevBench public 231 (orig/easy/hard 0.931/1.000/0.396) | 0.411 | **0.688** | Jev 0.866, Kev 8B 0.714, CLM-8B 0.407 |

All sizes (pointer QLoRA, 1 epoch, same data and recipe; each suite read once; `show_pointer.py`):

| base | typed-decisions | Kev dev | Kev transfer dev | Kev transfer test | transfer-v9 dev | JevBench public |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.736 | 0.785 | 0.597 | 0.585 | 0.525 | 0.576 |
| Qwen3-1.7B | 0.782 | 0.817 | 0.624 | 0.641 | 0.553 | 0.632 |
| Qwen3-4B | 0.789 | 0.854 | 0.700 | 0.716 | 0.634 | 0.688 |
| **Qwen3-8B** | **0.794** | **0.857** | **0.726** | **0.764** | **0.657** | **0.697** |
| reference | Jev 0.727, od1 0.796 | Kev-9B 0.872 | Jev 0.857, Kev-9B 0.822 | Kev-9B 0.852 | – | Jev 0.866, Kev 8B 0.714 |

8B ran on the RTX 3090 next to the experiment-34 Ollama. Its first final eval (job 19) was killed at
15:45 by an external process kill and re-run as job 27.

## Cycle 6 (queued): LoRA / adapter search through the GPU queue

All GPU work now goes through `gpuq.py` (Huey + SQLite, one worker per GPU; start gate = enough free
VRAM and utilisation <= 50% for 90 s). Spec shared with every session on the machine:
`experiments/_ops/GPU_QUEUE_SPEC.md`.

Selection never reads a test set. Stage A runs use `--holdout-sources boolq mnli sst5`: those Kev
sources are removed from training, and their kev_dev rows become the `src_holdout` proxy for
never-seen sources. The other select sets are the td_train 10% row holdout and 500 kev_dev rows.
- Stage A (Qwen3-1.7B proxy, 1 epoch unless noted): A1 LoRA r16 all, A2 DoRA r16, A3 rsLoRA r64,
  A4 LoRA attention-only, A5 lr 1e-4, A6 lr 4e-4, A7 2 epochs, A8 LoRA r64 (jobs 3-6 on gpu1; 12-15 on
  gpu0 after the 8B final eval, job 11).
- Stage B (next): the best 1-2 stage-A settings on Qwen3-4B, Qwen3.5-4B-Base, Qwen3-8B, trained on all
  sources; then each test suite is evaluated once.
- Stage C (adapters, user suggestion): per-domain adapters (classification vs reasoning/policy)
  with routing or merging, and merge-then-continue (a second adapter trained on top of the merged first).

### Stage A result and stage B (2026-09-28/29)
Stage A (Qwen3-1.7B proxy, dev mean of td_holdout / src_holdout / kev_dev500):
- A7 (2 epochs) 0.771
- A1 0.739, A5 0.739, A2 DoRA 0.732, A6 0.729, A4 attention-only 0.718, A8 r64 0.711
- A3 rsLoRA r64 diverged (NaN, fp16)

Stage B used the A7 recipe (LoRA r16, all projections, lr 2e-4, 2 epochs) on all sources. 8B was
trained in bf16 on the 3090, 4B in fp16 on the 2080 Ti. Each suite was read once
(`results/pointer/FINAL_B_*.json`).

| base | typed-decisions | Kev dev | Kev transfer dev | Kev transfer test | transfer-v9 dev | JevBench public |
|---|---|---|---|---|---|---|
| Qwen3-4B, 2 ep | 0.797 | 0.848 | 0.703 | 0.725 | 0.633 | 0.684 |
| **Qwen3-8B, 2 ep** | **0.805** | **0.864** | **0.729** | **0.772** | **0.661** | **0.706** |
| (1 ep: 4B / 8B) | 0.789 / 0.794 | 0.854 / 0.857 | 0.700 / 0.726 | 0.716 / 0.764 | 0.634 / 0.657 | 0.688 / 0.697 |

| **Qwen3.5-4B-Base, 1 ep** (bf16, 3090) | **0.805** | 0.861 | **0.755** | 0.764 | **0.675** | **0.710** |

Qwen3.5-4B at 1 epoch beats Qwen3-4B at 1 epoch by +4 to +5.5 pp on never-trained sources and
roughly matches Qwen3-8B at 2 epochs. The base-model generation matters more than epochs or LoRA
variants. **Qwen3.5-4B-Base, 2 epochs** ran fp16 on the 2080 Ti outside the queue at the user's request. It was
resumed at step 1600 with a fresh optimizer, after a shared-memory spill; it spilled again near the
end. Results (`FINAL_B_qwen35-4b-2ep.json`):

| suite | td_holdout | Kev dev | typed-dec | Kev T dev | Kev T test | T-v9 | JevBench |
|---|---|---|---|---|---|---|---|
| score | 0.868 | 0.866 | 0.802 | 0.709 | 0.734 | 0.646 | 0.706 |
| change vs the 1-epoch bf16 run | +2.3 | +0.5 | −0.3 | −4.6 | −3.0 | −2.9 | −0.4 |

The in-distribution dev score rose while never-trained sources fell. This matches A9 (3 epochs on
1.7B). The comparison is confounded, though: precision differs (fp16 vs bf16), the optimizer was
reset, and the run spilled, so "the 2nd epoch hurts transfer for Qwen3.5" is not established. Best
for transfer: Qwen3.5-4B 1 epoch. Best overall: Qwen3-8B 2 epochs.

**Clean re-run B5 (2026-09-30).** Qwen3.5-4B, 2 epochs, bf16 on the 3090, no restart, no spill:
td_holdout 0.855, Kev dev 0.862, typed-dec 0.800, Kev T dev 0.734, Kev T test 0.736, T-v9 0.647,
JevBench 0.684. Against the 1-epoch run, transfer falls 2.1 to 2.8 pp and JevBench 2.6 pp. With
precision and restart removed as confounds, **the second epoch hurts transfer for Qwen3.5**. One
epoch is the setting to use.

**Breadth D1 (2026-09-30).** Qwen3-4B, 1 epoch, fp16 on the 2080 Ti, with 3,000 extra public-pool-v6
questions (arc, openbookqa, csqa). There is zero text overlap with any eval suite. Against the same
recipe without them: Kev T test 0.746 (+3.0), T-v9 0.653 (+1.9), Kev T dev 0.709 (+0.9), JevBench
0.693 (+0.5), typed-dec 0.787 (−0.2). Per source, the gain is near-transfer:
- up: mmlu +18.1 (test) / +6.9 (dev), mmlu_pro +6.5, sciq +6.9 / +1.7, emotion +4
- down: paws −7.5 / −8.8, qnli −5.0 / −3.8, composition −6.2 / −3.1

Adding one task type (knowledge multiple-choice) helps that type and dilutes sentence-pair tasks.
Breadth has to be balanced across task types. This run uses more training sources than Kev did,
so it is not a like-for-like comparison with Kev.

**Improvement search on MiDM-4B-q35-e1 (2026-10-01).** Paired row-clustered bootstrap, `results/stats/stats.md`.
- **Statistics.**
  - The CLM DeepSWE selector gets 10/13 decidable tasks vs a random expectation of 7.0; the exact
    one-sided p is 0.062.
  - Generation (Qwen3 → Qwen3.5, 4B, 1 epoch) gains +4 to +5 pp on transfer suites (p < 0.005).
  - 8B-e2 vs 4B-q35-e1 shows no difference (Kev T test p = 0.69).
  - Seed noise reference (Qwen3-4B, seed 0 vs 1): ±1.2 pp, not significant.
- **TTA (4 option orders, `pointer_lora.py eval --tta`).** Adopted for 4B-q35 only, because dev
  improved there and fell for 8B. Test: +0.1 to +0.9 pp, p ≥ 0.09. This is **null**.
- **Balanced breadth (E1, `data/breadth_v1`).** Adds 6,000 questions from mrpc, qqp, rte,
  go_emotions, civil_comments and winogrande, with zero eval overlap. Changes are −1.4 to +1.4 pp,
  all p > 0.1. Per-source signs flip between the dev and test transfer suites. This is **null**.
- **Temperature scaling on dev.** The fitted T is 1.00 for 0.6B and 1.03 for 4B-q35. The cascade
  is unchanged. Off-distribution loss is a domain-shift problem, not global overconfidence. This is **null**.
- **Ensemble of 4B-q35-e1 and 8B-q3-e2 (dev weight 0.6/0.4).** It passes the dev adoption rule.
  Test: typed-dec 0.808 (+0.3), Kev T test 0.764 (±0), Kev T dev 0.751, JevBench 0.706. There is no
  transfer gain and the cost is 3x, so it is **not adopted**.
- **E2 = breadth_v1 + arc/obqa/csqa, i.e. balanced breadth including knowledge-MC (MiDM-4B-q35-e1-bx).**
  - Transfer: Kev T test 0.798 (+3.4, p = 0.042), Kev T dev 0.791 (+3.5, p = 0.011), T-v9 0.705
    (+3.0, p = 0.009).
  - In-domain and other: JevBench 0.710 (0), typed-dec 0.788 (−1.7), Kev dev 0.856 (−0.5).
  - Per source, the gain is broad, unlike D1. Changes vs baseline, test / dev (D1 in brackets):
    mmlu +6.9 / +7.8; paws +1.2 / +6.2 (D1: −7.5 / −8.8); qnli −2.5 / +7.5; composition +15.6 / +3.1;
    emotion +2.6 / +3.4; sciq −3.4 / −0.9.
  - Knowledge-MC alone (D1) dilutes sentence-pair tasks, and balanced breadth alone (E1) does nothing.
    Together they raise transfer broadly. This is the first significant improvement beyond the
    base-model generation, and -bx is now the router's unknown-domain model (D7).
- **Qwen3.5-0.8B, 1 epoch (MiDM-0.8B-q35-e1).** Typed-dec 0.770, Kev T test 0.592, JevBench 0.615,
  better than 0.6B-q3 on every suite. As cascade tier 1 the D1 rule still keeps 0.6B (dev cost 29% vs 30%).
- **Seeds (Qwen3-4B e1, s0/s1/s2).** Kev T test 0.716 / 0.726 / 0.728 (0.723 ± 0.006), Kev T dev
  0.702 ± 0.013, typed-dec 0.784 ± 0.005. Running: Qwen3.5-4B seeds, Qwen3.5-2B.

A9 (1.7B, 3 epochs) reaches a dev mean of 0.748, below A7's 0.771 at 2 epochs; its
src_holdout falls from 0.583 to 0.518, so 3 epochs overfits. Next is Qwen3.5-4B at 2 epochs (queued, job 57).

The second epoch adds +0.3 to +1.1 pp. The +4 pp src_holdout gain seen on the 1.7B proxy does not
carry over to 4B/8B transfer. The remaining gap to Kev (transfer test 0.838–0.852) and Jev (JevBench
0.866) looks like training-source breadth, not epochs. The typed-decisions 0.805 vs od1 0.796 gap is
inside about 1 pp SE, so it is not established.

## Server (`embed_server.py` + `serve_clm.py`)

- `embed_server.py`: OpenAI-compatible `/v1/embeddings` on transformers (vLLM stand-in), same
  recipe as training, fp16. Ran Qwen3-0.6B on the RTX 3090's spare memory (~1.5 GB) at :8090.
- `serve_clm.py --hidden 1024`: the repo's unchanged `clm.server` (System One API
  `/v1/systemone`, `/v1/rank`, playground at `/`) with the encoder width patched.
  Heads on CPU at :8700: `clm-latest` = 0.6B `all` head, plus four per-workflow heads.
  The model description string "Qwen3-8B encoder" is hard-coded in the repo and is wrong here.

`server_replay.py` replays the 400 test rows through the API (`results/server_replay_*.json`):

| model | accuracy | server p50, encoder cold | head-only cold | warm | 8 concurrent, warm |
|---|---|---|---|---|---|
| clm-latest (0.6B head) | 0.6515 (offline 0.6490) | 152 ms | 4.0 ms | 0.9 ms | 500 req/s |
| clm-raw (no head) | 0.3485 | 84 ms | – | 0.8 ms | 477 req/s |

The 0.25-point server/offline gap (5 of 2000 decisions) is fp16 padding/batching numerics.
"Encoder cold" includes the 0.6B forward; the 152 ms run happened while the 3090 was shared.
The article's 16 ms is a different task on an RTX 4090/H100 with vLLM and is not comparable.
The raw encoder cosine is below the 0.47 prior: the trained heads carry the task.
