# CLM and MiDM verification — 2026-10-05

These are completed local measurements. The new 9B model release and GitHub/Hugging Face release tags remain pending; this update publishes evaluation evidence only.

## Matched 4B/9B FP16 evaluation

Both models used the same item content and ordering (hashes checked), NF4, FP16, max input 4096, padded-token batch budget 4096 and TTA=1 on RTX 2080 Ti. Fixed checkpoints, no tuning on these outcomes. These are retrospective evaluations, not a new blind test.

| Test | Items | MiDM 4B-bx | MiDM 9B-bx | Difference (pp) | Historical CLM-style 4B* |
|---|---:|---:|---:|---:|---:|
| td_test | 2000 | 78.90% | 79.35% | +0.45 | 75.90% |
| kevT_dev | 764 | 78.93% | 79.19% | +0.26 | 36.65% |
| kevT_test | 764 | 79.97% | 82.59% | +2.62 | 41.10% |
| jb_original | 72 | 91.67% | 97.22% | +5.56 | 40.28% |
| jb_easy | 48 | 100.00% | 100.00% | +0.00 | 54.17% |
| jb_hard | 111 | 45.95% | 54.05% | +8.11 | 36.04% |

*CLM numbers are historical, not newly run under the MiDM protocol. That control used Qwen3-4B and td_train + kev_train; MiDM-bx uses Qwen3.5 and a broader data mix. These columns do not isolate architecture or training effects.

The 2,000-item td_test is distinct from the 600-item training-selection holdout. Do not compare their percentages as a version gain. No statistical significance is claimed here.

## DeepSWE replay

The released CLM DeepSWE head again solved 31/38 tasks (81.58%), matching the previous run. Best-of-4 (one task has three candidates), final 12-step mean; 13 tasks have mixed candidate correctness. The prior random-selection comparison had one-sided exact p=0.062.

MiDM 4B/9B DeepSWE scores remain unavailable: the inspected local dataset contains embeddings and labels, not the raw state/action texts required by MiDM. Existing embeddings cannot substitute for those inputs.

## Pending and resource limitations

- Primary BF16 matched evaluation and 9B SQL/Python candidate-pool evaluation remain queued on RTX 3090. Historical SQL/Python results remain 4B results.
- 9B kevT_test took 2470.9 seconds on the 11 GiB GPU. This run does not establish an efficient serving configuration or a minimum VRAM requirement.
- BF16 development/holdout measurements and single-request package measurements are separate protocols; do not mix them with this table.
- No raw examples, per-item outputs, private finance data, or credentials are published.

Machine-readable aggregate evidence: [metrics.json](metrics.json).

## Completed follow-up: SQL/Python and same-GPU concurrency

The previously queued 9B SQL/Python evaluation is complete. [Detailed selection results](SQL_PYTHON_9B.md) use the existing pools, with each model's blend weight selected on Spider dev. These are retrospective reused benchmarks, not fresh blind tests.

| Protocol | 4B historical | 9B new |
|---|---:|---:|
| Spider test, RAG5 + blend | 78.34% | 77.27% |
| SQL holdout, RAG5 + blend | 81.66% | 81.77% |
| Python hard80, RAG0 | 28.75% | 25.00% |

9B does not uniformly improve selection. For Python the new 9B gain over majority vote is +6.25 pp with p=0.059; it is not significant at 5%. Do not replace the 4B code-selection claim with a 9B superiority claim.

The separate same-GPU load pilot ran 1 versus 2 independent 4B NF4 FP16 processes on RTX 2080 Ti. Synthetic input lengths: 67 and 313 tokens; 24 requests per worker after loading/warmup. One-worker throughput: 1.801 requests/s, p50 0.539s, p95 0.673s. Two-worker throughput: 3.587 requests/s, per-worker p50 0.546/0.559s and p95 0.610/0.639s. Both modes completed without errors or within-worker prediction changes. Torch peak allocation was 3.327 GiB per worker (not total-device VRAM). This short single trial does not establish sustained capacity, long-context performance, 9B concurrency or production latency. Other system activity was not controlled. [Aggregate load results](concurrency_load_test.json).

DeepSWE MiDM scores and Terminal-Bench same-pool scores remain unavailable. Public CLM and author artifact inventories contain DeepSWE embeddings, but the matching raw candidate traces and Terminal-Bench evaluation assets were not found in the inspected sources. Training traces from other benchmarks are not substitutes. No published claim was copied into a local measurement cell.
