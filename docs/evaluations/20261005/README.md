# MiDM verification and load-test report — 2026-10-05

## Completion and scope

Completed: matched BF16 and supplementary FP16 4B/9B evaluations; CLM DeepSWE replay; 9B SQL/Python selection; routing replay; 1-vs-2-process short- and long-context load pilots. This report finalizes the available measurements, not a claim that every requested benchmark is scored.

Still unavailable: MiDM DeepSWE and local Terminal-Bench same-pool scores, because matching raw candidate traces/evaluation assets were not found in the inspected sources. Version 0.2.0 packages these results with the 9B model; see the versioned release and model card for publication identifiers. No financial states, raw predictions, private inputs or credentials are in this update.

## Primary matched BF16 evaluation

Same item content and ordering, NF4 BF16, 4096 maximum input length, 4096 padded-token batch budget, TTA=1, RTX 3090. Fixed checkpoints; retrospective tests, not new blind evaluation.

| Test | N | 4B | 9B | Difference (pp) |
|---|---:|---:|---:|---:|
| td_test | 2000 | 78.85% | 79.30% | +0.45 |
| kevT_dev | 764 | 79.06% | 79.06% | +0.00 |
| kevT_test | 764 | 79.84% | 82.59% | +2.75 |
| jb_original | 72 | 91.67% | 97.22% | +5.56 |
| jb_easy | 48 | 100.00% | 100.00% | +0.00 |
| jb_hard | 111 | 45.95% | 54.05% | +8.11 |

Typed Decisions test is 2,000 questions, distinct from the 600-item model-selection holdout. These are hard-label argmax accuracies, not JevBench official composite points. No statistical superiority is claimed from these deltas. Historical FP16 measurements remain in [metrics.json](metrics.json); small precision-dependent changes must not be mixed into a single comparison.

## CLM and benchmark coverage

- Released CLM DeepSWE head: 31/38 (81.58%) reproduced. Only 13 tasks have mixed candidate outcomes; prior exact one-sided comparison against random selection: p=0.062. This is successful numerical reproduction with inconclusive 5%-level evidence of superiority over random, not a failed reproduction.
- MiDM DeepSWE: missing raw state/action candidate texts. The local CLM-specific embeddings are not interchangeable with MiDM inputs.
- Terminal-Bench: the author-reported 87.6% is not a local measurement; matching 30-task traces, scoring details and head are missing from inspected artifacts.
- Historical CLM-style 4B is a Qwen3-4B reimplementation; MiDM-bx uses Qwen3.5 plus a broader training mix. Differences do not isolate architecture. It is not the released CLM-v0.1-8B.
- JevBench public231 measured Jev 1.13 Free 194/231, Solar 220/231, local Qwen3-4B reasoning 189/231; protocols and compute differ. Public accuracy is not the official full/sealed composite ranking.

## SQL/Python selection

| Protocol | Historical 4B | New 9B |
|---|---:|---:|
| Spider test, RAG5 + blend | 78.34% | 77.27% |
| SQL holdout, RAG5 + blend | 81.66% | 81.77% |
| Python hard80, RAG0 | 28.75% | 25.00% |

Same stored candidate pools; each blend weight selected only on Spider dev. 9B does not consistently improve selection. Python gain over majority vote is +6.25 pp, p=0.059; not significant at 5%. See [detailed results](SQL_PYTHON_9B.md). Retrospective benchmark reuse is disclosed.

## Learned router pilot

MiDM9B to Qwen3-4B reasoning. Ridge gain model trained on 400 development items, alpha=10, threshold=0 fixed before evaluation. Uses stored predictions: no new live end-to-end serving run, calculator or RAG.

| Suite | MiDM | Router | Reasoning alone | Routed requests |
|---|---:|---:|---:|---:|
| jb_easy | 100.00% | 100.00% | 100.00% | 40/48 |
| jb_original | 97.22% | 97.22% | 94.44% | 25/72 |
| jb_hard | 54.05% | 59.46% | 65.77% | 35/111 |
| ALL | 77.06% | 79.65% | 81.82% | 100/231 |

Decision: do not promote the router as the default. It underperforms reasoning alone, escalates 40/48 easy items unnecessarily, and does not establish non-inferiority or latency savings. Seen benchmark gains are exploratory; hard-suite descriptive delta interval includes zero.

## Same-GPU concurrency load pilots

Independent model processes, synchronized request start after loading and warmup. Two-process runs cap each CUDA allocator at 45% of GPU memory. Peak allocation below is per process, not total-device VRAM.

| Model / input | GPU | 1 process req/s | 2 processes req/s | Throughput ratio | p50: 1 process / 2 processes (s) | Peak allocated per process GiB |
|---|---|---:|---:|---:|---|---:|
| 4B short (67/313 tokens) | 2080 Ti | 1.801 | 3.587 | 1.99x | 0.539 / 0.559/0.546 | 3.327 |
| 4B 2048 tokens | 2080 Ti | 0.904 | 0.773 | 0.85x | 1.110 / 2.586/2.597 | 3.769 |
| 9B 4096 tokens | 3090 | 0.538 | 0.436 | 0.81x | 1.849 / 4.590/4.594 | 6.735 |

All three pilots completed without reported errors or within-worker prediction changes. Short run: 24 requests per worker; long runs: 12 per worker. Long contexts are synthetic repeated background text, not a representative application workload. This is a small single-trial load pilot, not endurance certification; external activity and phase order were not controlled.

Operational conclusion: the tested short 4B workload benefited from two processes. Long 4B and 9B inputs lost aggregate throughput and increased latency, despite fitting in VRAM. Keep long-input jobs to one model process per GPU; test batching separately before adopting it. Do not turn on unrestricted concurrency for every job.

## Files and traceability

[Primary four-benchmark rescore](four-benchmarks/README.md) · [SQL/Python details](SQL_PYTHON_9B.md) · [Historical FP16 metrics](metrics.json) · [Completion manifest](completion_manifest.json).

## Report integration

[Same-date local report audit](REPORT_INTEGRATION.md): identical as-of inputs and writer conditions, aggregate checks only. Better candidate-order agreement on one date is not evidence of better financial accuracy.

## Packaged-model verification

The final package reproduces the original development evaluation exactly: **513/600 (85.50%)**, identical choices on 600/600 questions and maximum absolute probability difference 0.0. The original protocol uses NF4/BF16, 4096-token context, 4096 padded tokens per batch, attention masks and TTA=1. Adapter weights, pointer head and encoded inputs match the source exactly. This is package reproducibility, not a new independent test.

A separate single-question, unmasked check gave **509/600 (84.83%)** and failed its original tolerance gate. It is retained rather than hidden. That verification bypassed the public API batching/mask path. The matched check changes batching and mask together, so their individual effects were not isolated. Do not assume identical predictions across serving configurations.

## v0.2.1 analysis update

[Benchmark coverage, architecture and parameter-performance atlas](atlas/README.md). Completed measurements, explicit missing-artifact analysis and 13 comparison rows; no new inference or imputed scores.
