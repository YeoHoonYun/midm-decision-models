# Speed

## 1. MiDM batched throughput (from the eval logs)

Token-budget batches (4096 tokens), 4-bit NF4 base, transformers on Windows, no vLLM. Each row covers all eval suites of that model (~6.5–7.1k questions).

| model | hardware | questions | q/s | ms per question | td_test ms/q |
|---|---|---|---|---|---|
| MiDM-0.6B-q3-e1 | manual run (before queue) | 6491 | 68.3 | 14.6 | 18.6 |
| MiDM-1.7B-q3-e1 | manual run (before queue) | 6491 | 35.6 | 28.1 | 36.1 |
| MiDM-0.8B-q35-e1 | RTX 2080 Ti fp16 | 7091 | 28.8 | 34.7 | 41.6 |
| MiDM-2B-q35-e1 | RTX 2080 Ti fp16 | 7091 | 21.4 | 46.7 | 56.9 |
| MiDM-2B-q35-e1 (bf16) | RTX 3090 bf16 | 7091 | 32.9 | 30.4 | 38.1 |
| MiDM-4B-q3-e1 | manual run (before queue) | 6491 | 16.0 | 62.4 | 79.8 |
| MiDM-4B-q3-e1 (bf16) | RTX 3090 bf16 | 7091 | 19.8 | 50.4 | 62.4 |
| MiDM-4B-q35-e1 | RTX 3090 bf16 | 7091 | 14.5 | 68.7 | 86.5 |
| MiDM-4B-q35-e1-bx | RTX 3090 bf16 | 7091 | 14.3 | 70.2 | 88.3 |
| MiDM-8B-q3-e1 | RTX 3090 fp16 | 7091 | 11.5 | 87.0 | 109.0 |
| MiDM-8B-q3-e2 | RTX 3090 bf16 | 7091 | 12.2 | 82.0 | 102.1 |

## 2. Single-question latency (batch 1)

Measured on NVIDIA GeForce RTX 3090 (torch.bfloat16), n=300 td_holdout questions.

| tier | p50 ms | p90 ms | mean ms | cost ratio vs smallest |
|---|---|---|---|---|
| MiDM-0.6B-q3-e1 | 150.0 | 161.7 | 151.3 | 1.00 |
| MiDM-1.7B-q3-e1 | 154.2 | 166.8 | 156.1 | 1.03 |
| MiDM-4B-q35-e1 | 300.4 | 363.6 | 309.0 | 2.04 |
| MiDM-8B-q3-e2 | 193.2 | 209.7 | 196.2 | 1.30 |

### Cascade expected latency with the measured tiers (dev escalation shares)

| cascade | tiers | dev acc | expected ms/question | top tier alone ms |
|---|---|---|---|---|
| cascadeB_0.6b_8b2 | MiDM-0.6B-q3-e1 → MiDM-8B-q3-e2 | 0.853 | 250 | 196 |
| cascadeB_1.7b_8b2 | MiDM-1.7B-q3-e1 → MiDM-8B-q3-e2 | 0.852 | 237 | 196 |
| cascadeC_0.6b_35x4b | MiDM-0.6B-q3-e1 → MiDM-4B-q35-e1 | 0.852 | 279 | 309 |
| cascadeC_0.6b_35x4b_8b2 | MiDM-0.6B-q3-e1 → MiDM-4B-q35-e1 → MiDM-8B-q3-e2 | 0.853 | 279 | 196 |
| cascadeC_35x4b_8b2 | MiDM-4B-q35-e1 → MiDM-8B-q3-e2 | 0.857 | 309 | 196 |

At batch 1 in this stack (4-bit NF4, transformers, Windows) per-call overhead dominates: the 0.6B tier costs ~150 ms and the 8B ~196 ms, and Qwen3.5-4B is slowest (~309 ms) because its linear-attention kernels fall back to the reference PyTorch implementation (no flash-linear-attention / causal-conv1d). The provisional cost weights (0.6B=1, 4B=6, 8B=12) therefore overstate the saving; with measured latency a cascade is not faster than the 8B tier alone here.

## 3. Published per-item response time on JevBench public (231 items)

From JevBench's per-task results file. Hosted endpoints, so this is end-to-end time including the network; it is not a like-for-like model speed comparison with the local numbers above.

| system | p50 s | p95 s |
|---|---|---|
| jev-1.13.0 | 0.665 | 0.804 |
| kev-8b | 0.591 | 1.925 |
| kev-4b | 0.586 | 1.645 |
| kev-0.6b | 0.587 | 1.087 |

## Notes
- At equal size, Qwen3.5-4B is about 40% slower than Qwen3-4B in this stack (70 vs 50 ms/q). Its hybrid linear-attention layers are less optimised in transformers; accuracy favours Qwen3.5.
- MiDM-4B-q35 and MiDM-8B-q3-e2 reach the same accuracy (p = 0.69); the 4B is slightly faster (70 vs 82 ms/q) and needs about half the memory.
- In the cascade, about 59% of dev questions stop at the 0.6B tier (~15 ms/q), so mean latency falls; the cascade cost will be recomputed from the measured single-question latencies.
