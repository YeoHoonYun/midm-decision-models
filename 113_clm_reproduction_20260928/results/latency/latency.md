# Per-question latency (NVIDIA GeForce RTX 3090, torch.bfloat16, batch 1, n=300 td_holdout)

| tier | p50 ms | p90 ms | mean ms | cost ratio | peak GiB |
|---|---|---|---|---|---|
| MiDM-0.6B-q3-e1 | 150.0 | 161.7 | 151.3 | 1.00 | 0.6 |
| MiDM-1.7B-q3-e1 | 154.2 | 166.8 | 156.1 | 1.03 | 1.4 |
| MiDM-4B-q35-e1 | 300.4 | 363.6 | 309.0 | 2.04 | 3.4 |
| MiDM-8B-q3-e2 | 193.2 | 209.7 | 196.2 | 1.30 | 4.9 |
