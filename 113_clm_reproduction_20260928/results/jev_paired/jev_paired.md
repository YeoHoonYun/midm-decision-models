# JevBench public items: per-item paired comparison with Jev 1.13.0

Jev and Kev outcomes are JevBench's published per-item results (v1.2 per-task file); ours are our own runs on the same 231 public items.

| subset | n | jev-1.13.0 | kev-8b | kev-4b | MiDM-4B-q35-e1-bx | MiDM-4B-q35-e1 | MiDM-8B-q3-e2 |
|---|---|---|---|---|---|---|---|
| all public (231) | 231 | 0.866 | 0.714 | 0.662 | 0.710 | 0.710 | 0.706 |
| tier original | 72 | 0.986 | 0.931 | 0.889 | 0.917 | 0.931 | 0.944 |
| tier easy | 48 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tier hard | 111 | 0.730 | 0.450 | 0.369 | 0.450 | 0.441 | 0.423 |
| hard: adversarial | 6 | 1.000 | 0.500 | 0.667 | 1.000 | 1.000 | 1.000 |
| hard: ambiguous | 7 | 0.857 | 0.429 | 0.286 | 0.429 | 0.429 | 0.571 |
| hard: judge_hard | 17 | 0.765 | 0.529 | 0.412 | 0.529 | 0.529 | 0.529 |
| hard: long_policy | 19 | 0.632 | 0.474 | 0.263 | 0.211 | 0.211 | 0.105 |
| hard: multi_hop | 18 | 0.833 | 0.333 | 0.278 | 0.500 | 0.500 | 0.222 |
| hard: probability | 10 | 0.700 | 0.500 | 0.400 | 0.300 | 0.400 | 0.700 |
| hard: routing_hard | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hard: temporal_numeric | 15 | 0.267 | 0.000 | 0.067 | 0.000 | 0.000 | 0.067 |
| hard: tradeoff | 6 | 0.833 | 0.333 | 0.000 | 0.500 | 0.333 | 0.333 |
| hard: trap | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 0.875 | 0.875 |

Paired (MiDM-4B-q35-e1-bx minus Jev, bootstrap over items):

- all public (231): -15.6 pp [-20.8, -10.8]; only MiDM correct 2, only Jev correct 38
- tier hard: -27.9 pp [-36.9, -19.8]; only MiDM correct 1, only Jev correct 32
- tier original: -6.9 pp [-13.9, +0.0]; only MiDM correct 1, only Jev correct 6
