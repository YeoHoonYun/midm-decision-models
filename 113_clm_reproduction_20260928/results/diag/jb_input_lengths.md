# JevBench public: MiDM input length vs the 1,024-token limit

| subset | n | median tokens | max tokens | over 1,024 |
|---|---|---|---|---|
| jb_easy | 48 | 77 | 97 | 0 |
| jb_hard:adversarial | 6 | 249 | 270 | 0 |
| jb_hard:ambiguous | 7 | 463 | 601 | 0 |
| jb_hard:judge_hard | 17 | 260 | 365 | 0 |
| jb_hard:long_policy | 19 | 2602 | 3897 | 19 |
| jb_hard:multi_hop | 18 | 2294 | 2930 | 17 |
| jb_hard:probability | 10 | 985 | 1885 | 4 |
| jb_hard:routing_hard | 5 | 255 | 274 | 0 |
| jb_hard:temporal_numeric | 15 | 562 | 1269 | 1 |
| jb_hard:tradeoff | 6 | 725 | 806 | 0 |
| jb_hard:trap | 8 | 268 | 317 | 0 |
| jb_original | 72 | 77 | 105 | 0 |

long_policy + multi_hop: 36 of 37 exceed 1,024 tokens (full MiDM input (state + instructions + 'Options:' + all option lines), Qwen3.5-4B-Base tokenizer).
