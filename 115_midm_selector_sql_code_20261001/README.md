# Study115 — offline best-of-N candidate pools for a selector

`build_pools.py` (deterministic, CPU only, no model calls) writes `data/{spider_dev,spider_test,holdout,lcb_hard}.jsonl`
and `data/POOL_STATS.json`. Run with the 113 venv python and `$env:PYTHONPATH=''`.

Nothing here opens, copies or executes a SQLite file, executes SQL, or runs candidate code. Only stored
generations, stored execution results and stored labels are read. Study111 is not read. Not used: Study95 k10
RAG arms, `*_GOLD_INFORMED_*` banks, `101/data/examples*.json`. Solar is never a candidate; it appears only
in `baseline`.

## Record
`{id, split, db_id, question, schema, state_text, options:[{key, sql_or_code, result_preview, n_votes, arms, correct, ...}], rag:[{question, sql}], baseline:{solar, per_arm}}`

- `state_text` = `Database/Schema/Question` (SQL splits) or `Title + statement (+ starter code)` (LCB, ≤2,500 Qwen3 tokens).
- Options appear in the fixed arm order: each option's first member is the earliest arm in it, and that member is the representative. Keys are A, B, ... The runner should shuffle keys if position bias matters.
- `correct` = the representative's label. `mixed_correctness` is set when a cluster's members disagree on the label (for example, the same row multiset but an ORDER BY-sensitive gold). `executable=false` marks clusters of failed executions; these are grouped by normalized SQL text, and the vote never picks them.
- Majority vote = the largest executable cluster. Ties go to the earliest arm, and if no cluster is executable the vote takes the first option. This is the same rule as `_ops/selection_full.py`.

## Sources
| split | candidates (arm order) | clustering | label | schema | RAG |
|---|---|---|---|---|---|
| spider_dev (400) | 109/spider_dev conditions k0,k3,k5_s73,opt_k5,k7,k10 (gpt-oss low) | row multiset of stored `execution.json` (generation/<c>/cells/low, executions/opt_k5) | scores/<c>/*.json `correct` | spider_data.zip `tables.json` | top-5 TF-IDF (word uni+bigram) over Spider TRAIN questions (train_spider+train_others; 0 DB overlap) |
| spider_test (2,147) | 86 direct_low, 88 low_a, low_b, 86 direct_medium, 88 medium_a, medium_b, 87 off, 90 off_a, off_b | row multiset of stored `execution.json` (resolved as `_ops/score_full_spider.resolve`) | 109/spider/cache `per_q[qid].f[0]` (0 disagreements with study scores) | `test_tables.json` | same |
| holdout (941 = 265+676) | oss_low, oss_medium, qwen_off, coder32b, and the `*_rag_bankv2` of each (109/rag_v2) | normalized SQL text (no stored rows; holdout fixtures were not executed) | 109/holdout/RESCORE_PER_TASK `corrected_strict` (None on 14 nondeterministic 265 items; `correct_unified_strict` also given) | DDL the prompts used: 101/data/schemas.json (265), `SCHEMAS` literals of 101/source/domains_d1.py read with `ast` (676) | first 5 of 109/bank_v2/examples_{265,676}.json |
| lcb_hard (80) | 112 oss_low_direct_p32k, oss_low_idea_k5_p32k, oss_low_full_k5_p32k | none (3 programs, code extracted with 112 grader's `extract`, ≤1,200 tokens) | SCORE_<arm>.json `solved` | — | none |

Solar baselines: 92 solar_direct (test), 101 solar_direct corrected_strict (holdout), 112 solar_direct (+ `solar_rmedium`) for LCB, none for dev.
Token counts use the Qwen3 tokenizer, because no Mi:dm tokenizer exists locally. See POOL_STATS.json for the stats and for the sanity check against `_ops/scoring/SELECTION_FULL.json`.
