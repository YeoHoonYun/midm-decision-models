# breadth_v1

This is a training-only set of 6,000 typed decision questions (one per row, in the Kev System One format). It is built
to balance the task types that the knowledge-MC sources in Breadth D1 diluted. Built on 2026-10-01 by
`build_breadth_v1.py` (CPU only, seed 20261001). All rows come from the HF **train** split.

| source | repo (config) | task type | q type | n | labels |
|---|---|---|---|---|---|
| mrpc | nyu-mll/glue (mrpc) | sentence pair / paraphrase | noul | 1000 | 500 / 500 |
| qqp | nyu-mll/glue (qqp) | sentence pair / paraphrase (questions) | choice (2) | 1000 | 500 / 500 |
| rte | nyu-mll/glue (rte) | sentence pair / entailment | noul | 1000 | 500 / 500 |
| go_emotions | google-research-datasets/go_emotions (simplified) | affect, 28 fine-grained labels | choice (28) | 1000 | 35–36 per label |
| civil_comments | google/civil_comments (train shard 0) | toxicity | noul | 1000 | 500 / 500 |
| winogrande | allenai/winogrande (winogrande_xl) | commonsense coreference | choice (2) | 1000 | 500 / 500 |

`manifest.json` holds the revisions, label maps and the full contamination report.

Design notes:
- The instruction templates differ on purpose from the eval templates for paws ("Does this sentence mean the same
  thing…"), qnli and tweet_offensive ("Is this post offensive?"). This avoids template leakage.
- go_emotions keeps only single-label rows, and its label space (28 labels) differs from the 6 labels in dair-ai/emotion.
- civil_comments counts toxicity ≥ 0.5 as toxic and toxicity = 0.0 as clean. Rows in between are dropped as ambiguous.
  Texts are limited to 30–700 chars.
- No row uses mmlu, mmlu_pro, sciq, dair-ai/emotion, tweet_eval, glue/qnli, PAWS, the Kev holdout, buried or
  unknowable records, JevBench, or typed-decisions test. Some qqp questions keep the source's CSV-style doubled quotes.

Contamination: every row was checked against kevT_dev, kevT_test, kevT9_dev, jb_original/easy/hard, td_test and kev_dev.
The checks were exact and normalized hashes of state texts, state leaves and hypotheses; hashes against eval candidate
texts over 20 chars; and 13-word shingles. **There were 0 hits in the candidate pools and 0 in the final file.** The checker
caught 120/120 eval rows fed back to it, 30/30 case- and whitespace-mangled eval states, and 17/17 embedded
14-word excerpts. There is also 0 normalized-state overlap with kev_train and public-pool-v6 train.

## Proposed loader branch (for `decision_data.load`)

```python
    if suite == "breadth_v1":   # balanced-breadth extra training sources (never an eval suite)
        return list(_kev(_rows(os.path.join(HERE, "data", "breadth_v1", "train.jsonl")), suite))
```

Verified: `build_pairs` parses all 6,000 questions, every label is a criteria key, and `_kev` yields 6,000 examples.
