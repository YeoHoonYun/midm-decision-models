# Frozen-date report integration audit

Market observation date 2026-09-28; 5/20-session horizons. Original data, RAG evidence, prompts, Qwen3 8B writer, seed and context frozen. Fresh 4B and 9B selectors use the same RTX 3090, NF4/BF16 and eight requests. Each report has 19 newly generated sections, two writer jobs in parallel. Historical 0.6B uses CPU FP32 and is a separate reproducibility reference; its eight choices and all scores reproduced exactly. Historical writing mixed CPU/GPU, so its total latency is not directly comparable.

|Metric|4B|9B|
|---|---:|---:|
|Eight selector requests, seconds (excluding model load)|4.994|5.167|
|Nineteen drafting sections, warm seconds|21.429|19.447|
|Same choice under forward/reversed candidates|1/4|3/4|
|Schema-valid sections|19/19|19/19|
|Unknown fact identifiers|1|1|
|Truncated sections|0|0|

Two of eight choices changed. All fresh scores match the rendered UI; three viewport sizes passed without external render requests or page errors. These are one-run diagnostic measurements, not confidence intervals or financial outcome accuracy. One unknown fact ID remains per report; schema/citation-ID checks are not semantic entailment checks. Twelve of eighteen unchanged-prompt sections differ on regeneration, so prose changes and small latency differences cannot be attributed solely to selector version. No semantic-quality, investment-return or calibrated-probability improvement is claimed.

Only these aggregate checks are public. Financial states, predictions, selections, RAG text, generated prose, reports, prompts and local databases are excluded. See [machine-readable aggregates](report_integration.json).
