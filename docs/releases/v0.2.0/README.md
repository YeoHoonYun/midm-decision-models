# MiDM v0.2.0 — 9B checkpoint and completed evaluation results

Adds the Qwen3.5-9B-Base LoRA adapter and learned pointer head, self-contained loader and verified 4096-token inference configuration. The original 4B release remains available. This model scores supplied options; it does not generate SQL, Python or report prose. The report writer remains Qwen3 8B.

## Matched primary evaluation

NF4/BF16 on RTX 3090, max input 4096, batch token budget 4096, TTA=1; same items and ordering.

|Suite|N|4B|9B|
|---|---:|---:|---:|
|Typed Decisions test|2000|78.85%|79.30%|
|Kev transfer test|764|79.84%|82.59%|
|JevBench original|72|91.67%|97.22%|
|JevBench easy|48|100.00%|100.00%|
|JevBench hard|111|45.95%|54.05%|

These are retrospective hard-label accuracies, not official JevBench full/sealed composite scores. The separate 600-question Typed Decisions holdout is a development/model-selection split; it is not the public 2000-question test. Earlier development results used their recorded configurations and must not be mixed into this matched table.

## Additional completed tests and limitations

- CLM released DeepSWE verifier replay: 31/38 (81.58%) reproduced. The 13 mixed-outcome tasks give exact one-sided p=0.062 against random selection. Numerical reproduction succeeded; significance at 5% was not established.
- MiDM DeepSWE and local Terminal-Bench remain unmeasured: matching raw candidate traces/evaluation artifacts were not found. No official benchmark score is substituted.
- SQL/Python best-of-N: 4B → 9B Spider 78.34% → 77.27%, SQL holdout 81.66% → 81.77%, Python hard80 28.75% → 25.00%. Stored pools, dev-selected blend weights; no consistent gain.
- Learned routing: JevBench public 231 accuracy 77.06% MiDM, 79.65% router, 81.82% local Qwen3-4B reasoning alone. Router not promoted; no non-inferiority or latency-saving claim.
- Concurrency: two short-input 4B processes nearly doubled throughput; 2048-token 4B and 4096-token 9B lost throughput. Small synthetic single-trial pilots, not endurance or minimum-memory certification.
- Same-date report integration: 4B/9B candidate-order agreement 1/4 versus 3/4, 19/19 schema-valid sections each, one unknown fact ID each. No semantic-quality or financial-outcome superiority established; only aggregate diagnostics are released.

Training: QLoRA NF4, rank16/alpha32, one epoch, seed0, learning rate2e-4, training context1024; td_train, kev_train, breadth_v1, pp6_new. Inference context4096 matches evaluation. The earlier 1024-token release candidate was not published; package verification uses the final loader. Base weights are separate and retain their own license, as do training datasets. No raw questions, answers, financial data, reports or credentials are included.

Model: https://huggingface.co/yunicro/MiDM-9B-q35-e1-bx/tree/v0.2.0

Code and full evidence: https://github.com/YeoHoonYun/midm-decision-models/releases/tag/v0.2.0

Zenodo versioning retains concept DOI https://doi.org/10.5281/zenodo.23084583 . The v0.1.0 version DOI does not archive this new release.

Version DOI: https://doi.org/10.5281/zenodo.23162744

## Packaged-model verification

The final package reproduces the original development evaluation exactly: **513/600 (85.50%)**, identical choices on 600/600 questions and maximum absolute probability difference 0.0. The original protocol uses NF4/BF16, 4096-token context, 4096 padded tokens per batch, attention masks and TTA=1. Adapter weights, pointer head and encoded inputs match the source exactly. This is package reproducibility, not a new independent test.

A separate single-question, unmasked check gave **509/600 (84.83%)** and failed its original tolerance gate. It is retained rather than hidden. That verification bypassed the public API batching/mask path. The matched check changes batching and mask together, so their individual effects were not isolated. Do not assume identical predictions across serving configurations.
