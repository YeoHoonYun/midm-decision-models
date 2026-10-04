# MiDM v0.2.0 ? 9B typed-decision checkpoint

Adds **MiDM-9B-q35-e1-bx**, a LoRA adapter plus a pointer head for `Qwen/Qwen3.5-9B-Base`. The existing 4B release remains available. This is a direct option-scoring model, not a chain-of-thought generator.

## Matched development evaluation

| Suite | Items | Previous 4B bx | New 9B bx | Change (pp) |
|---|---:|---:|---:|---:|
| kev_dev | 1468 | 85.56% | 86.92% | +1.36 |
| kevT_dev | 764 | 78.93% | 79.06% | +0.13 |
| kevT9_dev | 1264 | 70.33% | 71.20% | +0.87 |
| td_holdout | 600 | 83.17% | 85.50% | +2.33 |
| jb_original | 72 | 91.67% | 97.22% | +5.56 |
| jb_easy | 48 | 100.00% | 100.00% | +0.00 |
| jb_hard | 111 | 45.05% | 54.05% | +9.01 |

These are the experiment-116 full-suite evaluations of the named checkpoints (one evaluation configuration, TTA=1). Typed Decisions holdout is 600 questions from 10% of training-source row groups held out for model selection, not the separate public test split. These results are development evidence, not a new independent test or statistical significance claim. Earlier 200-item Solar comparisons use a different subset. The 9B training-time holdout score was 85.17%; this separate evaluation was 85.50%, so precision/batching differences should not be mistaken for training improvement.

## Scope and limitations

- SQL/Python best-of-N selection results in v0.1.0 belong to the 4B model. The 9B checkpoint has not been evaluated on those candidate pools.
- No new 9B `td_test` or `kevT_test` result is claimed. Historical test results stay attributed to their original models.
- No financial states, financial predictions, reports, per-item answers, or private datasets are included.
- Training recipe: QLoRA NF4, rank16, one epoch, seed0, learning rate2e-4, max input1024; `td_train`, `kev_train`, `breadth_v1`, `pp6_new`. Dataset terms remain separate from the Apache-2.0 adapter/code license.
- Base weights are downloaded separately. A 24GB GPU is the deployment verification target; minimum serving memory has not been benchmarked.
- This release does not include the queued Qwen3-4B completion-marker or longer-reasoning experiments.

## Model and versioning

Model: https://huggingface.co/yunicro/MiDM-9B-q35-e1-bx (tag `v0.2.0`).
Code: https://github.com/YeoHoonYun/midm-decision-models/releases/tag/v0.2.0 .
The previous model remains https://huggingface.co/yunicro/MiDM-4B-q35-e1-bx .
The existing Zenodo v0.1.0 DOI does not archive v0.2.0. A follow-up deposit must use the existing record's **new-version** action to preserve concept DOI `10.5281/zenodo.23084583`; no new DOI is claimed here.
