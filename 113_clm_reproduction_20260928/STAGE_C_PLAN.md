# Stage C plan: adapter composition (per-domain adapters, routing, merging, merge-then-continue)

Code: `adapters_c.py` (new; imports `pointer_lora.Encoder/forward/batches/evaluate/load_model/save`
unchanged). `pointer_lora.py`, `decision_data.py`, `gpuq.py` are not modified. Nothing here has run on a GPU yet.
Checked so far: `ast` parse, `--help` for every subcommand, and CPU `dry-run` (tokenizer only).

## Protocol (same as stage A)

- **Proxy model:** Qwen3-1.7B, fp16 (no `PTR_BF16`), stage-A recipe: LoRA r16 on all projections,
  lr 2e-4, head lr x10, cosine with 30 warm-up steps, 4096-token batches, max-len 1024, seed 0, **2 epochs**.
- **Transfer proxy:** `--holdout-sources boolq mnli sst5` on every 1.7B arm. These sources are never trained;
  their kev_dev rows form `src_holdout` (n=276).
- **Select sets:** `td_holdout` (the 10% td_train groups under the seed-0 rule, n=600) and `kev_dev500`
  (seed-0 500 kev_dev rows without the held-out sources). Both are built by the same code as
  `pointer_lora.cmd_train`. The dry run reproduces stage A exactly for the `all` spec: 17,976 items and
  4,129,055 tokens, identical to the A1–A8 logs.
- **Test suites:** `td_test`, `kevT_*` and `jb_*` are refused unless `--final` is passed. They are read once,
  only for an arm that has already been promoted.
- **Checkpoint rule:** fixed in advance as `final`. The `stepN` checkpoints exist only for curves.
- **Selection metric:** primary = mean(td_holdout acc, kev_dev500 acc); secondary = src_holdout acc.
  `adapters_c` also logs `in_domain` (trained sources) and `in_domain_unseen_src` slices.
- **Baseline:** A7 (1.7B, LoRA r16 all, 2 epochs, same holdout): td_holdout 0.848, kev_dev500 0.880,
  so primary **0.864**; src_holdout **0.583**. A1 (1 epoch) scored 0.825 / 0.848 / 0.543.
- **Promotion rule (1.7B to 4B):** primary >= 0.874 (+1.0 pt) with src_holdout >= 0.573, or
  src_holdout >= 0.613 (+3 pt) with primary >= 0.864.
  - The standard error is about 1 pt on the primary mean and about 3 pt on src_holdout (n=276).
  - Confirm a candidate with a paired bootstrap on `--save-probs` rows before promoting it. `route-eval`
    scores A7 on the same rows.
- **Compute parity:** the two domain adapters see 2 x 2.235M + 2 x 1.894M = 8.26M tokens. A7 saw
  2 x 4.13M = 8.26M tokens, so C1 matches A7 exactly.

## Domain presets (dry run, Qwen3 tokenizer)

| preset | sources | rows / tokens per epoch, holdout boolq mnli sst5 | rows / tokens, no holdout (stage-B protocol) |
|---|---|---|---|
| classification | agnews yelp dbpedia14 banking77 trec imdb sst5 amazon | 10,000 / 2.235M (782 steps) | 11,000 / 2.305M (832 steps) |
| reasoning_policy | mnli boolq compositional legacy_policy typed-decisions | 7,976 / 1.894M (614 steps) | 9,976 / 2.164M (733 steps) |
| all | everything | 17,976 / 4.129M (1460 steps) | 20,976 / 4.469M |

Under the stage-A holdout, reasoning_policy trains only on compositional, legacy_policy and typed-decisions.
That is intended: boolq and mnli then test whether the reasoning adapter transfers to unseen sources in its
own domain.

## Library facts verified in `.venv`

Versions: peft 0.21.0, bitsandbytes 0.50.2, transformers 5.17.0, torch 2.14.0+cu126, accelerate 1.15.0, Python 3.12.13.

**`LoraModel.add_weighted_adapter` works on a 4-bit bnb base for every method.** No method reads or writes
the base weight (`peft/tuners/lora/model.py`, lines 681–951):

- **Factor-space methods** (`linear`, `ties`, `dare_linear`, `dare_ties`, `magnitude_prune`):
  - They combine the A and B factors separately, with `sqrt(w * scaling)` applied to each.
  - They require equal ranks.
  - Linear is therefore **not** the average of dW: `B'A'` contains cross terms `B_i A_j`.
  - All C1 adapters use seed 0, so their `lora_A` initialisations are identical, which makes the factors
    roughly aligned. Linear is still an approximation.
- **Delta-space methods** (`cat`, `svd`, `ties_svd`, `dare_linear_svd`, `dare_ties_svd`, `magnitude_prune_svd`):
  - They use `get_delta_weight` = `B @ A * scaling` (for bnb `Linear4bit`, `peft/tuners/lora/bnb.py` line 449),
    so only LoRA tensors are involved.
  - `cat` is exact: its rank is the sum of the input ranks, and it gives exactly `sum_i w_i dW_i`.
  - The `svd` variants truncate the result to `--svd-rank`. `adapters_c` passes `svd_full_matrices=False`,
    because the default full U for a 6144x2048 or 9728x2560 delta is wasteful.
- **Weight defaults:** PEFT's ties and dare_ties `disjoint_merge` already averages the agreeing entries, so
  `adapters_c` defaults to weight 1 each for the ties family and 1/n each for the other methods.
- **DARE** is random, so the merge seeds torch first.

**`merge_and_unload()` also works on 4-bit, but it re-quantizes.** `bnb.Linear4bit.merge` dequantizes W,
adds dW, then builds a new `Params4bit`, which quantizes to NF4 again (`bnb.py` lines 355–412). The problem:

- Q(W) values sit exactly on NF4 levels, spaced roughly 0.08 x the block absmax near zero.
- A LoRA dW is usually far smaller than half that spacing.
- So Q(Q(W)+dW) mostly rounds back to Q(W), and the requant mode is expected to **erase most of adapter A**.
- `continue --mode requant` measures this with the `delta_retention` diagnostic before merging
  (kept = <W1 − W0, dW> / |dW|², on 12 sampled layers) and evaluates A at step 0.

**Exact merge-then-continue therefore takes one of two forms:**

- **stack** (default, recommended):
  - Setup: 4-bit Q(W) + A frozen + B trainable, both active through `LoraModel.set_adapter(["default", "B"])`.
    PeftModel's own `active_adapter` stays the string `"default"`, so forward and `disable_adapter` work.
  - Exactness: this is mathematically Q(W)+dA+dB, exactly what a full-precision merge of A would give. The
    only departure is that A's dropout is set to identity.
  - Memory: stage-A memory plus about 0.1 GiB.
  - Export: at the end, B is exported, plus `final/AB` = `cat(A, B)` with weights 1, 1. That is an exact
    single adapter that `pointer_lora.py eval --ckpt .../final/AB` can load.
- **dequant:**
  - Setup: `model.dequantize(dtype)` (transformers 5.17 `PreTrainedModel.dequantize`, i.e. bnb
    `dequantize_and_replace`), then a PEFT merge of A in fp16 or bf16, then B on the unquantized base.
  - `prepare_model_for_kbit_training` is deliberately **not** called here: on a non-quantized model it
    upcasts every fp16 parameter to fp32 (`peft/utils/other.py`). Gradient checkpointing is enabled with
    `use_reentrant=False`.
  - Caveat: with `PTR_BF16=1` the in-place bf16 merge rounds dW coarsely, since the bf16 ulp near 0.02 is
    about 1e-4. fp16 is fine.

## GPU memory estimates

| setting | Qwen3-1.7B | Qwen3-4B | Qwen3-8B |
|---|---|---|---|
| train-domain / stage-A recipe (4-bit, budget 4096) | 3.85 GiB (measured) | 6.43 GiB (measured, 2080 Ti) | 10.23 GiB (measured, bf16, 3090) |
| continue `stack` | ~3.95 GiB | ~6.6 GiB | ~10.4 GiB |
| continue `requant` | as stack (A mostly lost) | as stack | as stack |
| continue `dequant` (fp16 weights + ~activations) | 3.8 + ~2 = **~6 GiB** (2080 Ti ok) | 7.5 + ~3.3 = **~11 GiB: not on the 11 GB 2080 Ti**; 3090 ok | 14.1 + ~4.5 = **~19 GiB: 3090 only, and only with the exp-34 Ollama unloaded** |
| merge / route-eval (4-bit inference, several r16 adapters) | ~2.5–3.5 GiB | ~4–5 GiB | ~6–8 GiB |

- The dequant rows count the AutoModel parameters (no lm_head) at 2 bytes each: 2.03B, 4.02B and 7.57B.
- Activation headroom is taken from the measured 4-bit runs.
- The dequant step itself swaps one layer at a time, so its peak is about the fp16 size.

## GPU-hours

Throughput used: 1.7B at 1450 tok/s on the 2080 Ti and 1810 tok/s on the 3090 (measured, fp16).
4B at 650–717 tok/s on the 2080 Ti and about 1300 on the 3090. Select evaluation costs about 1 min
per pass at 1.7B.

| job | 1.7B, 2080 Ti | 1.7B, 3090 | 4B, 2080 Ti | 4B, 3090 |
|---|---|---|---|---|
| train-domain classification, 2 epochs | ~0.95 h | ~0.75 h | ~2.0 h (no holdout, 4.61M tok) | ~1.0 h |
| train-domain reasoning_policy, 2 epochs | ~0.8 h | ~0.65 h | ~1.85 h (4.33M tok) | ~0.95 h |
| route-eval on select sets (2 domain adapters + 2 singles + centroid router) | ~0.2 h | ~0.15 h | ~0.4 h | ~0.25 h |
| merge, 5 methods, head tune 300 steps each | ~0.6 h | ~0.5 h | ~1.2 h | ~0.7 h |
| continue stack, 730 steps (~0.5 epoch all sources) | ~0.5 h | ~0.4 h | ~1.0 h | ~0.5 h |

## Experiments, in order of expected value

All commands run from the experiment folder. They use priority 6, which is below the queued stage-B jobs
(8–10) and above A9 (5); use `--priority 10` to jump ahead. Omit `--env PTR_BF16=1` for 1.7B, so the
comparison with the fp16 A7 run stays clean. A queue job starts only when its `--mem` is free and the GPU
is idle, so the domain pair can run in parallel on gpu0 and gpu1.

### C1: two domain adapters, 1.7B, 2 epochs, stage-A holdout (prerequisite for everything else)

```
.venv\Scripts\python.exe gpuq.py submit --queue gpu1 --mem 5 --priority 6 --retries 1 --name C1a_cls_1p7b_2ep --owner midm -- python adapters_c.py train-domain --model Qwen/Qwen3-1.7B --domain classification --holdout-sources boolq mnli sst5 --epochs 2 --eval-every 400 --out runs/C/C1a_cls_1p7b
.venv\Scripts\python.exe gpuq.py submit --queue gpu0 --mem 5 --priority 6 --retries 1 --name C1b_rp_1p7b_2ep --owner midm -- python adapters_c.py train-domain --model Qwen/Qwen3-1.7B --domain reasoning_policy --holdout-sources boolq mnli sst5 --epochs 2 --eval-every 400 --out runs/C/C1b_rp_1p7b
```

- **Metric:** `in_domain` of each adapter against A7's accuracy on the same slice. C2 gives that per-slice
  number for A7.
- **Gate:** C1 is only a stepping stone. Nothing is promoted from C1 on its own.

### C2: routing (highest expected value); submit after C1a and C1b finish

```
.venv\Scripts\python.exe gpuq.py submit --queue gpu1 --mem 5 --priority 6 --retries 1 --name C2_route_1p7b --owner midm -- python adapters_c.py route-eval --model Qwen/Qwen3-1.7B --adapters runs/C/C1a_cls_1p7b/final runs/C/C1b_rp_1p7b/final --single A7=runs/search/A7_lora_r16_2ep/final --router both --suites select kev_dev --out results/stage_c/C2_route_1p7b.json --save-probs results/stage_c/C2_route_1p7b.jsonl
```

- **What it reports per suite:**
  - each adapter on every row;
  - `route_tag` (routing by source tag; unknown sources fall back to the centroid router);
  - `route_centroid`, built from nearest source centroids of the base model's mean-pooled hidden states,
    with adapters disabled and 200 train rows per source;
  - `router_agree_with_tag`;
  - `ensemble_mean`;
  - the oracle (diagnostic only);
  - `by_tag_domain` slices.
- **Promote** `route_tag` or `route_centroid` if the promotion rule holds against A7 in the same file, and
  `router_agree_with_tag` >= 0.95 for the centroid router. A lower agreement means the router is the
  bottleneck.
- **Why src_holdout matters here:** in route_tag, src_holdout rows go to the adapter of their domain even
  though that source was never trained. This is the cleanest test of whether specialization helps unseen
  sources.

### C3: merges; submit after C1

```
.venv\Scripts\python.exe gpuq.py submit --queue gpu1 --mem 5 --priority 6 --retries 1 --name C3_merge_1p7b --owner midm -- python adapters_c.py merge --model Qwen/Qwen3-1.7B --adapters runs/C/C1a_cls_1p7b/final runs/C/C1b_rp_1p7b/final --methods cat svd ties_svd dare_ties linear --density 0.5 --head tune --head-steps 300 --out runs/C/C3_merge_1p7b
```

- **Output:** one PEFT adapter per method at `runs/C/C3_merge_1p7b/<method>/`, which pointer_lora can load,
  plus `pointer_head.pt` and `adapter_c.json`.
- **Heads:** `train_summary.json` records `select_avg_head` (the weighted head average) and `select`
  (after 300 head-only steps on frozen merged features).
- **Expectations:**
  - `cat` is the exact average of dW, rank 32.
  - `svd` is its rank-16 truncation.
  - `ties_svd` and `dare_ties` resolve sign conflicts.
- **Promote** the best method to C4. It goes straight to 4B only if it meets the promotion rule, which is
  unlikely: merges usually land 1–3 pt below joint training.
- **Cheap follow-up:** rerun with `--weights 1 1 --methods cat svd` to test the task-arithmetic sum instead
  of the average.

### C4: merge-then-continue (heal), stack mode; submit after C3 (replace `<best>`)

```
.venv\Scripts\python.exe gpuq.py submit --queue gpu1 --mem 5 --priority 6 --retries 1 --name C4a_heal_1p7b --owner midm -- python adapters_c.py continue --model Qwen/Qwen3-1.7B --base-adapter runs/C/C3_merge_1p7b/<best> --mode stack --domain all --holdout-sources boolq mnli sst5 --max-steps 730 --epochs 1 --eval-every 0 --out runs/C/C4a_heal_1p7b
.venv\Scripts\python.exe gpuq.py submit --queue gpu0 --mem 5 --priority 6 --retries 1 --name C4b_ctrl_A7plus_1p7b --owner midm -- python adapters_c.py continue --model Qwen/Qwen3-1.7B --base-adapter runs/search/A7_lora_r16_2ep/final --mode stack --domain all --holdout-sources boolq mnli sst5 --max-steps 730 --epochs 1 --eval-every 0 --out runs/C/C4b_ctrl_A7plus_1p7b
```

- **Arms:**
  - C4a = merged domain adapters plus a fresh B trained for about 0.5 epoch on all sources.
  - C4b is the compute control: A7 plus a fresh B for 0.5 epoch.
  - The queued A9 (3-epoch plain LoRA) is a second control.
- **Promote** C4a if it beats both C4b and A7 by the promotion rule. Otherwise, conclude that composition
  adds nothing at 1.7B beyond extra steps.
- **Step 0:** each history begins with the A-only select eval, i.e. B = 0 at step 0.
- **Evaluation:** `final/AB` is an exact single adapter, so either of these works:
  `route-eval --single C4a=runs/C/C4a_heal_1p7b/final/AB ...` or `eval-compose --ckpt runs/C/C4a_heal_1p7b/final`.

### C5: sequential curriculum (reasoning first, then classification on top), stack mode

```
.venv\Scripts\python.exe gpuq.py submit --queue gpu0 --mem 5 --priority 6 --retries 1 --name C5_seq_rp_then_cls_1p7b --owner midm -- python adapters_c.py continue --model Qwen/Qwen3-1.7B --base-adapter runs/C/C1b_rp_1p7b/final --mode stack --domain classification --holdout-sources boolq mnli sst5 --epochs 2 --eval-every 400 --out runs/C/C5_seq_1p7b
```

- **Question:** does a frozen reasoning adapter keep td_holdout while B adds classification?
- **Compute:** C1b + C5 together use about the same compute as A7.
- **Promote** by the same rule. Also report any td_holdout drop against C1b alone, which would show
  forgetting through the shared head.

### C6: requant / dequant probes (informational, lowest value, short)

```
.venv\Scripts\python.exe gpuq.py submit --queue gpu1 --mem 5 --priority 4 --retries 0 --name C6a_requant_probe_1p7b --owner midm -- python adapters_c.py continue --model Qwen/Qwen3-1.7B --base-adapter runs/C/C1b_rp_1p7b/final --mode requant --domain classification --holdout-sources boolq mnli sst5 --max-steps 20 --eval-every 0 --out runs/C/C6a_requant_probe
.venv\Scripts\python.exe gpuq.py submit --queue gpu1 --mem 7 --priority 4 --retries 0 --name C6b_dequant_probe_1p7b --owner midm -- python adapters_c.py continue --model Qwen/Qwen3-1.7B --base-adapter runs/C/C1b_rp_1p7b/final --mode dequant --domain classification --holdout-sources boolq mnli sst5 --max-steps 20 --eval-every 0 --out runs/C/C6b_dequant_probe
```

- **Read-outs:** `info.delta_retention` (C6a) and the step-0 select accuracy of both, compared with C1b's
  final.
- **Expected:** C6a loses most of A (kept close to 0 and step-0 accuracy near the base-with-A-head level).
  C6b matches C1b to within rounding.
- These probes document the memory and precision facts. They are not candidates.

### C7: promotion to Qwen3-4B (only for an arm that passed at 1.7B)

- **Protocol:** stage B, i.e. all sources with no `--holdout-sources`; select on td_holdout + kev_dev500.
- **Baseline:** B2 (Qwen3-4B plain LoRA, 2 epochs, job 44).
- **Training commands:**

```
.venv\Scripts\python.exe gpuq.py submit --queue gpu1 --mem 8 --priority 6 --retries 1 --name C7a_cls_4b_2ep --owner midm -- python adapters_c.py train-domain --model Qwen/Qwen3-4B --domain classification --epochs 2 --eval-every 400 --out runs/C/C7a_cls_4b
.venv\Scripts\python.exe gpuq.py submit --queue gpu0 --mem 9 --priority 6 --retries 1 --name C7b_rp_4b_2ep --owner midm --env PTR_BF16=1 -- python adapters_c.py train-domain --model Qwen/Qwen3-4B --domain reasoning_policy --epochs 2 --eval-every 400 --out runs/C/C7b_rp_4b
```

- **Then:**
  - Run the promoted C2, C3 or C4 variant with `--model Qwen/Qwen3-4B`, selecting on `--suites select kev_dev`.
  - If it beats B2 by at least 1 pt on primary, read the test suites once.
- **Final read (once):**

```
... -- python adapters_c.py route-eval --model Qwen/Qwen3-4B --adapters runs/C/C7a_cls_4b/final runs/C/C7b_rp_4b/final --single B2=runs/<B2 dir>/final --router both --final --suites td_test kevT_dev kevT_test kevT9_dev jb_original jb_easy jb_hard --out results/stage_c/FINAL_C7_route_4b.json --save-probs results/stage_c/FINAL_C7_route_4b.jsonl
```

- **Precision caveat:** C7b trains in bf16 on the 3090 while C7a is fp16. For a clean pair, run both on the
  same queue, or both without `PTR_BF16`. The 3090 runs fp16 fine at 4B; bf16 is only needed for stability
  at higher rank.
- **4B dequant:** `continue --mode dequant` does not fit the 2080 Ti. Use stack.

## Known limits

- The queue has no job dependencies, so submit C2–C5 only after their inputs exist.
- The centroid router uses base-model mean pooling over the whole sequence, including options, with one
  centroid per trained source. Held-out sources, kevT and jb rows are routed by nearest centroid.
- `eval-compose` rebuilds `dequant` outputs as the 4-bit stack Q(W)+A+B, which equals the fp16 training
  base up to rounding. `requant` outputs are rebuilt by repeating the deterministic 4-bit merge.
- The step-0 A-only eval in `continue` costs about 1 min at 1.7B. Disable it with `--no-eval-start`.
