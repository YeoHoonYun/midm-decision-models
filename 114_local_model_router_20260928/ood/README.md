# Automatic unknown-domain (OOD) gate for decider-auto

Follow-up to DECISIONS.md D1a. The question: can a per-request gate decide whether the cascade is safe
without a domain tag? The analysis is CPU only and uses cached embeddings and probability dumps. Nothing
was trained, and no files in 113 or 114 outside `ood/` were changed.

Run it again with:
```
$env:PYTHONPATH=''; $env:CUDA_VISIBLE_DEVICES=''
..\..\113_clm_reproduction_20260928\.venv\Scripts\python.exe ood_gate.py   # ~3 min, writes results.json, run.log
```

## Method
- **Reference set.** This is what the pointer tiers were trained on: td_train minus the td_holdout
  groups, plus kev_train, giving 20,556 unique state texts. td_holdout is excluded so that dev distances
  are honest.
- **Embedding.** The cached state-text embedding from exp 113 `runs/emb_cache` (final-layer last token,
  L2-normalised, tail 2048 tokens). Four encoders: Qwen3-Embedding-0.6B and base Qwen3-0.6B / 1.7B / 4B.
- **Scores**, where higher means more OOD:
  - `knnK`: mean cosine distance to the K nearest reference states (K = 1, 5, 10).
  - `maha_pca`: Mahalanobis distance in the top-64 PCA space of the reference set.
  - `maha_pca+resid`: the same plus an isotropic residual term.
  - `<score>+unc[small]`: dev-z-scored distance plus dev-z-scored (1 − max prob) of the small tier.
- **Threshold rule (dev only).** t = the q-quantile of the score over kev_dev + td_holdout, with
  q ∈ {0.90, 0.95, 0.99}. q = 0.95 was the rule stated in advance. td_test, kevT_* and jb_* were never
  used to set a threshold.
- **Policy.** If score ≤ t, the question goes through the cascade with the D1 thresholds (not re-tuned):
  1.7B → 8B at 0.58, or 0.6B → 4B at 0.62. Otherwise it goes straight to the top tier (8B or 4B).
- **Costs.** Provisional costs are 0.6B = 1, 1.7B = 2, 4B = 6, 8B = 12. An escalated question pays for
  both tiers. The cost of the gate's own embedding is not included in the tables; see the caveats.
- **Baselines.**
  - top tier alone;
  - cascade always;
  - domain-tag oracle (D1a): td/kev suites are known, kevT/jb are unknown.
- **AUROC** of td_test (in-distribution eval) against the pooled OOD suites (kevT_dev, kevT_test,
  kevT9_dev, jb_*). This is a diagnostic only.
- **Alignment.** Probability dumps are aligned to `decision_data.load()` by position, and qid and label
  are checked. All rows are used (see Finding 2 on `calibrate_cascade.py`).

## Results
"OOD" means the pooled kevT_dev + kevT_test + kevT9_dev + jb_original/easy/hard, weighted by question
count (n = 3,023, of which 2,792 are kevT). td_test n = 2,000. Dev = kev_dev + td_holdout (n = 2,068).
Costs are mean cost per question.

### 1.7B → 8B (thr 0.58)
| policy | dev acc | td_test acc | td_test cost | OOD acc | OOD → cascade | OOD cost |
|---|---|---|---|---|---|---|
| 8B alone | 0.849 | 0.794 | 12.00 | **0.705** | 0% | 12.00 |
| cascade always | 0.839 | 0.795 | 7.70 | 0.646 | 100% | 6.18 |
| domain-tag oracle (D1a) | 0.839 | 0.795 | 7.70 | 0.705 | 0% | 12.00 |
| **gate: Qwen3-0.6B knn10, q0.95 (t = 0.0246)** | 0.842 | 0.795 (100% casc.) | 7.70 | 0.687 | 19% | 10.48 |
| gate: Qwen3-0.6B maha_pca+resid, q0.95 | 0.842 | 0.795 | 7.70 | 0.690 | 17% | 10.65 |
| gate: Qwen3-Emb-0.6B knn1, q0.95 | 0.841 | 0.795 | 7.70 | 0.691 | 19% | 10.62 |
| gate: Qwen3-Emb-0.6B knn10, q0.95 | 0.842 | 0.795 | 7.70 | 0.678 | 36% | 9.48 |
| gate: Qwen3-4B knn10, q0.95 | 0.842 | 0.795 | 7.70 | 0.687 | 20% | 10.48 |
| gate: Qwen3-0.6B knn10, q0.99 | 0.839 | 0.795 | 7.70 | 0.681 | 27% | 10.02 |
| gate: Qwen3-Emb-0.6B knn10, q0.99 | 0.840 | 0.795 | 7.70 | 0.657 | 84% | 6.90 |

Per suite for the recommended gate (Qwen3-0.6B knn10 q0.95; acc / share to cascade / cost), with
8B alone and cascade always for comparison:

| suite | n | gate | 8B alone | cascade always |
|---|---|---|---|---|
| td_test | 2000 | 0.795 / 100% / 7.7 | 0.794 | 0.795 / 7.7 |
| kevT_dev | 764 | 0.705 / 18% / 10.5 | 0.726 | 0.664 / 5.5 |
| kevT_test | 764 | 0.745 / 17% / 10.7 | 0.764 | 0.686 / 6.1 |
| kevT9_dev | 1264 | 0.639 / 25% / 10.1 | 0.657 | 0.611 / 6.8 |
| jb_original | 72 | 0.958 / 0% / 12 | 0.958 | 0.847 / 4.3 |
| jb_easy | 48 | 1.000 / 0% / 12 | 1.000 | 1.000 / 2.0 |
| jb_hard | 111 | 0.396 / 0% / 12 | 0.396 | 0.378 / 8.2 |
| (dev) kev_dev | 1468 | 0.845 / 93% / 4.3 | 0.857 | 0.841 / 3.6 |

### 0.6B → 4B (thr 0.62, the fallback pair)
| policy | td_test acc | td_test cost | OOD acc | OOD → cascade | OOD cost |
|---|---|---|---|---|---|
| 4B alone | 0.788 | 6.00 | 0.676 | 0% | 6.00 |
| cascade always | 0.784 | 4.77 | 0.661 | 100% | 4.34 |
| domain-tag oracle | 0.784 | 4.77 | 0.676 | 0% | 6.00 |
| gate: Qwen3-0.6B knn10, q0.95 | 0.784 | 4.77 | 0.666 | 19% | 5.70 |
| gate: Qwen3-0.6B knn10, q0.90 | 0.784 | 4.77 | 0.671 | 14% | 5.77 |

### AUROC, td_test vs OOD (diagnostic)
| encoder | knn1 | knn10 | maha_pca | maha_pca+resid | knn10+unc[1.7B] |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | 0.964 | 0.961 | 0.381 | 0.952 | 0.903 |
| Qwen3-0.6B | 0.972 | 0.976 | 0.810 | **0.989** | 0.908 |
| Qwen3-1.7B | 0.981 | 0.977 | 0.548 | 0.985 | 0.892 |
| Qwen3-4B | 0.980 | 0.981 | 0.685 | 0.979 | 0.919 |

kevT9_dev is the hardest suite to separate (0.93–0.98). JevBench is separated almost perfectly
(≥ 0.99). A Mahalanobis distance restricted to the PCA subspace fails; the OOD signal lives in the
residual directions. Adding the small tier's uncertainty lowers AUROC and does not improve policy
accuracy: in-distribution the tier is not much more confident than OOD, which is the overconfidence
behind D1a.

## Findings
1. **The distance gate is close to the domain-tag oracle in distribution.** All td_test questions are
   routed to the cascade, so td_test keeps the full cascade saving (0.795 at 64% of 8B cost).
   Off distribution it recovers most of the cascade's loss: OOD accuracy is 0.687–0.691, against 0.705
   for 8B alone and 0.646 for cascade always. That is about 1.5 pp below 8B instead of 6 pp, at 87–89%
   of 8B cost on OOD.
   - The remaining loss comes from the 17–25% of Kev-transfer questions that look like training
     states: they share the Kev format and some come from related sources.
   - JevBench is always sent to 8B.
2. **Side finding: `calibrate_cascade.py` collapses rows.** `load()` keys rows by (suite, qid, group),
   and several suites have duplicates: kev_dev 1468 → 1120, kevT_* 764 → 544, kevT9 1264 → 934,
   jb_original 72 → 36, jb_easy 48 → 4. So the D1 thresholds were chosen on a deduplicated subset of
   dev (last row per key wins), and the D1 eval numbers differ slightly from these full-row ones: for
   example, kevT_test with 1.7B → 8B is 0.693 in D1 and 0.686 here, and with 4B alone 0.735 in D1 and
   0.715 here.
   - Suggested fix (not applied): key rows by position within the suite, or add the row index to the key.
   - After the fix, re-run D1.
3. **The quantile is the sensitive knob; the encoder matters less.** At q ≤ 0.95 every encoder and kNN
   score gives OOD accuracy of 0.678–0.692. At q0.99, the Qwen3-Embedding-0.6B scores let 74–85% of
   OOD traffic into the cascade, because dev has a long kev_dev tail. The base Qwen3-0.6B stays at
   24–27%, so its threshold is more stable.
4. td_test is closer to the training set than dev is (AUROC dev vs td_test ≈ 0.2). td_test shares
   templates with td_train, so real in-domain traffic may sit further out than td_test and be sent to
   8B more often (about 5–7% of kev_dev is, by construction).

## Recommendation
- **Score.** Use `knn10`: the mean cosine distance of the request's state embedding (base Qwen3-0.6B,
  same recipe as the cache) to its 10 nearest training states. It has no fitted parameters apart from
  the reference matrix (20,556 × 1024 fp16, about 42 MB) and a stable threshold across q. It also has
  high AUROC (0.976). `maha_pca+resid` on the same encoder is equivalent (AUROC 0.989) but needs a
  PCA fit.
- **Threshold rule.** Use the 95th percentile of the score on kev_dev + td_holdout
  (t = 0.0246 for Qwen3-0.6B knn10). Recompute it whenever the training set or the encoder changes.
- **Adopt?** Yes, for untagged requests only, with the 1.7B → 8B pair.
  - A tag for a known domain still forces the cascade, as in D1a.
  - The gain is the cascade saving on untagged in-distribution traffic, where the cost is 64% of 8B
    rather than 100%.
  - The price is about 1.5 pp on unknown-domain traffic compared with 8B alone.
  - For the 0.6B → 4B fallback, the gate saves little on OOD (5.70 vs 6.00) and costs 1 pp, so it only
    pays through the in-distribution saving.
  - If a 1.5 pp OOD loss is unacceptable, keep D1a. q0.90 is slightly safer (0.691 on OOD).

### Router change it would need (described, not applied)
1. `registry.yaml` `cascade:` gets a new block:
   ```
   ood_gate:
     encoder_url: http://127.0.0.1:8091/v1     # embed_server.py --model Qwen/Qwen3-0.6B (same recipe)
     reference: ood/reference_qwen3-0.6b.npz    # the 20,556 L2-normalised training state vectors
     k: 10
     threshold: 0.0246
     apply_to: untagged
   ```
   Also add a feature entry for the base Qwen3-0.6B embed server. The current registry lists only
   Qwen3-Embedding-0.6B. Using that model instead would need its own threshold (0.338 for knn10 q0.95);
   it is less stable at high q.
2. `decision_router.py` `route()` changes as follows:
   - If `domain` is None, build the state text once with `build_pairs`. It is shared by all questions
     in the request. Embed it through the gate's encoder, compute the mean of the top-10 cosine
     distances to the reference matrix, and call the cascade if the distance is ≤ threshold, else take
     the `direct` path.
   - Add `"route": "cascade:ood_gate"` or `"direct:ood_gate"` and the gate distance to each answer.
   - A request whose domain tag is set but not known stays on the direct path.
   - If the embedder is down, fall back to D1a behaviour (direct).
3. Export the reference matrix once from `runs/emb_cache` with the reference definition in
   `ood_gate.py`.

## Caveats
- **Eval suites were re-read.** This analysis re-read the probability dumps of td_test, kevT_test,
  kevT_dev, kevT9_dev and jb_*, including the kevT_test test split. Thresholds are dev only. However,
  the recommended encoder and score were chosen after seeing eval numbers, which is mild selection on
  eval, so treat the ~1.5 pp OOD gap as optimistic by a little.
  - Log this read in DECISIONS.md; it was not edited here.
- **jb_\* samples are small** (72 / 48 / 111 rows, with many duplicate groups). The pooled OOD figures
  are about 92% Kev transfer.
- **Gate cost is not in the tables.** One state embedding per request, about 1 unit (0.6B) if
  amortised over the request's questions.
  - Charged per question, td_test cost becomes 8.7 (72% of 8B) and OOD cost becomes 11.5.
  - The Kev and td requests carry several questions per state, so the per-request cost is lower.
- **Provisional costs.** Costs are the provisional 1 / 2 / 6 / 12 weights. The probability dumps are
  the stage-A pointer adapters. When B_* or retrained tiers arrive, re-run `ood_gate.py`; the reference
  set also changes if training data changes.
