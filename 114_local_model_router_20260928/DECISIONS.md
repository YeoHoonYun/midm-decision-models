# Decision log: how router / model choices are made (kept up to date while building)

Principle: no choice reads a test suite. Choices use dev data: Kev decision-v7 dev, the 10%
typed-decisions train holdout (`td_holdout`), and the stage-A `src_holdout`, which is Kev sources
left out of training. Each test suite is then evaluated once with the frozen choice, and every read
is logged.

## D1. Cascade thresholds (decider-auto)
- Input: per-question probabilities from each tier on dev (`--save-probs` dumps, exp 113 results/probs).
- Rule: take the lowest mean cost (escalation-weighted) whose dev accuracy is within 0.5 pp of the
  best single tier's dev accuracy. If no setting qualifies, take the highest dev accuracy.
- Cost per tier is relative latency, measured on the serving GPU. Until measured, use provisional
  weights 0.6B = 1, 1.7B = 2, 4B = 6, 8B = 12, LLM judge = 50.
- Tool: `calibrate_cascade.py`. Re-run whenever a tier's model changes.

### D1 result (2026-09-28 17:58, `results/cascade_*.json`)
Dev = kev_dev + td_holdout (2,068 questions). Costs are provisional. Eval suites were each re-read
once, for the cascade combination only.

| cascade | thr | dev acc (best single) | escalated | cost vs top single | td_test (top) | Kev transfer test (top) |
|---|---|---|---|---|---|---|
| 0.6B → 4B | 0.62 | 0.840 (0.844) | 37% | 53% | 0.784 (0.788) | 0.704 (0.735) |
| 0.6B → 8B | 0.60 | 0.838 (0.842) | 34% | 42% | 0.791 (0.794) | 0.722 (0.774) |
| 1.7B → 8B | 0.58 | 0.838 (0.842) | 24% | 41% | 0.795 (0.794) | 0.693 (0.774) |
| 0.6B → 4B → 8B | 0.62/0.30 | 0.840 | 37% | — | 0.784 | 0.702 |

Finding: thresholds set on in-distribution dev hold in-distribution, where the cascade keeps
accuracy at 41–53% of the cost. They fail on never-trained sources (−3 to −8 pp), because small
tiers are overconfident there and escalate too little.

### D1 CORRECTION (2026-09-28 20:40): the table above is superseded
`calibrate_cascade.py` keyed rows by (suite, qid, group), which is not unique, so rows overwrote each
other. Dev used 1,720 of 2,068 rows, Kev transfer 544 of 764, jb_easy 4 of 48. Fixed: the key is now
(suite, position) and tiers are checked for alignment. Found by the report-draft agent and
independently by the OOD agent. Re-run on all rows:

| cascade | thr | dev acc (best single) | dev cost | td_test (top) | td_test cost | Kev transfer test (top) |
|---|---|---|---|---|---|---|
| 0.6B → 4B | 0.62 | 0.842 (0.846) | 53% | 0.785 (0.788) | 80% | 0.690 (0.715) |
| 0.6B → 8B | 0.62 | 0.845 (0.849) | 45% | 0.793 (0.795) | 71% | 0.728 (0.764) |
| 1.7B → 8B | 0.70 | 0.844 (0.849) | 54% | 0.794 (0.795) | 88% | 0.736 (0.764) |
| 0.6B → 4B → 8B | 0.64 / 0.30 | 0.846 (0.849) | 28% | 0.783 (0.795) | 42% | 0.691 (0.764) |

- The direction of the finding holds. The out-of-distribution loss is 2.5–3.7 pp (not 3–8). Test
  cost savings are smaller than on dev (71–88% for the pairs), because escalation is higher on test.
- The D1 rule applied across all candidate cascades picks **0.6B → 4B → 8B (0.64 / 0.30)**, with the
  lowest dev cost within 0.5 pp. It is adopted per the rule. Flag: on td_test it is −1.2 pp against
  8B. It is **not** re-chosen from test. Re-calibrate when the stage-B (2-epoch) models land.
- Test reads logged: this re-run and the OOD agent's analysis (`ood/`) re-read the eval-suite
  probability dumps (kevT_*, jb_*, td_test). No choice above used them.

### D1 with stage-B models (2026-09-29 00:55, `results/cascadeB_*.json`)
Tiers: 0.6B (1 ep), 4B and 8B (2 ep, stage B). Same rule, same dev (2,068 rows).

| cascade | thr | dev acc (best 0.857) | dev cost | td_test (8B/4B alone) | td_test cost | Kev transfer test (top alone) |
|---|---|---|---|---|---|---|
| 0.6B → 4B | 0.62 | 0.844 | 53% | 0.793 (0.798) | 80% | 0.695 (0.725) |
| 0.6B → 8B | 0.76 | 0.853 | 59% | 0.803 (0.805) | 87% | 0.770 (0.772) |
| 1.7B → 8B | 0.74 | 0.852 | 58% | 0.805 (0.805) | 93% | 0.755 (0.772) |
| **0.6B → 4B → 8B** | **0.80 / 0.50** | 0.852 | **42%** | **0.8035 (0.805)** | 75% | 0.741 (0.772) |

- The rule picks the 3-tier cascade (0.80 / 0.50), and the registry is updated to it. It sits within
  0.15 pp of 8B on td_test at 75% of the cost; the earlier 1-epoch 3-tier was −1.2 pp.
- Out of distribution it still loses 3.1 pp, so D1a stays (cascade only for known domains).
  Side note, not used for any choice: the 0.6B → 8B pair at 0.76 loses only 0.3 pp on Kev
  transfer. Its higher threshold escalates more.

### D1 with Qwen3.5-4B (2026-09-30, `results/cascadeC_*.json`)
Qwen3.5-4B (1 epoch, bf16) matches 8B on dev (0.8564 vs 0.8569) at half the cost. The rule applied
across all cascades picks **0.6B → Qwen3.5-4B → 8B (0.66 / 0.30)**: dev 0.853 at 29% of 8B cost.
td_test 0.802 vs 8B 0.805 at 43% cost; Kev transfer test 0.736. Qwen3.5-4B alone scores 0.806 /
0.764 at 50% cost. Adopted in the registry. For unknown domains (D1a), the direct route now starts
with Qwen3.5-4B, which equals 8B off-distribution at half the cost.

### D7. Unknown-domain model (2026-10-01)
- Rule: route unknown domains to the model that is best on the Kev transfer **development** partition
  (kevT_dev). This is Kev's own dev split and is never used for training. kevT_test only confirms.
- Candidates (kevT_dev): MiDM-4B-q35-e1-bx 0.791, MiDM-4B-q35-e1 0.755, MiDM-8B-q3-e2 0.729. Choice: **-bx**.
- Confirmation on kevT_test: 0.798 vs 0.764 / 0.772. Paired bootstrap vs q35-e1: +3.4 pp, p = 0.042.
- Known domains keep the D1 cascade, because -bx is 1.7 pp lower on typed-decisions test and its
  in-domain dev is not better.

### D6. Out-of-distribution gate (proposed, `ood/README.md`)
- Score: kNN-10 mean cosine distance of the base Qwen3-0.6B state embedding to 20,556 training states.
- Threshold: the 95th percentile of the dev score, t = 0.0246. With the old 1.7B → 8B at 0.58 it
  sent all of td_test to the cascade (8B accuracy at 64% of the cost) and 81% of unknown-source
  traffic to 8B. On unknown domains it scored about 1.5 pp below 8B alone.
- Caveats: the agent chose the encoder and score after seeing the eval numbers, so the result is
  slightly optimistic. The gate's own embedding cost is excluded. It must be re-run with the
  corrected thresholds and the B models.
- Status: not yet in the router. Untagged requests still go direct (D1a).

### D1a (adopted): cascade only for known domains
- Requests tagged with a registered, trained domain (typed-decisions workflows, Kev decision-v7
  sources) use the cascade with the D1 threshold. Default pair: 1.7B → 8B, or 0.6B → 4B when the
  8B is not resident.
- Untagged or unknown-domain requests go straight to the strongest resident tier (8B; else 4B).
- Open follow-ups:
  - an automatic out-of-distribution gate (e.g. distance of the state embedding to the training
    set) so untagged traffic can use the cascade;
  - recalibrating the small tiers with temperature scaling on a source-held-out dev;
  - measured latency costs to replace the provisional 1 / 2 / 6 / 12 weights.

## D2. Which models are tiers
- A model can enter the cascade only if it has dev probabilities and a one-time eval row in exp 113.
- Order by measured latency. Drop a tier if it is not more accurate on dev than the tier below it
  on the questions that tier would escalate.

## D3. LLM judge as the top tier
- Before use, measure its dev accuracy on questions the pointer tiers escalate, using a sample (e.g.
  300), because every call loads a model into experiment 34's shared Ollama on the 3090.
- Keep it only if it beats the best pointer tier on that escalated subset. Otherwise the cascade
  ends at the best pointer tier.
- Candidates: qwen3-8b, gpt-oss-20b. Run only when the 3090 is not busy (check the GPU queue and Ollama).

## D4. Serving GPU (open)
- Resident models hold VRAM and conflict with the training queue.
- Proposal: keep only small tiers resident (0.6B pointer, about 2 GB 4-bit) plus the embedder on
  whichever GPU has headroom. Start larger tiers on demand when the queue is idle.
- To be settled once stage B/C GPU use is known.

## D5. Stage-A adapter search (exp 113): pick by dev
- Score = mean of td_holdout, kev_dev500 and src_holdout at the final step. src_holdout breaks ties
  in favour of generalisation. Carry the top 1-2 settings to stage B.

## Log
- 2026-09-28 14:35: D1-D5 written. A1 (LoRA r16, Qwen3-1.7B): td_holdout 0.825, src_holdout 0.543,
  kev_dev500 0.848.
- 16:48 A3 (rsLoRA r64, alpha 128, fp16) diverged, loss NaN at step ~1070, so it was stopped.
  Consequence: with rsLoRA, lower alpha or use bf16 (3090). Do not carry it to stage B as is.
- 16:52 A5 (LoRA r16, lr 1e-4): td_holdout 0.812, src_holdout 0.554, kev_dev500 0.850 (mean 0.739
  vs A1 0.739). This is a tie. Under D5, src_holdout breaks it in A5's favour, but the gap is 1.1 pp
  on 276 items, which is noise level.
- 15:45: jobs 4 (A2), 19 (8B eval) and the queue workers died together. The cause was an external
  process kill, not a driver reset; the nvlddmkm 153 events are left by force-killed CUDA processes
  (see _ops/GPU_DRIVER_RESET_PROCEDURE.md). Both jobs were requeued. The 8B eval now runs next to
  other jobs on the 3090 (max-util 100) to save time.
