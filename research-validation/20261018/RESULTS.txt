# Coincident, Not Leading — aggregate revalidation supplement

Release: 18 October 2026. [Public summary](https://yeohoonyun.github.io/midm-decision-models/research-validation/20261018/) · [한국어](https://yeohoonyun.github.io/midm-decision-models/research-validation/20261018/index.ko.html).

This is a separate financial tabular-model sensitivity study. These values are not
MiDM generic decision benchmark scores or RT1–RT7 scenario hit rates. No public
model weights were changed in this release. Computational review is complete;
manuscript integration and information-time/provider certification remain pending.

## Table 8: US57 median dominance margins

| Signal | Market return | Down magnitude | Up magnitude |
|---|---:|---:|---:|
| raw_original | 0.01738634 | 0.03029602 | -0.00850577 |
| factor_residual | 0.01307724 | 0.02792290 | -0.03853288 |

Each cell is the median of 57 model-specific margins: near-band maximum absolute
correlation minus outer-band maximum absolute correlation. Phase34 is excluded.
All six cells were independently reaggregated from the stored US342 rows on
18 October and match the earlier values within 1e-14. This is not a new correlation
or significance analysis; a positive margin alone is not statistical significance.
Download `table8_us57.csv` for full precision and `summary.json` for definitions.

## Phase34: completed 30-model computational audit

All 30 runs finished on 17 October at 13:21 KST. Training: 2012-01-03–2023-12-21;
validation: 2024-01-02–2024-12-23; held-out signals: 2025-01-02–2026-07-10;
last exit: 2026-07-17. Retained rows: 1,422,692 / 122,889 / 189,766.
Three stocks are selected per signal day; entry is next-session open and exit is
the close five sessions after the signal. Round-trip cost: 0.30%.

Across-model mean next-open net trade EV: 0.794%;
median: 0.881%. All 30 point estimates are
positive, but these correlated models do not constitute 30 independent successes.

The pre-existing validation-only rule selects `34-3-US_base_plus_kr_market_mlp__mlp`.
Held-out next-open net trade EV: **0.478%**;
95% date-block interval: **[-0.422%, 1.345%]**; test AUC:
**0.9319**. The interval includes zero; stable positive
expected returns or benchmark outperformance are not established. The interval
uses 10,000 circular bootstrap replicates, 5-session date blocks, trade weighting
and seed 20261018. No bootstrap-positive proportion is called a p-value.

For all 30 models, 5,692,980 saved held-out predictions in total were reloaded;
classification metrics, top3 picks and price-derived trade EVs were recomputed.
All stored metrics reconciled within the numerical tolerances recorded in the
private audit. Snapshot copies were checked with SHA256.

## Scope and remaining limitations

- EV is a mean per selected trade, not portfolio NAV or annualized return.
- Validation price excess is relative to the panel median, not the S&P 500 index.
- Historical features and candidate eligibility remain frozen; full point-in-time
  validity and information leakage remediation are not certified.
- Single-vintage provider prices and corporate-action treatment remain uncertified.
- Original-label counterparts were not refitted on identical purged rows; no pure
  causal effect of relabelling is claimed.
- The 30-model arm does not replace the original 114-model census or US57 cohort.
- The latest manuscript file has not been replaced. This is not a submission-ready
  or peer-review acceptance claim.
- This release contains aggregates and audit outcomes. Underlying individual
  predictions, prices, features, private weights and scripts are not published here.

## Files

- `table8_us57.csv`: the six missing Table 8 values.
- `phase34_model_results.csv`: all 30 models, ordered by validation score.
- `summary.json`: full-precision aggregate figures, intervals and limitations.
- `audit_checks.json`: the scope and outcome of the computational rechecks.
- `RESULTS.txt`: portable method/results note.
