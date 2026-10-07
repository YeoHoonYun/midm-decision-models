# MiDM financial architecture and application

[Live ScenarioView report](https://yeohoonyun.github.io/midm-decision-models/)

## Active architecture versus experiments

The active model is unchanged: Qwen3.5-9B-Base + trained LoRA adapter + joint-option pointer head. B3 produces seven scenario scores, supplied alongside as-of market/position features in the MiDM prompt. MiDM chooses between LONG/CASH (BUY/WAIT or HOLD/SELL according to position). Price-based risk/expiry rules are implemented separately. The report writer and rendering templates are separate from this choice model.

```text
Local prices -> B3 7-case scores -> structured state and options
                                       |
                              frozen MiDM 9B + pointer
                                       |
                            LONG/CASH choice -> risk rules
                                       |
                               local report -> public aggregates

Experimental branch (NOT active):
numeric context / frozen latent features -> small residual head
                  + clipped MiDM logit -> sigmoid -> choice
```

## Implemented residual architecture

The accompanying residual_heads.py contains the actual architecture used in the experiment, separated from private paths and datasets. Dependencies: NumPy and SciPy. Do not load private training pickles from untrusted sources.

- Numeric input: 24 fields; six engineered fields add case entropy, top-two gap, two volatility-normalized returns, trend alignment and clipped MiDM margin.
- Numeric residual: 31 coefficients including intercept.
- Horizon/position residual: 91 coefficients, with interactions for 20-day versus 5-day horizon and held versus flat state.
- Latent residual: 39 coefficients; adds eight PCA components from 128-dimensional frozen projected hidden features. PCA and scaling are fitted only in training folds.
- Random-feature comparator: 63 coefficients; 32 fixed cosine features, without a MiDM logit offset. Its inputs still contain the MiDM margin, so this is NOT an entirely MiDM-independent model.
- Residual probability: sigmoid(clip(MiDM margin, -6, 6) + w^T x). Raw baseline retains its original logits. L2 penalties 0.1, 1, 10; all coefficients including intercept are penalized.

Numeric columns: ret1, ret5, ret20, ret60, annualized rv20, drawdown252, ma20gap, ma50gap, ma200gap, rv20change5, age_sessions, remaining_sessions, entry_pnl, support_distance, two_closes_below_ratcheted_support, held, horizon/20, RT1 through RT7. Mapping and units must match before calling the head. `d` contains numeric[N,24], latent[N,128], margin[N], and (training only) y[N], with LONG=1. No raw training data or trained financial head weights are included.

## Application evidence

KOSPI action choice, 5/20-day horizons, 756 hypothetical states across 129 dates (2024-01-02–2026-08-20). Expanding development years 2020–2023; 60-calendar-day purge plus label-expiry filter. Final head fit includes purged 2023; differences from earlier experiments are not architecture-only effects. This retrospective period had already been inspected.

| Model | Accuracy | Balanced accuracy |
|---|---:|---:|
| Existing MiDM | 54.37% | 49.24% |
| Numeric residual | 48.81% | 48.29% |
| Horizon/position residual | 47.49% | 47.75% |
| Latent residual | 50.53% | 49.98% |
| Random-feature comparator | 48.41% | 51.72% |

No financial candidate was promoted. This is implementation and negative-result documentation, not an improved-weight release. The current multi-asset application now executes MiDM locally for each asset and both horizons; stale source dates remain explicitly labeled. Prepared uncertainty/context prompts have not yet been inference-tested. The static report shows public macro references, case definitions, aggregate evidence and provenance; it does not disclose private per-date scores.

## Current application

The [report collection](https://yeohoonyun.github.io/midm-decision-models/) uses new local MiDM inference for 12 assets and 48 hypothetical action states (5/20-day, flat/held). Asset-matched B3 is used only for KOSPI; the other assets use causal price features and historical-analogue counts, explicitly not neural B3 probabilities. Financial transfer quality is not validated separately for each asset. Generation scripts and private financial weights/data are not published.
