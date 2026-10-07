# v0.2.2 — ScenarioView report and financial validation

Documentation/report release dated 2026-10-07. Model weights, tokenizer and loader are unchanged from v0.2.0; this is NOT an improved-accuracy model release.

- Publishes a mobile-friendly ScenarioView report using the previous local report layout styles and section order.
- Includes public macro observations with observation dates and source attribution, RT definitions, aggregate action metrics, and section-level provenance.
- Frozen MiDM action accuracy: 54.37%; B3+MiDM stacking: 50.93%; development-selected horizon/position residual: 47.49%. No candidate promoted.
- Evaluation: KOSPI, 2024-01-02 to 2026-08-20, 756 hypothetical decisions / 129 sampled dates, previously seen retrospective data; not independent trades or portfolio returns.
- No validated economic leading/coincident/lagging status. New prompt context packs are prepared locally but have not been inference-tested.
- No private source prices, per-date predictions, generated private narratives, checkpoint files or credentials published.
- The report is issued today but does not contain a freshly inferred today-market signal. No automatic daily schedule is configured.

Report: https://yeohoonyun.github.io/midm-decision-models/

The prior Zenodo DOI continues to refer to v0.2.1; no new Zenodo deposit is claimed.
