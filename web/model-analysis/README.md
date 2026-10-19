# ScenarioView daily model dashboard

## Daily model dashboard

[Open dashboard](https://yeohoonyun.github.io/midm-decision-models/model-analysis/index.html). This standalone browser UI reads reviewed, static model-level aggregates. Period/sector returns and simulated recommendation entry/exit counts are separate from personal account orders. Local exporters/schedulers, financial inputs, raw market data, broker credentials, quantities, prices and prediction scores are not published. User-authorized symbol/action details are in a collapsible lower panel. Daily data is supplied by the existing PC; no broker request is made by this page.

Files are mirrored to docs/scenarioview/model-analysis and GitHub Pages model-analysis/. Source JSON contract: schemaVersion 1. Serve this folder with data/latest.json from the reviewed daily export.
