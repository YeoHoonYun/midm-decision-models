# ScenarioView daily model dashboard

## Daily model dashboard

[Open dashboard](https://yeohoonyun.github.io/midm-decision-models/model-analysis/index.html). This standalone browser UI reads reviewed, static model-level aggregates. Period/sector returns and simulated recommendation entry/exit counts are separate from personal account orders. Local exporters/schedulers, financial inputs, raw market data, broker credentials, quantities, prices and prediction scores are not published. User-authorized symbol/action details are in a collapsible lower panel. Daily data is supplied by the existing PC; no broker request is made by this page.

Files are mirrored to docs/scenarioview/model-analysis and GitHub Pages model-analysis/. Source JSON contract: schemaVersion 1. Serve this folder with data/latest.json from the reviewed daily export.

## Privacy and Google tags

telemetry.js uses the existing public GA4 and AdSense identifiers (public configuration, not credentials). It runs only on the exact public dashboard path in a top-level HTTPS page, skips DNT/GPC, and skips URLs with queries or unknown fragments. It never loads Google tags in the local trading app. Manual page-view metadata is static; no account, search-filter, symbol or snapshot payload is attached. GA stream enhanced measurement can also measure public-page interactions. Advertising storage, user data and personalization default to denied; Google signals and GA ad personalization are disabled. AdSense may process public page content; see the Google data-use link in the page footer. Script installation is separate from site approval, ad serving and reporting.

The daily publisher projects approved public fields, rejects private account/price/quantity/score fields and known credential patterns, and compares local credentials in memory before committing. Only curated UI and reviewed public snapshots enter this repository. No private input or environment file is uploaded to a third-party scanner.
