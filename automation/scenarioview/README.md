# ScenarioView after-close refresh

[Report](https://yeohoonyun.github.io/midm-decision-models/) · [Architecture and application](../../docs/architecture/financial-residual/README.md)

Normal KOSPI regular session closes at 15:30 Korea time. The Windows task starts checking at **15:31 KST**, repeats every minute until 21:00, and skips weekends. The public monthly FRED downloads are throttled to once per 15 minutes; no faster provider-finalization SLA has been verified. Source observation dates remain visible. Special sessions/holidays may be configured in session_overrides (`YYYY-MM-DD`: `HH:MM`, or null). Missing holiday overrides cannot produce a fresh trading signal: this updater performs NO model inference. These hours refer to regular-session KOSPI, not extended-hours company-stock trading.

Official hours: https://global.krx.co.kr/contents/GLB/06/0602/0602020204/GLB0602020204T1.jsp

## What runs

1. Enforce Korea-time post-close window, single-process lock and public-source throttle.
2. Retrieve only three allowlisted public OECD/FRED series IDs, validate observation dates and numeric values.
3. Inspect the configured local KOSPI source date locally. Price values are not uploaded. This is a freshness check, not a price downloader or an inference pipeline.
4. Refresh public macro cards and issue date only when public content changes. Keep historical model scores and their evaluation dates fixed. No-op if unchanged; retain prior report on download/validation failure.
5. Push only index.html and macro_latest.json to gh-pages; GitHub Pages then builds. Public model documentation/HF weights are not rewritten every day.

**Daily B3/MiDM inference is not connected.** The current inherited local KOSPI source is historical. A live licensed price feed and a completed as-of-matched inference job must be integrated before presenting daily action recommendations. Do not label this scheduler as an end-to-end daily financial model service.

## Setup and execution

Use a dedicated gh-pages checkout with authenticated Git push. Python requires pandas for optional local-date inspection (and a parquet engine). Copy scenarioview_daily.example.json to scenarioview_daily.local.json next to the scripts and fill local paths; never commit credentials/local config. Point template_path at a reviewed public-only HTML template, not a private full report. All credentials stay in the existing local credential manager.

```powershell
.\run_scenarioview_daily.ps1 -DryRun -ForceCheck
.\install_scenarioview_daily.ps1
```

Or double-click run_scenarioview_daily.bat. `-ForceCheck` bypasses the clock window for explicit manual checks, not data validation. Installer requires Windows Korea Standard Time. It registers ScenarioView-KOSPI-AfterClose for the current interactive user, limited privileges, hidden PowerShell, nonoverlapping execution, wake-to-run and start-when-available. PC must be powered on and the user logged in; wake is not power-on. GitHub build time is additional and variable. A stale lock after an abrupt termination requires checking no runner is alive before removing it.

Inspect `Get-ScheduledTaskInfo -TaskName ScenarioView-KOSPI-AfterClose` and the local last_check JSON/logs. Disable using `Disable-ScheduledTask -TaskName ScenarioView-KOSPI-AfterClose`. Source/validation failures leave the last report intact. Git transport failures may require reconciling a pending local commit before retry.
