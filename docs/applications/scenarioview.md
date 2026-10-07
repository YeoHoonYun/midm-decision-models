# Application case: MiDM internal financial scenario reports

[Open the 12-asset report collection](https://yeohoonyun.github.io/midm-decision-models/) · [MiDM model](https://huggingface.co/yunicro/MiDM-9B-q35-e1-bx) · [Architecture and experimental limitations](https://github.com/YeoHoonYun/midm-decision-models/tree/main/docs/architecture/financial-residual)

## Report languages / 보고서 언어

The same 12 assets are available in Korean and English. Use the language links at the top of each report to switch without changing the asset. Both editions retain identical dates, observed numbers, scenario counts and MiDM selections. Localization uses local editorial templates, without an external translation API or new model inference. The linked research archive remains in Korean.

동일한 12개 자산 보고서를 한글·영문으로 제공합니다. 보고서 상단에서 언어를 전환할 수 있으며, 관측 수치·자료 기준일·시나리오 집계·MiDM 선택 결과는 같습니다. 일일 생성에도 두 언어를 적용합니다.

| Example | 한국어 | English |
|---|---|---|
| All 12 assets / 전체 | [한글](https://yeohoonyun.github.io/midm-decision-models/index.ko.html) | [English](https://yeohoonyun.github.io/midm-decision-models/index.en.html) |
| KOSPI | [한글](https://yeohoonyun.github.io/midm-decision-models/KOSPI.ko.html) | [English](https://yeohoonyun.github.io/midm-decision-models/KOSPI.en.html) |
| S&P 500 | [한글](https://yeohoonyun.github.io/midm-decision-models/SP500.ko.html) | [English](https://yeohoonyun.github.io/midm-decision-models/SP500.en.html) |
| Samsung Electronics / 삼성전자 | [한글](https://yeohoonyun.github.io/midm-decision-models/005930.ko.html) | [English](https://yeohoonyun.github.io/midm-decision-models/005930.en.html) |
| NVIDIA | [한글](https://yeohoonyun.github.io/midm-decision-models/NVDA.ko.html) | [English](https://yeohoonyun.github.io/midm-decision-models/NVDA.en.html) |

GitHub report files: `docs/scenarioview/*.ko.html` and `*.en.html`. Hugging Face report files: `reports/latest/*.ko.html` and `*.en.html`. Neutral URLs now select the edition by IP country (KR → Korean; elsewhere → English). Explicit `.ko.html` / `.en.html` links keep their specified language. Korean-market reports retain teal accents; US-market reports retain navy accents in both languages.

## Country routing and historical validation (2026-10-07)

[Automatic regional entry](https://yeohoonyun.github.io/midm-decision-models/) · [Historical validation including B3](https://yeohoonyun.github.io/midm-decision-models/validation.html)

Neutral report URLs request the visitor country from [Country.is](https://country.is/). This browser-side request exposes the visitor's public IP to the country service, but sends no financial inputs, report body, cookies or referrer. It is distinct from report generation, which still makes zero external data/model calls. A manual language choice takes priority; otherwise KR selects Korean and other countries select English. Failure or a 1.8-second timeout falls back to the browser's primary language. Only the country code is cached for one hour in session storage; IP addresses are not stored by this application. VPNs can change the country estimate. The “Auto by country” control clears the manual choice and cache. Explicit language URLs do not automatically redirect or call the country API.

과거 검증 화면에는 S&P 500 B3-1 전체 Top-2 및 RT별 재현율, B3-2 조건부 결과, B3-3 탐지 및 미채택 조합, 2026-07-27~31 재구성 보고서의 일자별 판정과 매매 시뮬레이션, 코스피 별도 학습 B3 계열의 매매 검증을 담았습니다. 현재 보고서의 미래 성과나 개별 종목 검증으로 해석하지 않습니다.

Historical evidence is generated from nine local aggregate result files, with source identifiers and SHA-256 hashes in `validation_summary.json`. B3-1 improves on B2 but does not dominate every comparator; conditional specialist scores are not all-date accuracy. The five historical reports used local Qwen3-8B prose, not current MiDM 9B. Their narrative forecasts and trading results are shown alongside failures. The KOSPI model is a separately trained 10-feature B3-style network, not unchanged US B3 weights. Neither retrospective results nor language routing constitute a model-weight upgrade. Market-specific reference panels appear in each report, and individual-stock pages label index results as reference only.

Both language editions and validation links are integrated into the daily local renderer. No training, report-generation or publication scripts are uploaded; the small browser language-routing code is part of the published report UI.

## Implemented application

ScenarioView combines locally stored financial inputs with MiDM 9B option scoring to produce reports for KOSPI, S&P 500, and five company representatives per market selected from internal market-cap records. The recorded run completed **48 real local decisions**: 12 assets × 5/20-day horizons × flat/held hypothetical positions.

The active model is Qwen3.5-9B-Base with the existing MiDM LoRA adapter and joint-option pointer head. The experimental financial residual heads were not promoted. This application demonstrates an internal reporting workflow, not improved model weights or demonstrated trading alpha.

| Component | Actual role |
|---|---|
| Internal stored data | As-of prices and market-cap records; previously collected provider data |
| B3 | KOSPI scenario scores only; not asserted to be validated for US assets or individual stocks |
| Historical analogues | Matured similar-path counts, not calibrated outcome probabilities |
| MiDM 9B | Actual local LONG/CASH option scoring for flat and held hypothetical states |
| Report text, tables and charts | Deterministic local rendering using observations and model choices; not free-form LLM narrative |
| Publication | Separate upload of reviewed report artifacts and model documentation |

## Fixed after-close editions — no repeated data checks

| Edition | Korea time | Days | Execution |
|---|---|---|---|
| After the Korean regular session | **18:30 KST** | Monday–Friday | One integrated 12-report generation job |
| After the US regular session | **09:00 KST** | Tuesday–Saturday | One integrated 12-report generation job |

These deliberately buffered times replace the earlier minute-by-minute data watcher. At the scheduled time, one GPU-queued job prepares inputs, runs local inference, renders reports, and then publishes the reviewed artifacts. There is no data-arrival polling, change-triggered regeneration, or automatic retry. An overlapping unfinished edition blocks a second submission. Completion depends on GPU availability and the publication build; these times are generation triggers, not guaranteed completion times or provider data-finalization guarantees.

Each run uses the stored data available at that moment and displays the actual observation date even if it is old. Exchange holidays and exceptional closes are not automatically calendar-integrated; stale input must not be presented as a current market signal. The Windows host must be on and the configured user logged in.

## Internal processing and public scope

**Generation inputs and model execution: 100% internal. External data API calls: 0. External model API calls: 0 in the recorded generation run.** This does not mean all source data were originally produced in-house: stored provider data are included. Publication to GitHub/Hugging Face happens separately after generation.

Only model documentation and reviewed reports are updated publicly. Generation, scheduling and publication scripts remain local. Raw prices, prompts, individual option probabilities and private financial head files are excluded. Previously published generator files were removed from the current branch; historical Git commits were not rewritten.

## Evidence and limits

- The initial reports use KOSPI observations through 2026-09-04, Korean stock data through 2026-09-21, and S&P/US stock data through 2026-10-06. Later reports show their own source dates.
- “Top five” means internal coverage: Korean common-share representatives from a cap cache with an unverified observation date; US close × last-known share-count proxies, with GOOG/GOOGL deduplicated. It is not a certified current whole-market company ranking.
- Choices are hypothetical, long-only and not broker orders. Asset-specific transfer performance and portfolio profitability are unvalidated. B3 case probabilities must not be equated with investment success probabilities.
- The four proposed financial residual architectures did not justify replacing the original MiDM model. Weights remain unchanged.
- Schedule boundary checks, orchestration/failure-stop tests and the preceding 48-decision local run support implementation status; future scheduled runs are not claimed as already completed.
