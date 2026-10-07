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

GitHub report files: `docs/scenarioview/*.ko.html` and `*.en.html`. Hugging Face report files: `reports/latest/*.ko.html` and `*.en.html`. Existing unqualified URLs remain Korean-compatible. Korean-market reports retain teal accents; US-market reports retain navy accents in both languages.

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
