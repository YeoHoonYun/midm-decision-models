# 누락된 상세 결과 보강 — 2026-10-22

기존 실측 파일에서 추출한 보충 자료이며 새로운 논문 원고나 신규 실험이 아니다. 값은 반올림된 표와 정밀 RESULTS.json을 함께 제공한다.

## 해석 및 명칭 주의

- A01 paired80은 80일 블록을 뜻한다. 코드의 2.5/97.5 분위수로 산출한 95% 구간이며, 다중비교 보정 전 기술적 구간이다.
- A05 p는 3,000회 studentized paired stationary bootstrap, 평균 블록80, Bartlett HAC lag80 결과이다. 공식 Ledoit–Wolf 구현 재현으로 명명하지 않는다. 비교 기준은 SP500 scheduled_long, KOSPI always_long_guard다. DSR은 N50/N412 민감도이며 전체 시행 수익행렬이 없어 정식 전체 탐색 보정·PBO는 미완이다. 일부 희소 정책의 bootstrap 퇴화 횟수와 추론 상태는 RESULTS.json을 확인한다.
- A03 2008/09는 학습 기간 밖이지만 이후에 학습한 모델의 역방향 진단이다. 2020/2022는 학습 중첩 stress다. 41특징 전체 모델이 아니라 고정10특징 모델 결과이며, 낮은 노출로 손실을 줄인 것을 노출 중립 타이밍 우위로 해석하지 않는다.
- A10은 2023 선택·보정 재사용의 후향 진단이다. 시계열 교환가능성과 독립 보정을 충족했다고 주장하지 않는다. 평균 집합 크기를 함께 제시한다.
- A04 마스킹은 등록 범위에서 완료했지만 사전학습 오염의 부재 또는 look-ahead 통제 충족을 입증하지 않는다. DSS 판정에서 look-ahead는 부분 점검/잔여 위험으로 남겨야 한다.
- REVIEW_KO.md는 추가 실험 이전 원고에 대한 당시 검토 기록이다. 그 안의 미실행 표현은 2026-10-21 후속 완료 결과로 대체해 해석한다. 익명성·분모·과장 표현 지적은 별도 수정 사항이며 여기서는 원고를 수정하지 않는다.

## A01 상세 표
# A01 actual matched-sample baselines

Coverage is percent; deltas and intervals are percentage points. All intervals are descriptive paired80date intervals.

| Model | Top2 % | Delta vs B3 pp | paired80 CI pp |
|---|---:|---:|---|
| h5_GBM | 47.8582 | -2.9542 | [-6.204, +0.148] |
| h5_multinomial_logistic | 50.0739 | -0.7386 | [-5.761, +4.284] |
| h20_GBM | 57.8053 | +1.8547 | [-8.037, +12.519] |
| h20_multinomial_logistic | 54.8686 | -1.0819 | [-5.873, +3.246] |

## A02 상세 표
# A02 actual rule baselines

## Actions on756states

| Rule | Accuracy | Balanced | Regret bp |
|---|---:|---:|---:|
| SMA200 | 53.5714% | 50.1348% | 207.990 |
| MOM12_1 | 49.2063% | 44.9622% | 233.113 |
| VOL15 | 56.8783% | 50.0000% | 200.938 |
| always_long_guard | 56.8783% | 50.0000% | 200.938 |

## Same-guard portfolio test

| h | Rule | Return% | MDD% | Exposure% |
|---|---|---:|---:|---:|
| 5 | SMA200 | 57.685 | -20.875 | 48.94 |
| 5 | MOM12_1 | 61.006 | -16.438 | 51.36 |
| 5 | VOL15 | 42.698 | -19.427 | 50.35 |
| 5 | always_long_guard | 73.937 | -20.118 | 65.18 |
| 5 | MiDM_archived | 68.642 | -20.683 | 64.76 |
| 20 | SMA200 | 80.526 | -16.810 | 43.48 |
| 20 | MOM12_1 | 55.834 | -25.800 | 45.08 |
| 20 | VOL15 | 58.464 | -21.426 | 47.63 |
| 20 | always_long_guard | 77.798 | -24.715 | 59.79 |
| 20 | MiDM_archived | 75.177 | -23.710 | 59.12 |

## A03 상세 표
# A03 backward and stress regime results

Frozen E8 ten-feature models trained2014-2022 and selected2023. Newly acquired public2000-2012prices reconciled on the2012 overlap.2008/09 is outside training but models were trained later: not contemporaneously investable forecasts.2020/2022 overlap training and are NOT holdouts.

| Market | Period | h | N | Top2% | Frequency% | Policy return% | MDD% | Exposure% | Scheduled% | Buy hold% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SP500 | 2008/09 backward | 5 | 377 | 29.97 | 36.34 | -24.80 | -37.59 | 72.40 | -34.34 | -39.28 |
| SP500 | 2008/09 backward | 20 | 377 | 56.50 | 56.50 | -0.73 | -1.86 | 1.01 | -33.76 | -32.68 |
| KOSPI | 2008/09 backward | 5 | 372 | 45.16 | 45.16 | -21.72 | -39.79 | 63.05 | -23.61 | -22.67 |
| KOSPI | 2008/09 backward | 20 | 372 | 60.22 | 55.65 | -3.22 | -17.57 | 27.36 | -25.09 | -17.72 |
| SP500 | 2020 IN TRAIN | 5 | 104 | 40.38 | 41.35 | -14.42 | -26.09 | 59.03 | -5.72 | -3.97 |
| SP500 | 2022 IN TRAIN | 5 | 251 | 41.83 | 41.83 | -18.04 | -24.52 | 75.65 | -18.78 | -18.89 |
| SP500 | 2020 IN TRAIN | 20 | 104 | 48.08 | 48.08 | 7.99 | -3.76 | 16.98 | -4.02 | -1.29 |
| SP500 | 2022 IN TRAIN | 20 | 251 | 56.18 | 56.18 | -2.35 | -3.44 | 4.51 | -15.62 | -15.04 |
| KOSPI | 2020 IN TRAIN | 5 | 103 | 47.57 | 47.57 | 4.41 | -19.32 | 44.66 | -2.84 | 0.19 |
| KOSPI | 2022 IN TRAIN | 5 | 246 | 55.69 | 55.69 | 4.38 | -12.48 | 38.27 | -23.19 | -23.47 |
| KOSPI | 2020 IN TRAIN | 20 | 103 | 60.19 | 54.37 | 14.80 | -6.36 | 37.19 | -2.57 | 4.49 |
| KOSPI | 2022 IN TRAIN | 20 | 246 | 71.14 | 41.46 | -0.65 | -0.85 | 0.38 | -16.15 | -18.95 |

No pooled unseen-regime score mixes2008 with2020/2022. Lower loss with lower exposure does not establish exposure-neutral timing skill. No new action-task backward accuracy is claimed. Cash return0 in this table; cash-rate sensitivity is separateA09.

## Requested pooled descriptive summary

This mixes training-overlap2020/2022 with backward2008. It is NOT a pooled unseen holdout. A continuous nominal-sleeve account holds cash through gaps.

| Market | h | N | Top2% | Frequency% | Pooled policy% | Scheduled% |
|---|---:|---:|---:|---:|---:|---:|
| SP500 | 5 | 732 | 35.52 | 38.93 | -46.33 | -48.84 |
| SP500 | 20 | 732 | 55.19 | 55.19 | 5.78 | -42.30 |
| KOSPI | 5 | 721 | 49.10 | 49.10 | -14.44 | -43.03 |
| KOSPI | 20 | 721 | 63.94 | 50.62 | 8.67 | -33.70 |

## A05 상세 표
# A05 actual return statistics

DSR values are conditional sensitivity using available policy-family Sharpe spread, not full412trial certification. PBO unavailable without candidate time matrix.

| Cohort | Policy | Sharpe | DSR N50 | DSR N412 | Paired p |
|---|---|---:|---:|---:|---:|
| SP500_h5 | B3-1 | 0.609 | 0.6113809878956745 | 0.5194429977356028 | 0.28523825391536156 |
| SP500_h5 | B3-1_fixed_exit | 0.771 | 0.711947195971703 | 0.6268682090343782 | 0.6401199600133289 |
| SP500_h5 | B3-2 | 0.801 | 0.717485191570125 | 0.6379421262578895 | 0.8450516494501833 |
| SP500_h5 | B3-2_fixed_exit | 0.889 | 0.7656714497794009 | 0.6912955988329598 | 0.9496834388537154 |
| SP500_h5 | B3-3 | 0.804 | 0.7084822146268706 | 0.6325511198326893 | 0.8770409863378874 |
| SP500_h5 | B3-3_fixed_exit | 0.804 | 0.7084822146268706 | 0.6325511198326893 | 0.8770409863378874 |
| SP500_h5 | scheduled_long | 0.874 | 0.7650098247055821 | 0.6880354614640846 | None |
| SP500_h5 | buy_hold | 1.286 | 0.9193004647192995 | 0.8789818512772325 | 0.0003332222592469177 |
| SP500_h20 | B3-1 | 1.067 | 0.39751088877787844 | 0.18936911116469213 | 0.7257580806397868 |
| SP500_h20 | B3-1_fixed_exit | 1.239 | 0.5064426209106441 | 0.2742180763214843 | 0.9936687770743086 |
| SP500_h20 | B3-2 | 0.470 | 0.11758925845257828 | 0.036509089910955715 | 0.24125291569476842 |
| SP500_h20 | B3-2_fixed_exit | 0.849 | 0.2741350348921706 | 0.11269435540945277 | 0.47184271909363545 |
| SP500_h20 | B3-3 | 0.013 | 0.024187751358564585 | 0.0046221567432841205 | 0.007676902536715621 |
| SP500_h20 | B3-3_fixed_exit | 0.013 | 0.024187751358564585 | 0.0046221567432841205 | 0.007676902536715621 |
| SP500_h20 | scheduled_long | 1.237 | 0.5052570861386905 | 0.26619647051709927 | None |
| SP500_h20 | buy_hold | 1.291 | 0.5402852492026866 | 0.2967810448367071 | 0.21992669110296567 |
| KOSPI_h5 | SMA200 | 1.071 | 0.9183920867910538 | 0.9047826820102537 | 0.3865378207264245 |
| KOSPI_h5 | MOM12_1 | 1.087 | 0.9231106002198561 | 0.9100187912438632 | 0.5631456181272909 |
| KOSPI_h5 | VOL15 | 1.182 | 0.9372757640709158 | 0.9263427311271577 | 0.8077307564145285 |
| KOSPI_h5 | always_long_guard | 1.252 | 0.9531437104482414 | 0.9442052315728556 | None |
| KOSPI_h5 | MiDM_archived | 1.194 | 0.9431623306212953 | 0.932802082728547 | 0.1622792402532489 |
| KOSPI_h20 | SMA200 | 1.419 | 0.9398190647604076 | 0.9142608965693764 | 0.49050316561146284 |
| KOSPI_h20 | MOM12_1 | 0.988 | 0.8226746007729717 | 0.7688266849899923 | 0.44551816061312893 |
| KOSPI_h20 | VOL15 | 1.392 | 0.9289861229595491 | 0.9011119518077018 | 0.6624458513828724 |
| KOSPI_h20 | always_long_guard | 1.301 | 0.918068030038206 | 0.8857529998400433 | None |
| KOSPI_h20 | MiDM_archived | 1.274 | 0.9117724848957269 | 0.8777081272961162 | 0.692102632455848 |

## A10 h20 상세 표
| h | Model | Method | Nominal | Coverage | Mean size | Delta vs B3 pair |
|---|---|---|---:|---:|---:|---:|
| 20 | B3_probabilities | ordinary_reused_calibration | 0.5 | 46.5224% | 1.564 | -9.4281% |
| 20 | B3_probabilities | ordinary_reused_calibration | 0.8 | 81.7620% | 3.816 | +25.8114% |
| 20 | B3_probabilities | ordinary_reused_calibration | 0.9 | 96.9088% | 5.934 | +40.9583% |
| 20 | B3_probabilities | block80_max | 0.5 | 100.0000% | 7.000 | +44.0495% |
| 20 | B3_probabilities | block80_max | 0.8 | 100.0000% | 7.000 | +44.0495% |
| 20 | B3_probabilities | block80_max | 0.9 | 100.0000% | 7.000 | +44.0495% |
| 20 | B3_probabilities | calibration_size_budget2 | None | 46.9861% | 1.580 | -8.9645% |
| 20 | training_frequency | ordinary_reused_calibration | 0.5 | 57.8053% | 2.000 | +1.8547% |
| 20 | training_frequency | ordinary_reused_calibration | 0.8 | 90.5719% | 5.000 | +34.6213% |
| 20 | training_frequency | ordinary_reused_calibration | 0.9 | 90.5719% | 5.000 | +34.6213% |
| 20 | training_frequency | block80_max | 0.5 | 100.0000% | 7.000 | +44.0495% |
| 20 | training_frequency | block80_max | 0.8 | 100.0000% | 7.000 | +44.0495% |
| 20 | training_frequency | block80_max | 0.9 | 100.0000% | 7.000 | +44.0495% |
| 20 | training_frequency | calibration_size_budget2 | None | 57.8053% | 2.000 | +1.8547% |
| 20 | B3_pair_Top2 | fixed2 | None | 55.9505% | 2.000 | +0.0000% |

전체 A10 h5/h20 결과와 부트스트랩 구간·클래스별 커버리지는 A10/RESULTS.json에 있다.

## 원고 검토 기록

manuscript_review/REVIEW_KO.md를 동봉했다. 사용자 원고 DOCX 및 원시 데이터·문항별 예측·스크립트는 이 묶음에 넣지 않았다.
