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
