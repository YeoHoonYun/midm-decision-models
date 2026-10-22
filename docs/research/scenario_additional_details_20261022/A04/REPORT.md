# A04 actual local inference results

756actions/128report/96stock questions per checkpoint. Generic once plus three CE seeds. No repeated generic RNG runs presented as independent seeds.

| Checkpoint | Condition | Action accuracy | Balanced | Regret bp | Report | Stocks |
|---|---|---:|---:|---:|---:|---:|
| generic | original | 54.6296% | 49.5442% | 201.558 | 53.9062% | 58.3333% |
| generic | masked | 55.2910% | 49.4578% | 205.942 | 53.9062% | 58.3333% |
| generic | masked_normalized | 55.2910% | 49.4578% | 205.942 | 53.9062% | 86.4583% |
| CE_s20261008 | original | 58.2011% | 54.2046% | 192.886 | 53.9062% | 59.3750% |
| CE_s20261008 | masked | 55.4233% | 50.6499% | 203.746 | 54.6875% | 59.3750% |
| CE_s20261008 | masked_normalized | 55.4233% | 50.6499% | 203.746 | 54.6875% | 86.4583% |
| CE_s20261009 | original | 50.0000% | 50.4080% | 232.107 | 55.4688% | 61.4583% |
| CE_s20261009 | masked | 49.8677% | 48.6967% | 219.885 | 55.4688% | 61.4583% |
| CE_s20261009 | masked_normalized | 49.8677% | 48.6967% | 219.885 | 55.4688% | 84.3750% |
| CE_s20261010 | original | 47.7513% | 51.7328% | 233.171 | 55.4688% | 60.4167% |
| CE_s20261010 | masked | 51.5873% | 50.8018% | 224.812 | 55.4688% | 60.4167% |
| CE_s20261010 | masked_normalized | 51.5873% | 50.8018% | 224.812 | 55.4688% | 85.4167% |

Actions already omit full calendar dates/absolute price paths; dates exist in row metadata but Encoder receives state/keys/choices only. No inference that zero date masking changes proves no pretraining contamination. FinQA monetary values are not arbitrarily normalized as stock prices. Registered identifier patterns do not certify all entity names were removed. Numerical normalized stock inputs retain scale-invariant labels.
Paired intervals in RESULTS.json. No statistically confirmed removal of look-ahead bias is claimed.
