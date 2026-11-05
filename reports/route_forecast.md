# Route-forecast evaluation (Phase 5)

The app's multi-stop forecast, simulated on 1,500 real runs from the test period (starting on or after 2025-12-01, never seen in training; `clean.parquet`, model `hurdle.joblib`). Each run starts at a random observed stop; only that stop's delay is known (as in the app), and the next 20 stops are forecast recursively. Strategies differ only in which minutes are fed forward:

- *median*: expm1(E[log1p delay]) (the Phase 3 point forecast)
- *mean*: P(late) x expm1(m_late) + (1 - P(late)) x expm1(m_low)
- *gate*: the branch the calibrated classifier picks (m_late if P(late) >= threshold, else m_low)
- *persistence*: the start delay, unchanged
- *cascade_profile*: start delay + the network-average change after h stops for its delay bucket (Phase 4; computed on all dates, so this baseline has seen the test months)

**MAE (minutes)**

| h | median | mean | gate | persistence | cascade_profile |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.90 | 6.46 | 6.22 | **6.11** | 6.58 |
| 2 | 12.02 | **9.22** | 9.40 | 10.01 | 10.16 |
| 3 | 15.06 | **11.56** | 11.95 | 12.99 | 12.97 |
| 5 | 19.71 | **15.40** | 16.36 | 17.67 | 17.11 |
| 10 | 28.91 | **22.83** | 24.61 | 26.55 | 25.10 |
| 15 | 37.65 | **30.00** | 32.10 | 34.72 | 32.65 |
| 20 | 41.93 | **33.13** | 35.45 | 38.46 | 38.73 |

**Bias: mean forecast - actual (minutes)**

| h | median | mean | gate | persistence | cascade_profile |
|---:|---:|---:|---:|---:|---:|
| 1 | -5.34 | -0.59 | **0.38** | -2.51 | -1.43 |
| 2 | -8.28 | -2.57 | **-1.43** | -4.66 | -2.09 |
| 3 | -11.08 | -4.38 | -3.18 | -6.60 | **-2.63** |
| 5 | -15.64 | -7.59 | -6.12 | -10.06 | **-3.50** |
| 10 | -25.08 | -14.55 | -13.20 | -16.89 | **-4.77** |
| 15 | -34.77 | -22.00 | -20.57 | -24.54 | **-7.33** |
| 20 | -39.45 | -25.46 | -23.24 | -25.26 | **-3.38** |

**RMSLE**

| h | median | mean | gate | persistence | cascade_profile |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.59 | 0.65 | **0.59** | 0.76 | 0.75 |
| 2 | **0.74** | 0.75 | 0.79 | 0.97 | 0.89 |
| 3 | 0.84 | **0.82** | 0.93 | 1.12 | 1.00 |
| 5 | 1.00 | **0.93** | 1.12 | 1.28 | 1.13 |
| 10 | 1.27 | **1.09** | 1.46 | 1.51 | 1.29 |
| 15 | 1.44 | **1.17** | 1.63 | 1.64 | 1.36 |
| 20 | 1.64 | **1.29** | 1.78 | 1.76 | 1.72 |

Rows per horizon: 1,493 (h = 1). Elapsed: 40.5 s.