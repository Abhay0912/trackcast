# Model report (Phase 3)

Source `features.parquet`, horizon = 1 stop(s). Temporal holdout: train on runs starting 2025-02-08 to 2025-11-27, test on runs starting 2025-12-01 to 2026-02-07 (cutoff 2025-12-01, 3-day embargo before it).

- Train: 999,274 rows / 53,785 runs (51.09% late > 15 min), 3.6% of the training period sampled by run.
- Test: 996,417 rows / 52,968 runs (57.89% late), 15.6% of the test period sampled by run.
- Selected on CV: **Stage 1 = rf, Stage 2 = hgb**; saved to `models\hurdle.joblib` (62.1 MB). Test predictions: `data\processed\test_predictions.parquet`.

## Test: overall forecast of log1p(delay) (all test rows)

Hurdle = P(late) x m_late + (1 - P(late)) x m_low. *Direct* = one regressor on all rows. *Perfect Stage 1* uses the true class to pick the regressor: an upper bound on what a better classifier could add.

| Model | RMSLE | R2 (log1p) | MAE (min) | RMSLE skill vs persistence (%) |
|---|---:|---:|---:|---:|
| persistence | 0.9517 | 0.5991 | 8.30 | 0.0 |
| persistence_minus_slack | 1.0709 | 0.4924 | 8.87 | -12.5 |
| direct_ridge | 0.7449 | 0.7544 | 19.35 | 21.7 |
| hurdle_logreg+ridge | 0.7264 | 0.7665 | 16.24 | 23.7 |
| hurdle_rf+ridge | 0.7036 | 0.7809 | 15.25 | 26.1 |
| direct_hgb | 0.6153 | 0.8324 | 7.06 | 35.3 |
| hurdle_logreg+hgb | 0.6565 | 0.8093 | 6.82 | 31.0 |
| **hurdle_rf+hgb** (selected) | 0.6355 | 0.8212 | 6.83 | 33.2 |
| hurdle_perfect_stage1+hgb | 0.4567 | 0.9077 | 5.21 | 52.0 |

Day-block bootstrap (95% CI) of the RMSLE difference; positive = the selected model is better:

- persistence - selected: **+0.3161** [+0.3127, +0.3198] over 69 test days
- direct_hgb - selected: **-0.0202** [-0.0217, -0.0187] over 69 test days

## Test: Stage 1 (late > 15 min)

*Raw* = the classifier's own probabilities. *Calibrated* = after an isotonic map fitted on the pooled CV validation months; the F1 threshold is tuned there too, never on the test set. The hurdle uses the calibrated probabilities.

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| persistence (lag1 > 15) | 0.8870 | 0.8947 | 0.8794 |  |  |  |  |
| logreg raw @ 0.5 | 0.9034 | 0.9254 | 0.8825 | 0.9510 | 0.9660 | 0.0823 | 0.2769 |
| logreg calibrated @ CV-tuned 0.453 | 0.9038 | 0.9145 | 0.8934 | 0.9510 | 0.9652 | 0.0821 | 0.2744 |
| rf raw @ 0.5 | 0.9125 | 0.9366 | 0.8897 | 0.9623 | 0.9739 | 0.0733 | 0.2478 |
| rf calibrated @ CV-tuned 0.458 | 0.9133 | 0.9290 | 0.8981 | 0.9623 | 0.9732 | 0.0725 | 0.2414 |

## Test: Stage 2 (log1p delay, conditional on the true class)

| Model | Rows | RMSLE | R2 (log1p) | MAE (min) |
|---|---:|---:|---:|---:|
| persistence (late rows) | 576,864 | 0.7036 | 0.1775 | 10.18 |
| ridge (late rows) | 576,864 | 0.3306 | 0.8184 | 24.53 |
| hgb (late rows) | 576,864 | 0.2236 | 0.9169 | 7.35 |
| persistence (low rows) | 419,553 | 1.2125 | -0.3851 | 5.72 |
| ridge (low rows) | 419,553 | 0.7717 | 0.4390 | 2.92 |
| hgb (low rows) | 419,553 | 0.6532 | 0.5981 | 2.28 |

## Where the model helps (test set by segment)

**RMSLE by month**

| Segment | Rows | Late % | persistence | direct_hgb | selected |
|---|---:|---:|---:|---:|---:|
| 2025-12 | 450,401 | 58.1 | 0.9517 | 0.6035 | 0.6296 |
| 2026-01 | 444,924 | 58.1 | 0.9499 | 0.6245 | 0.6395 |
| 2026-02 | 101,092 | 56.2 | 0.9596 | 0.6269 | 0.6443 |

**RMSLE by train type**

| Segment | Rows | Late % | persistence | direct_hgb | selected |
|---|---:|---:|---:|---:|---:|
| EXP-TRAINS | 364,250 | 60.3 | 0.9878 | 0.6409 | 0.6585 |
| GRB-TRAINS | 2,419 | 59.8 | 1.1618 | 0.7335 | 0.7482 |
| PASS-TRAINS | 474,245 | 55.4 | 0.8802 | 0.5741 | 0.5988 |
| PRM-TRAINS | 4,407 | 66.6 | 1.1250 | 0.7419 | 0.7627 |
| RAJ-TRAINS | 3,138 | 55.4 | 1.2351 | 0.8227 | 0.8479 |
| SF-TRAINS | 135,894 | 61.5 | 1.0422 | 0.6571 | 0.6713 |
| SHT-TRAINS | 2,453 | 45.8 | 1.2373 | 0.7523 | 0.7842 |
| T18-TRAINS | 9,611 | 38.1 | 1.2326 | 0.7530 | 0.7619 |

**RMSLE by upstream delay (lag1, min)**

| Segment | Rows | Late % | persistence | direct_hgb | selected |
|---|---:|---:|---:|---:|---:|
| lag1 <= 15 | 429,413 | 16.2 | 1.1588 | 0.7442 | 0.7615 |
| 15-60 | 373,639 | 84.8 | 0.8610 | 0.5462 | 0.5754 |
| 60-180 | 154,638 | 98.1 | 0.5375 | 0.3938 | 0.4073 |
| > 180 | 38,727 | 99.7 | 0.3287 | 0.3213 | 0.3241 |

**MAE (minutes) by upstream delay (lag1, min)**

| Segment | Rows | Late % | persistence | direct_hgb | selected |
|---|---:|---:|---:|---:|---:|
| lag1 <= 15 | 429,413 | 16.2 | 5.71 | 4.43 | 4.61 |
| 15-60 | 373,639 | 84.8 | 8.50 | 6.35 | 6.24 |
| 60-180 | 154,638 | 98.1 | 12.63 | 10.97 | 10.05 |
| > 180 | 38,727 | 99.7 | 17.81 | 27.43 | 24.37 |

**RMSLE by slack since lag stop (min)**

| Segment | Rows | Late % | persistence | direct_hgb | selected |
|---|---:|---:|---:|---:|---:|
| slack 0 | 273,789 | 54.4 | 0.8805 | 0.5566 | 0.5825 |
| (0,5] | 433,880 | 59.9 | 0.8087 | 0.5514 | 0.5710 |
| (5,20] | 203,431 | 61.2 | 1.0046 | 0.6797 | 0.6939 |
| > 20 | 85,317 | 51.1 | 1.5355 | 0.8832 | 0.9067 |

## Feature-group importance (selected model, test subsample)

Increase in RMSLE when the group's columns are shuffled together (no refit).

| Group | RMSLE increase | Std |
|---|---:|---:|
| cascade | 1.2729 | 0.0057 |
| slack | 0.1254 | 0.0007 |
| station (code, zone) | 0.1045 | 0.0007 |
| route | 0.0558 | 0.0007 |
| congestion | 0.0346 | 0.0006 |
| train (number, type) | 0.0156 | 0.0002 |
| time | 0.0068 | 0.0003 |
| season | -0.0002 | 0.0000 |

## Cross-validation (training period only)

Forward-chaining folds: 2025-09 (fit 682,843 / val 106,803), 2025-10 (fit 788,847 / val 105,919), 2025-11 (fit 895,514 / val 93,054).
Metric: log loss for classifiers; RMSE of log1p(delay) on the part's rows for regressors.

| Family | Part | Params | Mean | Std | Fit s (all folds) | Chosen |
|---|---|---|---:|---:|---:|---|
| logreg | clf | {"C": 0.01} | 0.2823 | 0.0044 | 121.8 |  |
| logreg | clf | {"C": 0.1} | 0.2823 | 0.0044 | 130.8 | yes |
| logreg | clf | {"C": 1.0} | 0.2823 | 0.0044 | 132.2 |  |
| rf | clf | {"min_samples_leaf": 5} | 0.2503 | 0.0052 | 156.9 | yes |
| rf | clf | {"min_samples_leaf": 25} | 0.2648 | 0.0047 | 158.2 |  |
| ridge | late | {"alpha": 0.1} | 0.3274 | 0.0055 | 23.9 | yes |
| ridge | late | {"alpha": 1.0} | 0.3274 | 0.0055 | 22.7 |  |
| ridge | late | {"alpha": 10.0} | 0.3274 | 0.0055 | 22.4 |  |
| ridge | late | {"alpha": 100.0} | 0.3274 | 0.0055 | 25.9 |  |
| ridge | low | {"alpha": 0.1} | 0.7613 | 0.0036 | 23.8 |  |
| ridge | low | {"alpha": 1.0} | 0.7613 | 0.0036 | 21.0 |  |
| ridge | low | {"alpha": 10.0} | 0.7613 | 0.0036 | 20.9 |  |
| ridge | low | {"alpha": 100.0} | 0.7613 | 0.0036 | 21.6 | yes |
| ridge | direct | {"alpha": 0.1} | 0.7563 | 0.0109 | 48.7 |  |
| ridge | direct | {"alpha": 1.0} | 0.7563 | 0.0109 | 54.0 |  |
| ridge | direct | {"alpha": 10.0} | 0.7563 | 0.0109 | 50.3 |  |
| ridge | direct | {"alpha": 100.0} | 0.7563 | 0.0109 | 50.1 | yes |
| hgb | late | {"max_leaf_nodes": 31, "min_samples_leaf": 20} | 0.2270 | 0.0036 | 95.7 |  |
| hgb | late | {"max_leaf_nodes": 127, "min_samples_leaf": 100} | 0.2218 | 0.0041 | 147.9 | yes |
| hgb | low | {"max_leaf_nodes": 31, "min_samples_leaf": 20} | 0.6596 | 0.0016 | 95.8 |  |
| hgb | low | {"max_leaf_nodes": 127, "min_samples_leaf": 100} | 0.6387 | 0.0033 | 149.2 | yes |
| hgb | direct | {"max_leaf_nodes": 31, "min_samples_leaf": 20} | 0.6455 | 0.0141 | 172.3 |  |
| hgb | direct | {"max_leaf_nodes": 127, "min_samples_leaf": 100} | 0.6199 | 0.0163 | 244.8 | yes |

CV RMSLE of each hurdle pair, the direct regressors and the baselines (mean over folds):

| Model | CV RMSLE |
|---|---:|
| direct_hgb | 0.6199 |
| hurdle_rf+hgb | 0.6446 |
| hurdle_logreg+hgb | 0.6679 |
| hurdle_rf+ridge | 0.7163 |
| hurdle_logreg+ridge | 0.7414 |
| direct_ridge | 0.7563 |
| persistence | 0.9728 |
| persistence_minus_slack | 1.0908 |

CV-tuned Stage 1 thresholds: logreg 0.453, rf 0.458.

Elapsed: 2533.2 s.