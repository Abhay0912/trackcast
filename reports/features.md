# Feature report (Phase 2)

Source `clean.parquet` -> `features.parquet`, horizon = 1 stop(s): stop k is predicted from delays observed at stops <= k - 1 plus the timetable.

- 36,500,515 stops -> **34,195,522 model rows** (observed delay, not the origin, at least one observed upstream delay).
- Late > 15 min: 52.57%.
- Lag taken from further back than the previous stop: 0.87% of rows; lag2 missing: 5.34%.
- Statistics below use a random subsample of 467,759 rows.

## Persistence baseline (predict delay_k = lag1)

| RMSLE | R2 (log1p) | F1 (late) | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.9688 | 0.5958 | 0.8755 | 0.8828 | 0.8683 |

Delay is highly persistent stop to stop, so this naive forecast is strong. Phase 3 models are judged by how much they improve on it, not by their absolute R2.

## Does schedule slack absorb delay?

Trains already late (lag1 > 15 min), by scheduled slack between the lag stop and stop k. Change = delay_k - lag1 (negative = recovered).

| Slack (min) | Rows | Mean change | Median change | % recovered |
|---|---:|---:|---:|---:|
| 0 | 59,323 | 0.58 | 0.0 | 30.93 |
| (0,2] | 64,953 | 1.21 | 0.0 | 32.0 |
| (2,5] | 38,941 | 2.08 | 0.0 | 38.42 |
| (5,10] | 28,025 | 1.06 | 0.0 | 48.55 |
| (10,20] | 25,133 | -2.45 | -4.0 | 60.25 |
| (20,40] | 17,777 | -11.97 | -16.0 | 76.08 |
| >40 | 7,866 | -28.54 | -29.0 | 86.24 |

## Features

**cascade**

| Feature | Non-null % | Mean | Median | Min | Max | Spearman vs log1p(delay) |
|---|---:|---:|---:|---:|---:|---:|
| lag1 | 100.0 | 36.5793 | 17.0 | 0.0 | 1440.0 | 0.8363 |
| lag2 | 94.65 | 36.3062 | 16.0 | 0.0 | 1439.0 | 0.7749 |
| delay_velocity | 94.65 | 1.7376 | 0.0 | -1242.0 | 1413.0 | 0.183 |
| lag_stops_back | 100.0 | 1.0101 | 1.0 | 1.0 | 46.0 | -0.0 |
| lag_sched_min_back | 100.0 | 24.2055 | 12.0 | 0.0 | 1325.0 | 0.0053 |
| lag_from_origin | 100.0 | 0.0524 | 0.0 | 0.0 | 1.0 | -0.1394 |
| upstream_max_delay | 100.0 | 47.2373 | 28.0 | 0.0 | 1440.0 | 0.7325 |
| n_upstream_obs | 100.0 | 14.1966 | 11.0 | 1.0 | 118.0 | 0.2486 |

**slack**

| Feature | Non-null % | Mean | Median | Min | Max | Spearman vs log1p(delay) |
|---|---:|---:|---:|---:|---:|---:|
| seg_km | 100.0 | 21.1014 | 10.0 | 0.0 | 465.0 | 0.0693 |
| seg_sched_min | 100.0 | 23.9007 | 12.0 | 0.0 | 475.0 | 0.0051 |
| seg_speed_kmph | 99.98 | 54.2881 | 54.5455 | 0.0 | 900.0 | 0.1174 |
| seg_n_trains | 100.0 | 11.0331 | 7.0 | 1.0 | 115.0 | 0.0334 |
| run_slack_min | 100.0 | 5.4271 | 1.0 | 0.0 | 203.0 | 0.0039 |
| dwell_prev_min | 94.79 | 2.2473 | 1.0 | 0.0 | 110.0 | 0.0567 |
| slack_since_lag | 100.0 | 6.3221 | 2.0 | 0.0 | 363.0 | 0.0262 |

**route**

| Feature | Non-null % | Mean | Median | Min | Max | Spearman vs log1p(delay) |
|---|---:|---:|---:|---:|---:|---:|
| station_no | 100.0 | 15.3935 | 12.0 | 2.0 | 119.0 | 0.2487 |
| distance_from_origin | 100.0 | 297.1736 | 127.0 | 0.0 | 4188.0 | 0.2217 |
| route_frac | 100.0 | 0.5271 | 0.5287 | 0.0 | 1.0 | 0.1054 |
| n_sched_stops | 100.0 | 28.7546 | 25.0 | 3.0 | 119.0 | 0.2042 |
| is_terminus | 100.0 | 0.0521 | 0.0 | 0.0 | 1.0 | -0.1386 |

**time**

| Feature | Non-null % | Mean | Median | Min | Max | Spearman vs log1p(delay) |
|---|---:|---:|---:|---:|---:|---:|
| hour_sin | 100.0 | -0.0454 | -0.1089 | -1.0 | 1.0 | -0.0503 |
| hour_cos | 100.0 | -0.1227 | -0.2036 | -1.0 | 1.0 | 0.0297 |
| dow_sin | 100.0 | 0.0008 | 0.0 | -0.9749 | 0.9749 | -0.0077 |
| dow_cos | 100.0 | -0.0019 | -0.2225 | -0.901 | 1.0 | -0.0064 |
| month_sin | 100.0 | -0.0116 | 0.0 | -1.0 | 1.0 | -0.02 |
| month_cos | 100.0 | -0.0137 | -0.0 | -1.0 | 1.0 | 0.0556 |
| is_weekend | 100.0 | 0.2844 | 0.0 | 0.0 | 1.0 | 0.0037 |

**season**

| Feature | Non-null % | Mean | Median | Min | Max | Spearman vs log1p(delay) |
|---|---:|---:|---:|---:|---:|---:|
| fog_season | 100.0 | 0.1066 | 0.0 | 0.0 | 1.0 | 0.0718 |
| fog_night | 100.0 | 0.0507 | 0.0 | 0.0 | 1.0 | 0.0451 |
| monsoon | 100.0 | 0.3383 | 0.0 | 0.0 | 1.0 | -0.0393 |

**congestion**

| Feature | Non-null % | Mean | Median | Min | Max | Spearman vs log1p(delay) |
|---|---:|---:|---:|---:|---:|---:|
| station_trains_hour | 100.0 | 3.5805 | 2.0 | 1.0 | 35.0 | 0.0291 |
| station_trains_day | 100.0 | 60.7322 | 28.0 | 1.0 | 469.0 | 0.0426 |

**categorical** (encoded in Phase 3): station_name, type_code, station_zone
