# Delay and route profiles (Phase 4)

Source `clean.parquet`: 36,500,515 stops of 6,908 trains at 8,142 stations (29,502 station pairs). Tables in `data\processed\profiles/`. All delays in minutes.

- Network: mean arrival delay **36.05 min**, **50.59%** of arrivals late > 15 min.
- *Change* = delay at the stop minus delay at the previous observed stop of the same run (negative = time recovered).
- Rankings need support: stations >= 2,000 arrivals, segments >= 500 late arrivals, trains >= 60 runs.

## How far does a delay cascade?

For every observed stop, the delay h observed stops later in the same run, by the delay at the start.

| h | Initial delay | Pairs | Mean initial | Mean after h | Change | % late after h | % lower after h |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | <= 15 | 16,498,274 | 5.16 | 8.42 | 3.26 | 14.31 | 26.48 |
| 1 | (15, 60] | 12,088,189 | 31.93 | 30.93 | -1.0 | 83.63 | 43.24 |
| 1 | (60, 180] | 4,570,114 | 98.91 | 96.93 | -1.98 | 97.76 | 41.81 |
| 1 | > 180 | 1,038,945 | 310.2 | 311.06 | 0.86 | 99.73 | 40.27 |
| 2 | <= 15 | 15,674,755 | 5.17 | 10.46 | 5.29 | 21.28 | 25.63 |
| 2 | (15, 60] | 11,425,691 | 31.93 | 31.62 | -0.31 | 79.36 | 45.99 |
| 2 | (60, 180] | 4,299,256 | 98.81 | 97.88 | -0.93 | 96.98 | 43.65 |
| 2 | > 180 | 965,536 | 309.15 | 313.34 | 4.2 | 99.62 | 40.95 |
| 3 | <= 15 | 14,857,865 | 5.17 | 12.32 | 7.15 | 26.62 | 24.96 |
| 3 | (15, 60] | 10,751,837 | 31.92 | 32.46 | 0.54 | 76.56 | 47.42 |
| 3 | (60, 180] | 4,029,234 | 98.76 | 98.65 | -0.11 | 96.19 | 44.98 |
| 3 | > 180 | 896,018 | 308.36 | 315.34 | 6.98 | 99.55 | 41.28 |
| 5 | <= 15 | 13,217,795 | 5.17 | 15.48 | 10.31 | 34.16 | 23.7 |
| 5 | (15, 60] | 9,456,820 | 31.89 | 34.28 | 2.39 | 73.18 | 48.65 |
| 5 | (60, 180] | 3,512,729 | 98.65 | 100.39 | 1.73 | 95.0 | 46.3 |
| 5 | > 180 | 770,791 | 307.26 | 319.0 | 11.74 | 99.47 | 41.29 |
| 10 | <= 15 | 9,386,628 | 5.15 | 21.84 | 16.69 | 44.52 | 21.65 |
| 10 | (15, 60] | 6,605,955 | 31.83 | 39.04 | 7.21 | 69.66 | 48.95 |
| 10 | (60, 180] | 2,421,367 | 98.53 | 103.74 | 5.21 | 92.48 | 48.09 |
| 10 | > 180 | 524,066 | 304.96 | 325.95 | 20.98 | 99.29 | 40.87 |
| 15 | <= 15 | 6,351,360 | 5.15 | 27.37 | 22.22 | 50.02 | 20.34 |
| 15 | (15, 60] | 4,423,652 | 31.81 | 43.64 | 11.83 | 68.42 | 48.57 |
| 15 | (60, 180] | 1,624,474 | 98.51 | 107.55 | 9.04 | 90.83 | 48.48 |
| 15 | > 180 | 354,160 | 303.03 | 331.89 | 28.87 | 99.11 | 40.2 |
| 20 | <= 15 | 4,175,547 | 5.15 | 32.31 | 27.16 | 53.43 | 19.36 |
| 20 | (15, 60] | 2,911,236 | 31.78 | 48.22 | 16.44 | 68.21 | 47.63 |
| 20 | (60, 180] | 1,076,438 | 98.48 | 111.02 | 12.54 | 89.38 | 48.43 |
| 20 | > 180 | 235,498 | 301.91 | 336.62 | 34.7 | 98.99 | 39.64 |

## Stations with the highest mean arrival delay

| Code | Station | Zone | Arrivals | Mean | Median | p90 | % late | Mean change | % late arrivals that recover |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MOU | MANOHARPUR | SER | 6,033 | 142.1 | 109 | 296 | 88.71 | 13.62 | 9.36 |
| CTHR | CHITAHRA | WCR | 2,189 | 133.77 | 132 | 238 | 94.15 | 8.3 | 31.89 |
| SWR | SONUA | SER | 3,422 | 133.49 | 97 | 302 | 81.53 | 5.13 | 16.58 |
| GOL | GOILKERA | SER | 5,664 | 132.41 | 99 | 281 | 87.89 | 7.63 | 15.91 |
| SRC | SANTRAGACHI JN | SER | 19,380 | 131.95 | 68 | 350 | 69.99 | 24.88 | 25.57 |
| CIH | CHILHIA | NER | 4,168 | 131.74 | 81 | 317 | 84.26 | 6.09 | 28.57 |
| PRZ | PARSA | NER | 4,167 | 131.57 | 78 | 327 | 78.64 | 6.51 | 30.63 |
| KPB | KHARPOKHRA | ECR | 2,147 | 131.07 | 103 | 285 | 85.89 | 2.1 | 27.41 |
| CKP | CHAKRADHARPUR | SER | 15,016 | 130.92 | 94 | 290 | 78.18 | 17.83 | 24.55 |
| AWS | ASWANI HALT | ECR | 2,148 | 130.88 | 100 | 288 | 85.06 | -2.27 | 80.12 |
| CAMU | CHAMUA | ECR | 2,147 | 130.49 | 102 | 282 | 83.88 | 4.82 | 33.69 |
| BRJN | BRAJRAJNAGAR | SECR | 6,627 | 129.86 | 91 | 282 | 90.45 | 4.45 | 35.31 |
| TKYR | TIKARIA | WCR | 2,189 | 129.65 | 127 | 233 | 95.16 | 9.59 | 9.14 |
| KSNR | KHUSHAL NAGAR | NER | 2,511 | 126.37 | 96 | 278 | 83.23 | -7.25 | 51.03 |
| SAGM | SAGMA T | WCR | 2,189 | 125.46 | 123 | 236 | 89.13 | 12.0 | 33.15 |

## Stations where trains gain the most delay (mean change on arrival)

| Code | Station | Zone | Arrivals | Mean | Median | p90 | % late | Mean change | % late arrivals that recover |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| SBGB | SUTEI BADAGAN | SER | 2,160 | 52.53 | 0 | 35 | 13.15 | 48.99 | 76.92 |
| BTQ | BETNOTI | SER | 3,539 | 75.14 | 4 | 184 | 35.74 | 36.98 | 35.64 |
| TRDI | TIRODI | SECR | 4,842 | 62.67 | 9 | 142 | 45.17 | 32.88 | 43.92 |
| SRC | SANTRAGACHI JN | SER | 19,380 | 131.95 | 68 | 350 | 69.99 | 24.88 | 25.57 |
| TSR | TSUNDURU | SCR | 4,006 | 80.01 | 70 | 156 | 93.76 | 23.62 | 5.34 |
| LAD | LOHARDAGA BS | SER | 3,983 | 28.82 | 2 | 21 | 14.91 | 23.53 | 51.72 |
| BRQ | BAGRAKOT | NFR | 2,796 | 74.87 | 39 | 213 | 75.36 | 22.28 | 10.59 |
| ANB | AMMANABROLU | SCR | 5,097 | 73.51 | 68 | 131 | 90.7 | 21.7 | 14.68 |
| NEO | NEORA | ECR | 7,119 | 56.78 | 41 | 127 | 77.6 | 21.6 | 5.88 |
| BRA | BIRSOLA | SECR | 5,607 | 74.56 | 60 | 166 | 79.42 | 21.42 | 4.71 |

## Segments where late trains lose the most time

| From | To | Pairs | Sched min | Mean change | Mean change if late | % late that recover |
|---|---|---:|---:|---:|---:|---:|
| ASH | BST | 686 | 199.5 | 91.25 | 96.51 | 2.71 |
| BST | ASH | 686 | 249.57 | 75.86 | 82.58 | 7.35 |
| CKP | TATA | 2,743 | 56.82 | 73.85 | 79.68 | 3.69 |
| NDLS | CNB | 5,067 | 295.36 | 31.67 | 72.16 | 17.79 |
| PUNA | TRDI | 1,310 | 17.7 | 61.53 | 70.83 | 36.29 |
| CPR | MFP | 965 | 138.65 | 66.22 | 66.83 | 9.31 |
| GNHI | NEO | 1,456 | 7.0 | 60.37 | 62.1 | 4.15 |
| MFP | CPR | 966 | 151.34 | 53.23 | 54.42 | 13.71 |
| MB | LKO | 1,037 | 317.28 | 43.49 | 46.59 | 22.49 |
| TATA | CKP | 2,751 | 49.39 | 42.95 | 46.51 | 1.61 |

## Segments where late trains recover the most time

| From | To | Pairs | Sched min | Mean change | Mean change if late | % late that recover |
|---|---|---:|---:|---:|---:|---:|
| NDKD | GZA | 1,094 | 14.0 | -57.55 | -84.21 | 100.0 |
| MAGH | BPQ | 659 | 110.89 | -59.03 | -73.23 | 99.12 |
| BXLL | CPU | 730 | 93.5 | -71.2 | -73.05 | 99.44 |
| DTV | JSG | 1,032 | 76.78 | -46.58 | -72.11 | 96.9 |
| DTX | NBQ | 729 | 92.43 | -69.49 | -71.88 | 99.29 |
| CCT | COA | 1,883 | 43.25 | -66.31 | -70.68 | 99.89 |
| BYPL | SMVB | 727 | 63.69 | -63.13 | -68.54 | 99.7 |
| BTA | PATL | 1,456 | 4.0 | -58.02 | -67.24 | 99.36 |
| DKI | DRA | 728 | 89.5 | -65.78 | -67.01 | 99.44 |
| PK | SUR | 725 | 97.5 | -64.07 | -65.34 | 100.0 |

## Suspected timetable errors (sawtooth stations)

67 of 5,385 ranked stations show a change on arrival of at least 15 min that the next segment undoes (opposite sign, also >= 15 min). The recorded delay there is systematically off, most likely because the timetable time at that station is wrong, so it is not real delay or recovery. These stations and the 264 ranked segments touching them are left out of the change-based rankings above, as are stations whose arrivals mostly come from such a station (they inherit the mirror-image error). Everything stays in the tables, flagged.

| Code | Station | Zone | Arrivals | Change on arrival | Change on next segment |
|---|---|---|---:|---:|---:|
| JAQ | JANDRAPETA | SCR | 2,916 | -48.15 | 53.08 |
| RPRL | RAPARLA H | SCR | 2,916 | -46.67 | 51.78 |
| IPPM | IPURUPALEM | SCR | 2,916 | -46.58 | 49.86 |
| MCVM | MACHAVARAM | SCR | 2,551 | -44.41 | 50.24 |
| MHBT | MAHABUBNAGAR TOWN H | SCR | 2,485 | -43.71 | 40.67 |
| MDKU | MODUKURU | SCR | 2,551 | -43.11 | 51.19 |
| MQN | MANYAMKONDA | SCR | 2,837 | 42.07 | -37.54 |
| THPR | TIPPAPUR | SCR | 3,179 | -36.82 | 34.64 |
| KPLL | KOTTA PNDLPALLI | SCR | 2,551 | -33.82 | 15.54 |
| PRH | POWERPET | SCR | 3,605 | -31.69 | 34.75 |

## Trains with the highest mean terminus delay

| Train | Name | Type | Runs | Km | Origin delay | Terminus delay | Gained en route | % late at terminus |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 4651 | PNBE FZR HUMSAFAR | PRM-TRAINS | 156 | 1670.0 | 176.54 | 559.33 | 382.79 | 100.0 |
| 4652 | JYG HUMSAFR SPL | PRM-TRAINS | 156 | 1670.0 | 71.32 | 530.7 | 459.38 | 99.36 |
| 12152 | SAMARSATA EXP | SF-TRAINS | 102 | 2079.0 | 179.22 | 518.29 | 340.94 | 94.85 |
| 2570 | DBG HUMSAFAR | PRM-TRAINS | 356 | 1169.0 | 182.26 | 506.23 | 323.97 | 99.72 |
| 2563 | BJU NDLS HUMSFAR | PRM-TRAINS | 355 | 1182.0 | 96.09 | 489.62 | 399.23 | 99.43 |
| 2569 | DBG NDLS HMSAFAR | PRM-TRAINS | 357 | 1169.0 | 53.11 | 488.32 | 437.06 | 100.0 |
| 14111 | MFP PRYJ EXP | SF-TRAINS | 78 | 665.0 | 205.56 | 455.3 | 195.09 | 95.45 |
| 2564 | BJU HUMSAFAR | PRM-TRAINS | 354 | 1182.0 | 60.41 | 451.22 | 392.23 | 98.87 |
| 12130 | AZAD HIND EXP | SF-TRAINS | 363 | 2015.0 | 102.13 | 400.42 | 308.76 | 93.61 |
| 12203 | GARIB RATH EXP | GRB-TRAINS | 156 | 1716.0 | 129.12 | 381.69 | 252.57 | 98.72 |
| 12871 | ISPAT EXPRESS | SF-TRAINS | 156 | 745.0 | 185.94 | 380.26 | 185.15 | 92.14 |
| 18478 | KALINGAUTKAL EXP | EXP-TRAINS | 365 | 2386.0 | 62.93 | 375.29 | 314.28 | 94.78 |
| 1028 | GKP DR SPL | EXP-TRAINS | 198 | 1867.0 | 31.23 | 363.76 | 349.76 | 97.45 |
| 1026 | BUI DR SPL | EXP-TRAINS | 148 | 1805.0 | 41.26 | 362.59 | 323.36 | 97.95 |
| 12129 | AZAD HIND EXP | SF-TRAINS | 365 | 2015.0 | 93.24 | 362.01 | 275.09 | 96.08 |

## By train type (weighted by runs)

| Type | Trains | Runs | Terminus delay | Gained en route |
|---|---:|---:|---:|---:|
| PASS-TRAINS | 3,056 | 1,028,942 | 26.83 | 14.26 |
| EXP-TRAINS | 2,056 | 471,263 | 26.05 | 18.76 |
| SF-TRAINS | 1,354 | 247,001 | 41.47 | 32.33 |
| T18-TRAINS | 194 | 42,631 | 11.82 | 7.48 |
| SHT-TRAINS | 42 | 13,162 | 14.34 | 9.81 |
| PRM-TRAINS | 106 | 11,449 | 133.46 | 99.94 |
| RAJ-TRAINS | 48 | 9,722 | 40.13 | 32.15 |
| GRB-TRAINS | 52 | 6,114 | 53.44 | 39.91 |

## Calendar

**Month** (of the scheduled arrival)

| Month | Arrivals | Mean delay | % late |
|---:|---:|---:|---:|
| 1 | 3,029,955 | 40.85 | 56.43 |
| 2 | 2,607,013 | 41.08 | 51.67 |
| 3 | 2,974,974 | 34.71 | 49.13 |
| 4 | 2,967,184 | 31.71 | 46.86 |
| 5 | 3,077,136 | 35.92 | 49.5 |
| 6 | 2,982,280 | 34.45 | 49.92 |
| 7 | 3,086,544 | 33.44 | 49.27 |
| 8 | 3,098,752 | 31.38 | 47.59 |
| 9 | 3,023,268 | 31.91 | 47.3 |
| 10 | 3,119,862 | 35.02 | 50.3 |
| 11 | 3,012,350 | 37.59 | 53.23 |
| 12 | 3,046,488 | 45.3 | 56.0 |

**Day of week**

| Day | Arrivals | Mean delay | % late |
|---|---:|---:|---:|
| Mon | 5,180,136 | 36.05 | 50.32 |
| Tue | 5,142,345 | 35.69 | 50.27 |
| Wed | 5,148,015 | 35.59 | 50.08 |
| Thu | 5,149,936 | 36.31 | 50.8 |
| Fri | 5,174,480 | 36.1 | 51.06 |
| Sat | 5,205,181 | 36.53 | 51.47 |
| Sun | 5,025,713 | 36.11 | 50.11 |

**Hour of scheduled arrival**

| Hour | Arrivals | Mean delay | % late |
|---:|---:|---:|---:|
| 0 | 730,354 | 42.34 | 56.77 |
| 1 | 637,093 | 44.16 | 58.5 |
| 2 | 653,554 | 45.49 | 59.15 |
| 3 | 734,129 | 42.78 | 54.44 |
| 4 | 995,399 | 35.68 | 45.59 |
| 5 | 1,526,159 | 29.26 | 40.84 |
| 6 | 1,969,703 | 27.39 | 41.48 |
| 7 | 2,300,384 | 28.1 | 44.94 |
| 8 | 2,243,641 | 30.38 | 49.13 |
| 9 | 1,961,344 | 33.79 | 51.43 |
| 10 | 1,766,918 | 36.29 | 52.36 |
| 11 | 1,531,682 | 37.51 | 52.54 |
| 12 | 1,361,576 | 38.44 | 51.38 |
| 13 | 1,284,854 | 38.74 | 51.31 |
| 14 | 1,441,872 | 37.49 | 49.65 |
| 15 | 1,611,287 | 37.29 | 50.03 |
| 16 | 1,701,480 | 37.32 | 50.38 |
| 17 | 1,850,576 | 35.37 | 48.77 |
| 18 | 2,166,133 | 34.3 | 49.84 |
| 19 | 2,116,096 | 38.89 | 55.7 |
| 20 | 1,810,738 | 40.82 | 55.36 |
| 21 | 1,518,614 | 40.53 | 55.17 |
| 22 | 1,168,771 | 41.23 | 54.28 |
| 23 | 943,449 | 40.79 | 53.91 |

Elapsed: 238.5 s.