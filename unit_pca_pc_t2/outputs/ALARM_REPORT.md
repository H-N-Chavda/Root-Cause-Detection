# Alarm report v2 (2026-11-05): moving-window T², x2 limits, ISA-18.2 alarm counting

Replaces v1 (2026-11-04): v1 used a fixed Jul-Sep baseline; T² drifted above the limit from September on, so v1 overstated detection.

## Changes
| item | value |
|---|---|
| baseline | trailing 1,250 flare-free-filtered rows (50 x 25 vars, TA Main.py), re-estimated every 10 min |
| limits | Eq. 13, m = 1,250, x2: overall 74.69, per-PC 7.70-7.76 |
| alarm | on after 3 min above, off after 10 min below; 1 activation = 1 alarm |
| constant-slope rule (TA) | added; drops 0 tags on local Dataset_HPCL, so PCs and graph unchanged (no PC rerun) |
| spline fill | already TA's cubic spline; flare/recovery tags have 0 gaps |
| cycles | PC output: 5 cycles; final graph: 0 (`check_cycles.py`, `graph/cycles.csv`) |

## Metrics (`tables/metrics.csv`, `RCA_report.xlsx`)
| count | value |
|---|---|
| actual flare events scored / not scored | 587 / 33 |
| predicted (alarms outside flares) | 405 = 65 correct (TP) + 340 incorrect (FP) |
| TP events / FN events | 65 / 522 (FN: 150 no alarm, 372 already in alarm) |
| TN (30-min flare-free blocks, no alarm) | 659 |
| FP alarm minutes | 27,292 |
| accuracy / precision / recall | 0.456 / 0.160 / 0.111 |
| median lead time (TP) | 11 min |

## Root cause (top-1 PC at the alarm's first minute above limit)
- Detected events: DAK_PC1 10, DIU_PC1 6, ISOM_PC2 5, SEU3_PC1 4, NSU_NHT_CCR_PC1 4.
- False alarms: DIU_PC1 38, DAK_PC1 36, APS_NVPS_LEU_PC2 25: same PCs drive true and false alarms.

## Findings
- Overall T² does not separate pre-flare from flare-free time: precision 16%, recall 11%.
- 372 of 522 misses: alarm already on when the 30-min window starts (flares 50% of time, median gap 49 min).
- DAK_PC1 (3 tags, rank 1) and ISOM_4_PC1 (1 tag) are flat in most windows; std floored at 1% of whole-period std, so small steps give large T²_j. UNVERIFIED whether DAK is a real cause.
- 33 events not scored: < 100 flare-free rows in the trailing 1,250.
