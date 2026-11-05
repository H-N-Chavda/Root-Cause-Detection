# SUMMARY: unit-wise PCA → PC algorithm → causation-based T²

Generated 2026-11-05 16:03 from `outputs/` by `summary.py`.

## Data and cleaning
| item | value |
|---|---|
| time window | 2025-07-01 00:00:00 to 2025-11-01 00:00:00 (end excluded) |
| sampling interval | 0 days 00:01:00 |
| tags loaded | 416 (4 flare tags) |
| process tags used in PCA | 306 |
| tags dropped | 106: negative values 55, too much missing 40, constant 7, near-constant 4 |
| rows in matrix | 177120 |
| NOC rows (baseline) | 12157 |
| flare-event rows / pre-flare rows | 88590 / 14281 |
| flare_net threshold | 1.500 t/h (manual FLARE_THRESHOLD) |
| flare events (>= 30 min) | 620, 88590 event minutes |
| shorter runs above the threshold (not events, kept out of NOC) | 1723 |

## PCA variance
| unit | tags | rank | PCs kept | variance captured |
|---|---|---|---|---|
| APS_NVPS_LEU | 12 | 12 | 2 | 61.4% |
| DAK | 3 | 1 | 1 | 100.0% |
| DIU | 17 | 17 | 2 | 38.6% |
| ISOM | 41 | 41 | 2 | 38.1% |
| ISOM_3 | 14 | 14 | 2 | 54.7% |
| ISOM_4 | 1 | 1 | 1 | 100.0% |
| LOUP | 52 | 52 | 2 | 69.3% |
| NSU_NHT_CCR | 53 | 53 | 2 | 42.4% |
| PDA | 28 | 28 | 2 | 45.7% |
| PG | 34 | 34 | 2 | 51.6% |
| SEU1 | 19 | 19 | 2 | 60.3% |
| SEU2 | 14 | 14 | 2 | 52.8% |
| SEU3 | 13 | 13 | 2 | 43.8% |
| unit_wise_control_data | 5 | 5 | 2 | 65.2% |
| **all units** | 306 | | 26 | 51.6% |

## Graph and loops
- PC algorithm input: 26 PCs (flare_net not included); dropped before PC (collinear / dependent): 0.
- PC output: 87 edges = 85 directed, 2 undirected, 0 bidirected.
- DAG among PCs: 87 edges = 9 same-unit, 78 cross-unit. Then every PC -> flare_net added: 26 edges.
- Loops: 2 directed edges reversed to break cycles; 0 forced Dor-Tarsi steps (no consistent extension); final graph is acyclic (checked in the run).
- Parents per PC (among PCs): mean 3.35, max 10; root PCs (no parents): 5.

## T² detection and root causes
- Baseline: moving window (1350 rows, step 10). Limits x2: overall 79.74, per-PC 7.702 to 7.760.
- Alarm: on after 3 min above the limit, off after 10 min below; TP = flare event within 30 min of the alarm switching on.

| metric | value |
|---|---|
| actual flare events (scored) | 591.0 |
| flare events not scored | 29.0 |
| alarms (activations) | 792.0 |
| alarms during a flare (not predictions) | 377.0 |
| predicted flare events (alarms outside flares) | 415.0 |
| correct predictions (TP alarms) | 76.0 |
| incorrect predictions (FP alarms) | 339.0 |
| TP (events detected) | 74.0 |
| FN (events missed) | 517.0 |
| FN: no alarm | 157.0 |
| FN: already in alarm | 360.0 |
| TN (30-min flare-free blocks, no alarm) | 728.0 |
| 30-min blocks with an FP activation | 209.0 |
| 30-min blocks inside a standing alarm (not counted) | 431.0 |
| FP alarm minutes | 21740.0 |
| base accuracy (TP+TN)/(TP+FN+FP+TN) | 0.484 |
| precision TP alarms/(TP+FP alarms) | 0.183 |
| recall TP events/(TP+FN events) | 0.125 |
| median lead time TP (min) | 13.5 |

| event | onset | min | peak t/h | outcome | lead (min) | top root-cause PCs (T²_j / limit) |
|---|---|---|---|---|---|---|
| E1 | 07-01 00:00 | 3537 | 6.77 | not scored | - | - |
| E2 | 07-03 11:51 | 79 | 3.33 | not scored | - | - |
| E3 | 07-03 13:15 | 469 | 4.70 | not scored | - | - |
| E4 | 07-03 21:28 | 369 | 3.17 | not scored | - | - |
| E5 | 07-04 05:28 | 66 | 1.72 | FN (already in alarm) | - | ISOM_4_PC1 12.7; APS_NVPS_LEU_PC2 3.3; DIU_PC1 2.0 |
| E6 | 07-04 08:30 | 592 | 2.62 | FN (already in alarm) | - | NSU_NHT_CCR_PC1 5.3; ISOM_PC1 2.1; DAK_PC1 2.1 |
| E7 | 07-05 01:01 | 43 | 1.91 | FN (already in alarm) | - | LOUP_PC1 1.2; APS_NVPS_LEU_PC2 0.9; LOUP_PC2 0.8 |
| E8 | 07-05 09:47 | 63 | 2.02 | FN (already in alarm) | - | ISOM_PC2 1.1; APS_NVPS_LEU_PC2 0.7; SEU1_PC1 0.7 |
| E9 | 07-05 10:59 | 269 | 2.28 | FN (already in alarm) | - | SEU1_PC1 9.0; SEU2_PC1 1.8; PDA_PC1 1.3 |
| E10 | 07-06 13:59 | 49 | 1.82 | TP | 8 | SEU2_PC1 14.3; SEU3_PC1 1.1; APS_NVPS_LEU_PC1 0.7 |
| E11 | 07-06 15:38 | 73 | 1.74 | TP | 22 | LOUP_PC2 1.5; ISOM_3_PC2 1.0; DIU_PC2 0.8 |
| E12 | 07-06 18:40 | 40 | 2.34 | TP | 2 | LOUP_PC1 2.6; ISOM_PC2 1.9; ISOM_3_PC1 1.2 |
| E13 | 07-07 07:10 | 121 | 6.01 | FN (already in alarm) | - | ISOM_4_PC1 22.2; ISOM_PC1 2.0; DAK_PC1 1.5 |
| E14 | 07-07 10:33 | 206 | 2.65 | FN (no alarm) | - | DAK_PC1 0.9; NSU_NHT_CCR_PC1 0.8; NSU_NHT_CCR_PC2 0.8 |
| E15 | 07-08 10:47 | 38 | 2.83 | FN (already in alarm) | - | SEU3_PC1 75.1; APS_NVPS_LEU_PC1 7.9; NSU_NHT_CCR_PC1 6.6 |
| E16 | 07-08 14:50 | 62 | 1.89 | FN (no alarm) | - | SEU2_PC1 109.0; SEU3_PC1 16.5; SEU3_PC2 7.8 |
| E17 | 07-08 16:15 | 41 | 1.68 | FN (already in alarm) | - | NSU_NHT_CCR_PC2 1.9; ISOM_PC2 1.4; unit_wise_control_data_PC1 1.2 |
| E18 | 07-08 17:30 | 61 | 2.15 | FN (already in alarm) | - | SEU3_PC1 11.9; SEU2_PC1 2.2; DAK_PC1 1.2 |
| E19 | 07-09 07:52 | 38 | 1.64 | FN (no alarm) | - | SEU1_PC1 0.7; NSU_NHT_CCR_PC1 0.3; PG_PC1 0.3 |
| E20 | 07-09 15:32 | 63 | 2.76 | FN (already in alarm) | - | APS_NVPS_LEU_PC1 1.2; ISOM_PC2 1.1; ISOM_PC1 0.8 |
| E21 | 07-09 16:54 | 183 | 4.21 | FN (already in alarm) | - | APS_NVPS_LEU_PC1 3.1; DAK_PC1 1.2; NSU_NHT_CCR_PC1 0.8 |
| E22 | 07-10 09:12 | 58 | 5.51 | TP | 24 | DIU_PC1 0.9; SEU1_PC1 0.8; DAK_PC1 0.5 |
| E23 | 07-10 10:18 | 45 | 1.66 | FN (already in alarm) | - | SEU2_PC2 5.5; SEU1_PC1 3.8; SEU3_PC1 1.9 |
| E24 | 07-10 11:52 | 68 | 1.82 | FN (already in alarm) | - | SEU1_PC2 4.4; NSU_NHT_CCR_PC2 1.8; SEU1_PC1 1.3 |
| E25 | 07-10 14:01 | 90 | 3.70 | FN (already in alarm) | - | SEU3_PC2 5.1; SEU2_PC2 3.9; APS_NVPS_LEU_PC2 2.1 |
| E26 | 07-10 18:14 | 34 | 4.24 | FN (already in alarm) | - | DAK_PC1 9.2; LOUP_PC1 4.2; SEU3_PC2 1.8 |
| E27 | 07-10 19:02 | 33 | 2.41 | FN (already in alarm) | - | LOUP_PC1 1.4; SEU2_PC2 0.7; LOUP_PC2 0.6 |
| E28 | 07-10 20:22 | 132 | 4.78 | FN (already in alarm) | - | SEU3_PC2 1.4; LOUP_PC1 1.1; SEU1_PC2 0.7 |
| E29 | 07-10 23:02 | 72 | 2.04 | FN (already in alarm) | - | NSU_NHT_CCR_PC1 4.4; APS_NVPS_LEU_PC1 1.0; SEU1_PC1 1.0 |
| E30 | 07-11 01:02 | 43 | 2.66 | FN (no alarm) | - | unit_wise_control_data_PC1 1.0; DAK_PC1 0.9; unit_wise_control_data_PC2 0.7 |
| E31 | 07-11 06:58 | 32 | 2.97 | TP | 21 | DAK_PC1 1.9; APS_NVPS_LEU_PC1 0.6; LOUP_PC2 0.6 |
| E32 | 07-11 08:00 | 35 | 1.72 | TP | 3 | SEU3_PC1 2.6; NSU_NHT_CCR_PC1 1.7; SEU3_PC2 0.8 |
| E33 | 07-11 11:23 | 82 | 2.26 | TP | 19 | PG_PC1 4.9; PG_PC2 2.5; SEU3_PC1 0.8 |
| E34 | 07-11 13:09 | 168 | 6.09 | FN (already in alarm) | - | SEU1_PC2 1.6; LOUP_PC2 1.4; ISOM_3_PC2 1.1 |
| E35 | 07-11 16:17 | 208 | 6.48 | FN (already in alarm) | - | DIU_PC2 11.9; LOUP_PC2 2.5; SEU3_PC2 1.8 |
| E36 | 07-11 21:23 | 97 | 4.35 | FN (already in alarm) | - | DIU_PC1 4.5; ISOM_PC2 1.9; DIU_PC2 1.8 |
| E37 | 07-11 23:05 | 69 | 1.92 | FN (already in alarm) | - | PG_PC1 6.6; APS_NVPS_LEU_PC2 1.2; PDA_PC2 1.1 |
| E38 | 07-12 02:12 | 54 | 1.94 | FN (no alarm) | - | PG_PC1 2.1; NSU_NHT_CCR_PC1 1.4; ISOM_PC2 0.7 |
| E39 | 07-12 03:49 | 85 | 1.87 | FN (no alarm) | - | NSU_NHT_CCR_PC1 3.2; DAK_PC1 1.6; NSU_NHT_CCR_PC2 0.5 |
| E40 | 07-12 06:05 | 47 | 1.91 | FN (no alarm) | - | NSU_NHT_CCR_PC1 0.8; SEU2_PC2 0.8; unit_wise_control_data_PC2 0.6 |

580 more events in `tables/event_outcomes.csv` and `RCA_report.xlsx`.

- TA cross-check: not applicable, the TA's period [2025-05-01, 2025-06-01) is outside the time window.
