# SUMMARY: unit-wise PCA → PC algorithm → causation-based T²

Generated 2026-11-06 01:29 from `outputs/` by `summary.py`.

## Data and cleaning
| item | value |
|---|---|
| time window | 2025-07-01 00:00:00 to 2025-11-01 00:00:00 (end excluded) |
| sampling interval | 0 days 00:01:00 |
| tags loaded | 380 (4 flare tags) |
| process tags used in PCA | 301 |
| tags dropped | 75: negative values 49, too much missing 19, constant 5, near-constant 2 |
| rows in matrix | 177120 |
| NOC rows (baseline) | 35978 |
| flare-event rows / pre-flare rows | 88590 / 23923 |
| flare_net threshold | 1.500 t/h (manual FLARE_THRESHOLD) |
| flare events (>= 30 min) | 620, 88590 event minutes |
| shorter runs above the threshold (not events, kept out of NOC) | 1723 |

## PCA variance
| unit | tags | rank | PCs kept | variance captured |
|---|---|---|---|---|
| APS_NVPS_LEU | 12 | 12 | 2 | 57.5% |
| DAK | 3 | 1 | 1 | 100.0% |
| DIU | 17 | 17 | 2 | 55.8% |
| ISOM | 41 | 41 | 2 | 32.6% |
| ISOM_3 | 14 | 14 | 2 | 49.7% |
| ISOM_4 | 1 | 1 | 1 | 100.0% |
| LOUP | 52 | 52 | 2 | 61.1% |
| NSU_NHT_CCR | 53 | 53 | 2 | 34.9% |
| PDA | 28 | 28 | 2 | 39.4% |
| PG | 34 | 34 | 2 | 45.9% |
| SEU1 | 19 | 19 | 2 | 58.7% |
| SEU2 | 14 | 14 | 2 | 47.7% |
| SEU3 | 13 | 13 | 2 | 39.2% |
| **all units** | 301 | | 24 | 46.7% |

## Graph and loops
- PC algorithm input: 24 PCs (flare_net not included); dropped before PC (collinear / dependent): 0.
- PC output: 75 edges = 72 directed, 3 undirected, 0 bidirected.
- DAG among PCs: 75 edges = 7 same-unit, 68 cross-unit. Then every PC -> flare_net added: 24 edges.
- Loops: 0 directed edges reversed to break cycles; 0 forced Dor-Tarsi steps (no consistent extension); final graph is acyclic (checked in the run).
- Parents per PC (among PCs): mean 3.12, max 11; root PCs (no parents): 7.

## T² detection and root causes
- Baseline: moving window (1250 rows, step 10). Limits x2: overall 74.69, per-PC 7.704 to 7.773.
- TP = flare event with T² above the limit at least once in the 60 min before onset; FN = without; FP = non-flare block (time between [onset - window, end] blocks) with T² above the limit at least once. TN not counted. F1 = 2TP / (2TP + FP + FN).

| metric | value |
|---|---|
| flare events (>= 30 min) | 620.0 |
| flare events not scored (no T² in the window) | 33.0 |
| flare events scored | 587.0 |
| TP (flare events with T² above the limit in the window) | 519.0 |
| FN (flare events without) | 68.0 |
| non-flare blocks | 270.0 |
| non-flare blocks not scored | 4.0 |
| FP (non-flare blocks with T² above the limit) | 250.0 |
| F1 = 2TP / (2TP + FP + FN) | 0.765 |
| TP: T² already above the limit when the window opened | 394.0 |
| median minutes from first T² above the limit to onset (TP) | 45.0 |

| event | onset | min | peak t/h | outcome | lead (min) | top root-cause PCs (T²_j / limit) |
|---|---|---|---|---|---|---|
| E1 | 07-01 00:00 | 3537 | 6.77 | not scored | - | - |
| E2 | 07-03 11:51 | 79 | 3.33 | not scored | - | - |
| E3 | 07-03 13:15 | 469 | 4.70 | not scored | - | - |
| E4 | 07-03 21:28 | 369 | 3.17 | not scored | - | - |
| E5 | 07-04 05:28 | 66 | 1.72 | TP | 55 | APS_NVPS_LEU_PC2 0.6; ISOM_PC1 0.4; APS_NVPS_LEU_PC1 0.3 |
| E6 | 07-04 08:30 | 592 | 2.62 | TP | 60 | SEU1_PC2 0.6; PG_PC1 0.3; DAK_PC1 0.3 |
| E7 | 07-05 01:01 | 43 | 1.91 | TP | 60 | LOUP_PC1 2.7; APS_NVPS_LEU_PC1 2.1; DIU_PC2 0.7 |
| E8 | 07-05 09:47 | 63 | 2.02 | FN | - | SEU2_PC2 0.5; SEU1_PC2 0.5; PG_PC2 0.4 |
| E9 | 07-05 10:59 | 269 | 2.28 | TP | 9 | SEU1_PC1 5.6; SEU2_PC1 1.7; DIU_PC2 0.9 |
| E10 | 07-06 13:59 | 49 | 1.82 | TP | 10 | SEU2_PC1 12.4; DAK_PC1 1.3; LOUP_PC2 1.3 |
| E11 | 07-06 15:38 | 73 | 1.74 | TP | 50 | APS_NVPS_LEU_PC1 6.0; SEU1_PC1 4.1; LOUP_PC2 2.4 |
| E12 | 07-06 18:40 | 40 | 2.34 | TP | 59 | APS_NVPS_LEU_PC1 2.8; ISOM_3_PC1 1.5; NSU_NHT_CCR_PC2 1.0 |
| E13 | 07-07 07:10 | 121 | 6.01 | TP | 59 | PDA_PC1 1.4; APS_NVPS_LEU_PC1 1.2; PG_PC2 1.2 |
| E14 | 07-07 10:33 | 206 | 2.65 | TP | 24 | SEU1_PC2 1.0; NSU_NHT_CCR_PC2 0.8; SEU1_PC1 0.8 |
| E15 | 07-08 10:47 | 38 | 2.83 | TP | 54 | SEU3_PC1 1.3; PDA_PC2 0.9; LOUP_PC1 0.7 |
| E16 | 07-08 14:50 | 62 | 1.89 | TP | 24 | SEU2_PC1 3.1; DAK_PC1 1.6; SEU3_PC2 1.4 |
| E17 | 07-08 16:15 | 41 | 1.68 | TP | 23 | SEU1_PC1 2.0; APS_NVPS_LEU_PC1 1.2; ISOM_PC1 1.2 |
| E18 | 07-08 17:30 | 61 | 2.15 | TP | 34 | ISOM_PC1 0.9; ISOM_3_PC1 0.9; SEU1_PC1 0.8 |
| E19 | 07-09 07:52 | 38 | 1.64 | TP | 54 | SEU2_PC2 2.0; SEU2_PC1 1.0; LOUP_PC1 0.8 |
| E20 | 07-09 15:32 | 63 | 2.76 | TP | 45 | ISOM_PC1 0.8; SEU1_PC2 0.7; APS_NVPS_LEU_PC1 0.5 |
| E21 | 07-09 16:54 | 183 | 4.21 | TP | 19 | APS_NVPS_LEU_PC1 4.6; DIU_PC1 2.5; PDA_PC2 1.9 |
| E22 | 07-10 09:12 | 58 | 5.51 | TP | 46 | LOUP_PC1 0.5; ISOM_PC2 0.4; DAK_PC1 0.4 |
| E23 | 07-10 10:18 | 45 | 1.66 | TP | 8 | LOUP_PC2 2.6; SEU2_PC2 2.1; DAK_PC1 1.7 |
| E24 | 07-10 11:52 | 68 | 1.82 | TP | 49 | SEU1_PC2 8.0; LOUP_PC1 7.5; LOUP_PC2 7.1 |
| E25 | 07-10 14:01 | 90 | 3.70 | TP | 60 | APS_NVPS_LEU_PC2 11.8; DAK_PC1 9.0; LOUP_PC2 3.5 |
| E26 | 07-10 18:14 | 34 | 4.24 | TP | 60 | DAK_PC1 22.9; SEU1_PC2 2.0; SEU2_PC1 1.2 |
| E27 | 07-10 19:02 | 33 | 2.41 | TP | 60 | DAK_PC1 2.6; LOUP_PC1 2.4; SEU3_PC1 1.7 |
| E28 | 07-10 20:22 | 132 | 4.78 | TP | 47 | SEU1_PC2 2.2; NSU_NHT_CCR_PC1 2.0; DAK_PC1 1.1 |
| E29 | 07-10 23:02 | 72 | 2.04 | TP | 28 | NSU_NHT_CCR_PC1 2.4; APS_NVPS_LEU_PC1 1.5; PDA_PC2 0.8 |
| E30 | 07-11 01:02 | 43 | 2.66 | TP | 48 | SEU1_PC2 2.4; NSU_NHT_CCR_PC1 2.0; APS_NVPS_LEU_PC1 1.7 |
| E31 | 07-11 06:58 | 32 | 2.97 | FN | - | PDA_PC1 1.3; DAK_PC1 1.1; ISOM_3_PC1 1.0 |
| E32 | 07-11 08:00 | 35 | 1.72 | FN | - | SEU3_PC1 0.8; DAK_PC1 0.8; APS_NVPS_LEU_PC1 0.7 |
| E33 | 07-11 11:23 | 82 | 2.26 | TP | 60 | PG_PC2 1.1; DIU_PC2 0.5; NSU_NHT_CCR_PC1 0.4 |
| E34 | 07-11 13:09 | 168 | 6.09 | TP | 24 | LOUP_PC2 2.0; NSU_NHT_CCR_PC2 1.7; APS_NVPS_LEU_PC2 1.5 |
| E35 | 07-11 16:17 | 208 | 6.48 | TP | 20 | LOUP_PC2 2.1; SEU3_PC1 1.9; PDA_PC2 1.6 |
| E36 | 07-11 21:23 | 97 | 4.35 | TP | 60 | PDA_PC2 3.3; DIU_PC1 3.2; ISOM_3_PC2 1.1 |
| E37 | 07-11 23:05 | 69 | 1.92 | TP | 5 | PG_PC1 5.9; DAK_PC1 2.3; SEU2_PC1 1.8 |
| E38 | 07-12 02:12 | 54 | 1.94 | TP | 47 | PG_PC1 1.0; SEU1_PC2 0.6; PG_PC2 0.5 |
| E39 | 07-12 03:49 | 85 | 1.87 | TP | 24 | DAK_PC1 1.1; ISOM_3_PC1 0.6; SEU1_PC2 0.3 |
| E40 | 07-12 06:05 | 47 | 1.91 | TP | 49 | DAK_PC1 3.7; SEU2_PC1 0.9; NSU_NHT_CCR_PC1 0.7 |

580 more events in `tables/event_outcomes.csv` and `RCA_report.xlsx`.

- TA cross-check: not applicable, the TA's period [2025-05-01, 2025-06-01) is outside the time window.
