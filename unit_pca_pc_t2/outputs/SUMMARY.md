# SUMMARY: unit-wise PCA → PC algorithm → causation-based T²

Generated 2026-11-04 17:20 from `outputs/` by `summary.py`.

## Data and cleaning
| item | value |
|---|---|
| time window | 2025-07-01 00:00:00 to 2025-11-01 00:00:00 (end excluded) |
| sampling interval | 0 days 00:01:00 |
| tags loaded | 380 (4 flare tags) |
| process tags used in PCA | 301 |
| tags dropped | 75: negative values 49, too much missing 19, constant 5, near-constant 2 |
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
| **all units** | 301 | | 24 | 51.4% |

## Graph and loops
- PC algorithm input: 24 PCs (flare_net not included); dropped before PC (collinear / dependent): 0.
- PC output: 76 edges = 74 directed, 2 undirected, 0 bidirected.
- DAG among PCs: 76 edges = 8 same-unit, 68 cross-unit. Then every PC -> flare_net added: 24 edges.
- Loops: 3 directed edges reversed to break cycles; 0 forced Dor-Tarsi steps (no consistent extension); final graph is acyclic (checked in the run).
- Parents per PC (among PCs): mean 3.17, max 9; root PCs (no parents): 6.

## T² detection and root causes
- Overall T² (PCs only, no flare data) limit 36.51: report Eq. 13 (corrected), m = 12157 NOC rows, p = 24 PCs, k = 0, α = 0.05. Above it: 11.70% of NOC rows (expected about 5%), 78.06% of pre-flare rows.
- Flare events found (T² above its limit at least once in the 30 min before onset): 532 of 620.
- Per-PC limits on T²_j: Eq. 13 with p = 1, k = number of parents, α = 0.05: 3.843 to 3.845. NOC rows above: mean 4.97% per PC (expected about 5%).
- Overall T² = Zᵀ R⁻¹ Z with the sample NOC correlation R (as the TA). The causal terms sum to Zᵀ Σ_G⁻¹ Z (max relative error 3.2e-15); median gap between that sum and T² on NOC: 0.149.

| event | onset | min | peak t/h | found | first T² alarm (min before onset) | top root-cause PC | minutes above its limit |
|---|---|---|---|---|---|---|---|
| E1 | 01 Jul 00:00 | 3537 | 6.77 | no | - | none | - |
| E2 | 03 Jul 11:51 | 79 | 3.33 | yes | 30 | DAK_PC1 | 30 of 30 |
| E3 | 03 Jul 13:15 | 469 | 4.70 | yes | 5 | DIU_PC1 | 5 of 5 |
| E4 | 03 Jul 21:28 | 369 | 3.17 | yes | 24 | SEU1_PC2 | 24 of 24 |
| E5 | 04 Jul 05:28 | 66 | 1.72 | yes | 23 | ISOM_4_PC1 | 16 of 30 |
| E6 | 04 Jul 08:30 | 592 | 2.62 | yes | 28 | ISOM_PC1 | 13 of 30 |
| E7 | 05 Jul 01:01 | 43 | 1.91 | no | - | SEU3_PC1 | 18 of 30 |
| E8 | 05 Jul 09:47 | 63 | 2.02 | no | - | LOUP_PC1 | 19 of 30 |
| E9 | 05 Jul 10:59 | 269 | 2.28 | no | - | SEU1_PC2 | 9 of 9 |
| E10 | 06 Jul 13:59 | 49 | 1.82 | yes | 9 | NSU_NHT_CCR_PC1 | 30 of 30 |
| E11 | 06 Jul 15:38 | 73 | 1.74 | no | - | NSU_NHT_CCR_PC1 | 30 of 30 |
| E12 | 06 Jul 18:40 | 40 | 2.34 | no | - | none | - |
| E13 | 07 Jul 07:10 | 121 | 6.01 | yes | 27 | ISOM_4_PC1 | 30 of 30 |
| E14 | 07 Jul 10:33 | 206 | 2.65 | no | - | NSU_NHT_CCR_PC2 | 27 of 30 |
| E15 | 08 Jul 10:47 | 38 | 2.83 | yes | 9 | NSU_NHT_CCR_PC2 | 29 of 30 |
| E16 | 08 Jul 14:50 | 62 | 1.89 | yes | 30 | SEU1_PC2 | 30 of 30 |
| E17 | 08 Jul 16:15 | 41 | 1.68 | yes | 14 | SEU1_PC2 | 23 of 23 |
| E18 | 08 Jul 17:30 | 61 | 2.15 | yes | 4 | SEU1_PC2 | 25 of 30 |
| E19 | 09 Jul 07:52 | 38 | 1.64 | no | - | none | - |
| E20 | 09 Jul 15:32 | 63 | 2.76 | yes | 21 | APS_NVPS_LEU_PC2 | 20 of 30 |
| E21 | 09 Jul 16:54 | 183 | 4.21 | yes | 19 | SEU3_PC1 | 1 of 19 |
| E22 | 10 Jul 09:12 | 58 | 5.51 | no | - | NSU_NHT_CCR_PC1 | 14 of 30 |
| E23 | 10 Jul 10:18 | 45 | 1.66 | yes | 6 | PDA_PC1 | 5 of 8 |
| E24 | 10 Jul 11:52 | 68 | 1.82 | yes | 30 | SEU1_PC2 | 30 of 30 |
| E25 | 10 Jul 14:01 | 90 | 3.70 | yes | 30 | PDA_PC1 | 30 of 30 |
| E26 | 10 Jul 18:14 | 34 | 4.24 | yes | 30 | DIU_PC1 | 30 of 30 |
| E27 | 10 Jul 19:02 | 33 | 2.41 | yes | 7 | DIU_PC1 | 13 of 14 |
| E28 | 10 Jul 20:22 | 132 | 4.78 | yes | 30 | SEU3_PC1 | 24 of 30 |
| E29 | 10 Jul 23:02 | 72 | 2.04 | yes | 19 | PDA_PC1 | 28 of 28 |
| E30 | 11 Jul 01:02 | 43 | 2.66 | no | - | LOUP_PC1 | 30 of 30 |
| E31 | 11 Jul 06:58 | 32 | 2.97 | yes | 30 | LOUP_PC1 | 30 of 30 |
| E32 | 11 Jul 08:00 | 35 | 1.72 | yes | 30 | LOUP_PC1 | 30 of 30 |
| E33 | 11 Jul 11:23 | 82 | 2.26 | yes | 30 | PDA_PC1 | 30 of 30 |
| E34 | 11 Jul 13:09 | 168 | 6.09 | yes | 24 | NSU_NHT_CCR_PC1 | 24 of 24 |
| E35 | 11 Jul 16:17 | 208 | 6.48 | yes | 20 | LOUP_PC1 | 20 of 20 |
| E36 | 11 Jul 21:23 | 97 | 4.35 | yes | 30 | LOUP_PC1 | 30 of 30 |
| E37 | 11 Jul 23:05 | 69 | 1.92 | yes | 5 | LOUP_PC1 | 5 of 5 |
| E38 | 12 Jul 02:12 | 54 | 1.94 | yes | 30 | LOUP_PC1 | 30 of 30 |
| E39 | 12 Jul 03:49 | 85 | 1.87 | yes | 30 | LOUP_PC1 | 30 of 30 |
| E40 | 12 Jul 06:05 | 47 | 1.91 | yes | 30 | LOUP_PC1 | 30 of 30 |

580 more events in `tables/flare_events.csv` and `tables/root_cause_by_event.csv`.

- TA cross-check: not applicable, the TA's period [2025-05-01, 2025-06-01) is outside the time window.
