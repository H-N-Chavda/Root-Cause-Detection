# SUMMARY: PCA (26 PCs) + flare tags -> PC algorithm -> causation-based T²

Rerun everything: `python pca_pc_t2/run_all.py --force` (paths in `pca_pc_t2/.env`; without `--force` finished stages are skipped). Generated 2026-10-26. Time window: 2025-05-01 00:00:00 to 2025-06-01 00:00:00 (end excluded). Method: Li, Jin & Shi (2008), J. Quality Technology 40(1).

## Tag and node counts per stage
| stage | count | flare tags | note |
|---|---|---|---|
| loaded (unique tags) | 298 | 4 | 0 duplicate copies skipped |
| S1 combined matrix (tags) | 273 | 3 | 270 process; dropped 25: {'constant': 10, 'missing': 8, 'near-constant': 6, 'reference': 1}; 44640 rows, 0 days 00:01:00 grid; prompt expected about 376 |
| S2 PCA input (process tags) | 270 | 0 | 26 PCs keep 80.09% of the variance |
| S3 PC nodes | 29 | 3 | 26 PCs; dropped 0 (collinear / dependent) |
| S5 T² nodes | 29 | 3 | left out (zero reference variance): none |

## PCA
- Fit on 44640 rows (full cleaned matrix); 26 components keep 80.09% of the variance.
- Variance per PC (%): PC1 19.6, PC2 12.4, PC3 9.2, PC4 6.0, PC5 4.8, PC6 3.3, PC7 3.2, PC8 2.2, PC9 2.2, PC10 1.9 ...
- Dominant unit per PC: PC1 LOUP (27%); PC2 LOUP (29%); PC3 PG (33%); PC4 DIU (22%); PC5 LOUP (29%); PC6 ISOM (22%); PC7 LOUP (57%); PC8 ISOM_3 (37%); PC9 ISOM (42%); PC10 ISOM (36%) ...

## Graph
- Alpha sweep: alpha 0.01: 50 edges (PC-PC 0, PC-flare 50); alpha 0.05: 55 edges (PC-PC 0, PC-flare 55). Used downstream: alpha 0.01.
- PC: depth cap 3, 4464 rows (every 10th), 36308 CI tests, 2 s.
- Edges: 50 = 50 directed, 0 undirected, 0 bidirected; PC-PC 0, PC-flare 50.
- PCs are nearly uncorrelated with each other (PCA output), so PC-PC edges are expected to be sparse and most structure to be PC -> flare. Observed: 0 PC-PC vs 50 PC-flare edges. This is not a bug.
- Flare edges straight from PC: {'directed_out_of_flare': 0, 'undirected_or_bidirected_touching_flare': 0, 'flare_flare': 0}; changed after PC: 0.
- DAG: 50 edges; CPDAG directed kept 50/50, reversed 0, forced Dor-Tarsi steps 0; roots 26, parents mean 1.72 / max 19.

## T² decomposition (Eqs. 7, 9-11, 13)
- Limits: T² > chi²_29(1 - 0.05) = 42.56; |T_j·PA(j)| > z = 3.134 (Bonferroni alpha' = 0.05/29).
- T² out of control: all 46.42%, reference 12.99%, flare rows 46.07%, non-flare rows 98.66%.
- Reference check: max |mean| 2.7e-10, max |std - 1| 8.4e-10, median excess kurtosis 0.76, mean % beyond z 0.610 (N(0,1) expects 0.172). Sample: PC1 -0.00/1.00; PC4 +0.00/1.00; PC7 -0.00/1.00; PC10 -0.00/1.00; PC13 -0.00/1.00; PC16 -0.00/1.00; PC19 +0.00/1.00; PC22 +0.00/1.00; PC25 -0.00/1.00; 509mw5101.daca.pv -0.00/1.00 (mean/std).
- Proposition 1 relative discrepancy |T² - Σ T²_j·PA(j)| / T² (median / p90): all 0.286 / 0.961; reference 0.130 / 0.358; T2_ooc 0.923 / 0.978; flare 0.282 / 0.960.

## Top findings
- Most often significant when T² alarms: PC1 (PC (LOUP), 16.9%; tags: 42PI4304.PV (SEU3, +0.11); 71TI8403.PV (DIU, +0.11); 71TI0405.PV (DIU, +0.11); 99TI3269.DACA.PV (LOUP, +0.10); 99TI2366A.DACA.PV (LOUP, +0.10)); (LRflare.pv+LREflare.pv+99fi4601.daca.pv)/1000 (flare_data, 13.4%; tags: -); 509mw5101.daca.pv (flare_data, 10.6%; tags: -); 509fi5201.daca.pv/1000 (flare_data, 9.0%; tags: -); PC14 (PC (SEU3), 7.1%; tags: 42PC4215.PV (SEU3, +0.31); 42PC4501.PV (SEU3, +0.31); 99PC1527.DACA.PV (LOUP, +0.23); 42PI4205.PV (SEU3, +0.22); 42PC4502.PV (SEU3, +0.21)).
- Flagged PCs (significant at the first T² alarm of a flare event): PC13, PC7; their top-loaded tags are in `tables/flagged_pc_tags.csv`.
- Flare event 1 (first T² alarm 2025-05-01): PC13 (PC (LOUP)), PC7 (PC (LOUP)), PC5 (PC (LOUP)).

## Runtime (s)
s1_load_clean 7 | s2_pca 1 | s3_pc 5 | s4_dag 0 | s5_t2 2 | s6_figures 7

## Assumptions
- [s1_load_clean] time window read: 2025-05-01 00:00:00 to 2025-06-01 00:00:00 (end excluded) (WINDOW_START / WINDOW_END in .env; empty = all data); flare thresholds, reference period and all limits are computed inside this window only
- [s1_load_clean] duplicate tag names (spaces and case ignored) keep the first copy in folder order flare_data > DAK > DIU > ISOM > ISOM_3 > LOUP > PDA > PG > SEU1 > SEU2 > SEU3; flare folder first so a flare tag is always marked is_flare
- [s1_load_clean] all tags resampled to the most common file sampling interval 0 days 00:01:00 (bin mean), as in pca_pc_t2
- [s1_load_clean] flaring = any flare tag > median + 3 * MAD (MAD x 1.4826), flags <= 5 samples apart merged into one event (pca_pc_t2 s0_audit rule)
- [s1_load_clean] reference = rows before 2025-05-19 14:23:24 (first 60% of time), flare windows NOT excluded
- [s1_load_clean] combined matrix has 273 tags = 270 process + 3 flare (prompt expected about 376: all tags -103, process tags only -106)
- [s2_pca] PCA fit rows = the full cleaned matrix (44640 rows); each process tag standardized over these rows (zero mean, unit variance)
- [s2_pca] PC scores standardized to unit variance (each divided by its std over the fit rows) before Stage 3
- [s3_pc] flare tags enter PC as standardized in Stage 1 (reference mean / std); PC scores as standardized in Stage 2
- [s3_pc] PC runs on every 10th row of the node matrix (4464 rows): consecutive samples are strongly autocorrelated, and Fisher-Z uses the data only through the correlation matrix and n
- [s3_pc] PC depth cap max_k = 3: largest depth <= 3 whose CI-test upper bound (8.17e+04) fits the budget 3e+07
- [s3_pc] PC: causal-learn pc, Fisher-Z, stable = True, uc_rule = 0, uc_priority = 2, alphas [0.01, 0.05]; flare tags in tier 1 (no edge into tier 0, none within tier 1, so flare-flare edges are forbidden too)
- [s3_pc] downstream alpha = 0.01: the stricter level of the sweep, as in causal_t2; with thousands of rows Fisher-Z rejects independence for very small correlations, so the stricter level keeps the graph sparser
- [s4_dag] DAG = Dor-Tarsi consistent extension of the PC CPDAG; flare tags removed first (all edges into flares); bidirected edges treated as undirected; ties broken by the matrix column order
- [s5_t2] reference period for Σ = Stage 1 'reference' rows (the repo's normal-operation rule)
- [s5_t2] Σ = correlation matrix of the 26784 reference rows; tags re-standardized with reference mean and std so that Σ is exactly their covariance
- [s5_t2] per flare event: first out-of-control T² from 0 days 06:00:00 before onset to the event end; nodes ranked by the share of significant |T_j·PA(j)| (then mean |T|) over the 0 days 00:30:00 after it
- [s5_t2] flagged node = |T_j·PA(j)| > z at the first T² alarm of a flare event (paper Fig. 5 reading); each flagged PC is listed with its top 15 tags by |loading| (tables/flagged_pc_tags.csv)

## Deviations
- [s1_load_clean] flare-free reference had < 1370 rows (5 x tags); fell back to: reference = rows before 2025-05-19 14:23:24 (first 60% of time), flare windows NOT excluded

## Limitations
- PCA is linear and unsupervised: the 26 PCs keep 80.1% of the variance, not flare relevance, so a PC can matter for flaring while explaining little variance.
- PC treats rows as independent; the 1-min data are autocorrelated (method kept as requested).
- The learned graph is not guaranteed correct. The paper assumes the BN is free of learning and sampling errors and Σ known (Section 'Causation-Based T² Decomposition'); the discrepancy above measures how far that fails.
- Σ is estimated on 26784 reference rows (reference = rows before 2025-05-19 14:23:24 (first 60% of time), flare windows NOT excluded).
- The flare label rule (reused) marks 99.3% of rows as flaring in 1 events.

## Definition of Done
| check | result | detail |
|---|---|---|
| No hardcoded path in code (paths only in .env); DATA_ROOT never written to outputs | PASS | code hits: none; output files with DATA_ROOT: none |
| No git command in the pipeline code | PASS | files: none |
| Matrix code from causal_t2; no NaN/inf, no constant column; tag count matches tag_registry.csv | PASS | 44640 rows x 273 tags (270 process + 3 flare); registry rows in matrix: 273 |
| PCA: exactly 26 components; variance reported; loadings tables exist; scores have no NaN | PASS | 26 PCs, cumulative variance 80.09% |
| PC graph nodes = 26 PCs + flare tags; 0 directed edges out of a flare, 0 undirected touching one; DAG acyclic | PASS | nodes 29 (expected 29 after 0 collinear drops); PC out-of-flare 0, undirected at flare 0; DAG out-of-flare 0, acyclic True |
| Reference T_j·PA(j) about N(0,1); discrepancy computed; every node has a chart | PASS | max |mean| 2.7e-10, max |std-1| 8.4e-10; charts 29 for 29 nodes |
| figures_index.csv, SUMMARY.md and LOG.md exist and agree | PASS | index rows 47 (all files exist: True); registry 'used' 3 = flare nodes in T² 3; LOG.md PASS rows for 6/6 stages |
| One command reruns the whole pipeline (documented at the top) | PASS | python pca_pc_t2/run_all.py --force |
