# LOG: PCA (26 PCs) + flare tags -> PC algorithm -> causation-based T² (Li, Jin & Shi 2008)

## Plan (2026-10-26)
1. s1_load_clean: copy of `causal_t2/src/s1_load_clean.py` (load, dedupe, clean, z-score, `tag_registry.csv`) + split into `X_process` / `X_flare`.
2. s2_pca: standardize `X_process`, scikit-learn PCA with exactly 26 components; scores, loadings, variance, top-15 tags and unit shares per PC.
3. s3_pc: node matrix = 26 standardized PC scores + flare tags; copy of `causal_t2/src/s2_pc.py` (Fisher-Z, flare tiers, collinear drops, post-check).
4. s3: alpha sweep 0.01 / 0.05 with edge counts; 0.01 used downstream; adjacency, edge list, GraphML.
5. s4_dag: copy of `causal_t2/src/s3_dag.py` (Dor-Tarsi, flares as sinks, acyclic check) on the node graph.
6. s5_t2: copy of `causal_t2/src/s4_t2.py` (Eqs. 7, 9-11, 13; Bonferroni z; Prop. 1 discrepancy) on the nodes; trace flagged PCs to top-loaded tags.
7. s6_figures: scree, unit composition, loadings, DAG, degree summary, per-node charts + contact sheet, T², heatmap, per-event bars, `figures_index.csv`.
8. s7_summary: `SUMMARY.md` + Definition-of-Done checks (copy of `causal_t2/src/s6_summary.py`, edited).
9. Runner: `python pca_pc_t2/run_all.py` (copy of `causal_t2/run_all.py`); one log per stage in `outputs/logs/`.
10. Time window from `.env` (default May 2025); empty window = all data.

## Notes
- 2026-10-26: code written without running it (user rule: no code runs on the author's machine; the TA runs it). UNVERIFIED until the first lab run.
- No git command is part of the pipeline. Paths only in `.env`.
- Old pca_pc_t2 pipeline (s0-s5, run_all.sh) and all its outputs (RESULTS.md, audit, figures, tables, run.log) removed on request (2026-10-26).

## Runs (appended by run_all.py; add error / fix notes under a failed row)
| time | stage | result | seconds | log / last error |
|---|---|---|---|---|
