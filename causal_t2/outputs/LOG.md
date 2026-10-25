# LOG: causal PC + causation-based T² (Li, Jin & Shi 2008)

## Plan (2026-10-26)
1. s1_load_clean: tags from flare, unit, control and All_unit_tags folders; dedupe by tag name; `tag_registry.csv`.
2. s1: resample to the dominant interval; missing values as in `pca_pc_t2/src/s1_build_matrix.py` (sentinels, ffill <= 5, > 30 % missing, row >= 90 %, reference-mean impute).
3. s1: drop constant, near-constant (> 99 % one value), duplicate tags. Flare events and reference period = earlier repo rules.
4. s2_pc: causal-learn PC, Fisher-Z, stable; flare -> X forbidden via BackgroundKnowledge tiers; collinear pairs dropped; depth cap from a CI-test budget.
5. s2: post-check flare edges, orient leftovers X -> flare; save adjacency, edge list, GraphML.
6. s3_dag: Dor-Tarsi consistent extension, flares removed first as sinks; assert acyclic.
7. s4_t2: Σ from reference rows; T² (Eq. 7); T_j·PA(j) (Eqs. 9-11, 13); Bonferroni z limit; Prop. 1 discrepancy.
8. s5_figures: per-tag charts, T² chart, heatmap, per-event root causes, graph views, `figures_index.csv`.
9. s6_summary: `SUMMARY.md` and Definition-of-Done checks.
10. Runner: `python causal_t2/run_all.py`; each stage log in `outputs/logs/`.

## Notes
- 2026-10-26: code written without running it (user rule: no code runs on the author's machine). First real run is on the lab computer. UNVERIFIED until then.
- Reused: readers + `ffill_short_gaps` copied from `pca_pc_t2/src/common.py` and `pca_pc_t2/src/s1_build_matrix.py`.

## Runs (appended by run_all.py; add error / fix notes under a failed row)
| time | stage | result | seconds | log / last error |
|---|---|---|---|---|
