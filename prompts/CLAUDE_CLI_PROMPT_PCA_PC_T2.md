# Task: PCA → PC Algorithm → T² Decomposition on Plant Data (BTP – Causal AI)

> Paste this whole file into Claude CLI (or save it as `PROMPT.md` and run: `claude "Follow PROMPT.md exactly"`).
> Structure follows the Karpathy-style coding guidelines: **Think Before Coding · Simplicity First · Surgical Changes · Goal-Driven Execution**.

---

## 0. Behavioral Guidelines (read first, apply throughout)

### 0.1 Think Before Coding
- Do **not** assume. If something is ambiguous (timestamp column, units, which folders are "units"), **inspect the data and state your assumption** in `outputs/00_data_audit.md` before proceeding.
- If a step's result looks wrong (see the pre-flight checks in §3.3), **stop and report** instead of silently working around it.
- Surface trade-offs. Don't pick silently between alternatives that change scientific meaning.

### 0.2 Simplicity First
- Minimum code that solves the stated task. No frameworks, no config systems, no abstract base classes.
- Plain Python scripts, one per stage, in `src/`. Shared helpers only in `src/common.py`.
- Use standard libraries: `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `networkx`, `causal-learn`.

### 0.3 Surgical Changes
- Create only the files listed in §2. Do not modify or delete anything in the raw data folders. **Raw data is read-only.**
- If a stage fails, fix that stage only; don't rewrite the others.

### 0.4 Goal-Driven Execution
- Each stage has an explicit **success criterion** and a **verification step**. Run the verification, print the result, and only then move on.
- Every stage saves its artifacts to disk so later stages can be rerun independently.
- Fix seed: `SEED = 42` everywhere.

---

## 1. Context

**Project:** Causal discovery for root-cause attribution of flaring events in a process plant.
**Data (sent by TA):** Folders of parquet files. Set `DATA_ROOT = <path to the data folder>` (ask me if you can't find it).

Visible top-level folders:
`All_unit_tags, APS_NVPS_LEU, DAK, DIU, flare_data, ISOM, ISOM_3, ISOM_4, LOUP, NSU_NHT_CCR, PDA, PG, SEU1, SEU2, SEU3, unit_wise_control_data`

**Expected structure (verify, don't trust):**
- **13 process units** = `APS_NVPS_LEU, DAK, DIU, ISOM, ISOM_3, ISOM_4, LOUP, NSU_NHT_CCR, PDA, PG, SEU1, SEU2, SEU3`
- **418 parameters (tags/columns)** in total across these 13 units
- `All_unit_tags` → probably a tag master list/metadata (use it for tag→unit mapping and descriptions if it contains that)
- `flare_data` → probably flare event/flow data (candidate **ground-truth fault labels** for T² validation)
- `unit_wise_control_data` → control-loop data; **exclude from the 418 unless the audit shows it is part of them**

If the audit contradicts these expectations (e.g., parameter count ≠ 418), report the discrepancy and proceed with what the data supports.

**Out of scope (do NOT do these now):** PCMCI+, VAR-LiNGAM, Lag-Specific Transfer Entropy, any slides/PPT generation. Figures only.

---

## 2. Output Layout

```
project/
├── src/
│   ├── common.py              # paths, seed, plotting style, helpers
│   ├── s0_audit.py
│   ├── s1_build_matrix.py
│   ├── s2_pca.py
│   ├── s3_pc_algorithm.py
│   ├── s4_t2_analysis.py
│   └── s5_figures.py          # only if figures aren't produced in-stage
├── outputs/
│   ├── 00_data_audit.md
│   ├── data/                  # intermediate .parquet / .npz / .csv
│   ├── figures/               # PNG (300 dpi) + PDF vector copy of each
│   └── tables/                # CSV of every number plotted
├── RESULTS.md                 # final summary (see §8)
└── run_all.sh                 # runs s0→s5 in order
```

---

## 3. Pipeline

### Stage 0 — Data Audit (`s0_audit.py`)
1. Walk every folder; list parquet files, row counts, column names, dtypes, time range, sampling interval.
2. Identify the **timestamp column** and confirm it's consistent across units.
3. Build a **tag → unit** mapping (from `All_unit_tags` if usable, otherwise from folder membership).
4. Report per-unit: #parameters, % missing, constant/near-constant columns, duplicated timestamps, obvious outliers/sentinel values (e.g., -999, 0-floods).
5. Inspect `flare_data` and describe what it contains and whether it gives **flare event windows** (start/end) or a continuous flare signal. Define the fault label from it **only if it is unambiguous**; otherwise state the options and use the most defensible one.

**Success criterion:** `00_data_audit.md` states the parameter count, time range, sampling interval, and the tag→unit mapping saved to `outputs/data/tag_unit_map.csv`.

### Stage 1 — Build the Master Matrix (`s1_build_matrix.py`)
1. Concatenate files within each unit chronologically; outer-join all 13 units on timestamp. Resample to the dominant common sampling interval (state which).
2. Handle missing data: forward-fill short gaps (≤ N samples, choose N and justify), drop columns with > 30 % missing or zero variance (log every dropped tag), drop remaining incomplete rows. Report the final shape.
3. **Split by time, not randomly:**
   - `TRAIN` = normal-operation period (exclude flare event windows ± a buffer, if labels exist)
   - `TEST` = remaining data including flaring events
   If no reliable labels exist, use the first 60 % of time as TRAIN and say so.
4. **Standardize (z-score) using TRAIN statistics only**; apply the same mean/std to TEST. PCA on unscaled mixed-unit tags (°C, bar, etc.) is meaningless — never skip this.

**Success criterion:** `X_train`, `X_test` saved; shape printed; no NaN/inf; column count reported vs the expected 418.

### Stage 2 — PCA (`s2_pca.py`)
1. Fit `sklearn.decomposition.PCA(n_components=26)` on `X_train` (user-specified k = 26). Also fit a full PCA to get the complete spectrum for plots.
2. Save: eigenvalues λ, explained-variance ratio, cumulative variance, loadings matrix `P` (variables × 26), scores `T_train`, `T_test`.
3. Report cumulative variance at 26 PCs, and the PC count needed for 80/90/95/99 %.
4. Compute **per-unit loading mass** for each PC (sum of squared loadings of that unit's tags) → table showing which units dominate which PCs.

**Verification:** reconstruct `X_train` from 26 PCs; print reconstruction R². Confirm `np.allclose(P.T @ P, I)`.

### 3.3 Pre-flight check before Stage 3 (IMPORTANT — do not skip)
PCA scores on the fitting data are **uncorrelated by construction**. The PC algorithm with a Fisher-Z (partial-correlation) test will therefore see almost no dependence among `T_train` columns and is expected to return a **near-empty skeleton**.

1. Compute the correlation matrix of `T_train` and of `T_test`; print max |off-diagonal|.
2. Run the planned PC algorithm on `T_train` and report the number of edges.
3. **If the graph is empty/near-empty on `T_train`:** do NOT silently change the method. Report it, then run these in order and report each:
   - (a) PC on `T_test` scores (projected with TRAIN-fit PCA) — correlations there are non-zero when the process deviates from normal; also on **flare-window-only** scores;
   - (b) PC with a nonlinear CI test (KCI or RCIT from `causal-learn`; subsample if too slow, state the subsample size);
   - (c) PC on **time-lagged** PC scores (lags 0..τ) only as a diagnostic, clearly labelled.
4. State in `RESULTS.md` which variant produced the final graph and why.

### Stage 3 — PC Algorithm (`s3_pc_algorithm.py`)
1. Use `causallearn.search.ConstraintBased.PC.pc` with `alpha = 0.05` (also run 0.01 as sensitivity), `indep_test = fisherz` by default, `stable=True`, seed fixed.
2. Output the **CPDAG** (undirected + directed edges kept as-is; do not force-orient).
3. Save adjacency matrix, edge list (with edge type: →, —, ↔), and number of CI tests run.
4. **Stability:** bootstrap (B = 100, block bootstrap with block length ≈ autocorrelation time, since rows are time-ordered) → per-edge selection frequency. Keep edges with frequency ≥ 0.7 as the "stable graph".
5. **Map back to the physical world:** for every edge PCi → PCj, list the top-5 contributing tags (by |loading|) and dominant units of each PC, so edges can be read as *unit/tag-level* hypotheses. Save to `tables/pc_edge_interpretation.csv`.

**Success criterion:** graph + edge frequencies saved; every PC node has a unit-level label.

### Stage 4 — Hotelling T² (and companion Q) Decomposition (`s4_t2_analysis.py`)
Use the 26-PC model from Stage 2.

1. **Statistics per sample:**
   - `T² = Σ_i t_i² / λ_i`  (i = 1..26)
   - `Q (SPE) = ‖x − x̂‖²` (residual outside the 26 PCs) — report it too; T² alone ignores the residual subspace.
2. **Control limits (from TRAIN):**
   - T² UCL: `k(n²−1) / (n(n−k)) · F_α(k, n−k)` at 95 % and 99 %
   - Q UCL: Jackson–Mudholkar approximation (or empirical 99th percentile of TRAIN; state which)
3. **T² decomposition (two levels):**
   - **PC-wise:** contribution of each PC `c_i(t) = t_i(t)² / λ_i` (sums exactly to T²).
   - **Variable-wise:** per-tag contribution via the standard PCA contribution formula (Kourti–MacGregor / complete decomposition contributions); contributions can be negative — keep sign, then also report the clipped version. Aggregate to the **13 units**.
4. **Fault analysis:**
   - Flag samples with T² > UCL (and/or Q > UCL). Group into **events** (merge flags within a gap tolerance; state it).
   - For each event: start time, duration, peak T², top-3 PCs, top-10 tags, top-3 units, and — if `flare_data` gives events — the **lead/lag between T² alarm and flare onset**.
   - Detection metrics against flare labels (if available): detection rate, false alarm rate on TRAIN, detection delay. Do not invent labels.
5. **Link to Stage 3:** for each event, check whether the dominant contributing PCs are **ancestors** in the PC-algorithm graph of the PC that is most elevated at flare onset → candidate root-cause units. Report as *hypotheses*, not conclusions.
6. Verification: `np.allclose(sum of PC contributions, T²)` and variable contributions sum consistent with their definition.

---

## 4. Figures (the part for the professor)

All figures: 300 dpi PNG **and** vector PDF, consistent colour palette (colour-blind-safe), readable fonts (≥ 11 pt), titled, axes labelled with units, **one-line takeaway as the subtitle**, and a matching CSV in `tables/`. Name as `figNN_description.png`. Use a **fixed colour per unit** across all figures.

### A. Data & preprocessing
1. Parameters per unit (bar) and missing-data % per unit (bar/heatmap) — before vs after cleaning.
2. Time coverage of each unit (timeline/Gantt) with flare events shaded.
3. Correlation heatmap of the 418 tags, ordered by unit, with unit boundaries drawn (shows block structure motivating PCA).

### B. PCA
4. **Scree plot** (first ~60 eigenvalues) with a marker at PC 26.
5. **Cumulative explained variance** with 80/90/95/99 % lines and PC 26 marked.
6. **Loadings heatmap** (26 PCs × 13 units, using per-unit loading mass) — "which unit lives in which PC".
7. **Top-tag loading bar plots** for the first 4–6 PCs.
8. **Score plots** PC1 vs PC2 and PC1 vs PC3 (TRAIN vs TEST coloured; flare windows highlighted).
9. **PC time series** of the first 6 PCs with flare events shaded.
10. Correlation matrix of PC scores (TRAIN vs TEST) — supports the §3.3 diagnostic.

### C. PC algorithm
11. **Causal graph (CPDAG) among the 26 PCs** — node colour = dominant unit, node size = explained variance, edge width = bootstrap frequency, undirected edges drawn without arrowheads. Use a clean layout (`networkx` spring/Kamada-Kawai or layered), no overlapping labels.
12. **Edge stability histogram** (bootstrap frequencies) and a stable-edge-only graph.
13. **Unit-level collapsed graph:** edges aggregated from PC level to unit level via loading mass (13 nodes) — the figure most useful for plant discussion.
14. Sensitivity: edge count and Jaccard similarity of graphs for α = 0.05 vs 0.01 (and for each method variant from §3.3).

### D. T² / Q analysis
15. **T² monitoring chart** over time with 95 % / 99 % UCL, TRAIN/TEST boundary, flare events shaded.
16. **Q (SPE) chart** with UCL, same shading. Optionally a **T²–Q scatter** (quadrant plot) with limits.
17. **T² distribution** TRAIN vs TEST (histogram/ECDF) with UCL.
18. **PC contribution heatmap** (26 PCs × time) around each major event — shows which PCs drive the alarm.
19. **Contribution bar plot at the alarm instant / event peak:** top-15 tags (with unit colour) for each of the top ~5 events.
20. **Unit-level contribution heatmap over time** (13 units × time) with flare onsets marked.
21. **Event summary table figure** (start, duration, peak T², top PCs, top units, lead time to flare).
22. **Detection performance:** ROC (or detection-rate vs false-alarm-rate) if labels exist; **alarm-to-flare lead-time histogram**.
23. **Root-cause hypothesis figure:** for one illustrative event, show (i) T² spike, (ii) top contributing PCs, (iii) those PCs' positions in the causal graph, (iv) implied upstream units — a single multi-panel "storyline" figure.

If a figure is not meaningful with the actual data (e.g., no flare labels), skip it and **say why in `RESULTS.md`**, rather than producing a placeholder.

---

## 5. Engineering Rules
- Print progress per stage with elapsed time; log to `outputs/run.log`.
- No hard-coded absolute paths except `DATA_ROOT` in `common.py`.
- Vectorized `numpy`/`pandas`; avoid row loops over the full dataset.
- If a step takes > 10 min (e.g., KCI, bootstrap), subsample or reduce B and **state the reduction**.
- Never fabricate results, labels, or physical interpretations. Where tag descriptions are unavailable, label by tag ID only.

---

## 6. Order of Execution & Gates
Run `s0 → s1 → s2 → (§3.3 pre-flight) → s3 → s4 → s5`.
After **each** stage print: the success-criterion check, any assumption made, and any anomaly. If a gate fails, stop and ask me.

---

## 7. Definition of Done
- [ ] All stage scripts run end-to-end via `run_all.sh` from a clean state
- [ ] Parameter count and unit count reconciled against 418 / 13
- [ ] PCA: cumulative variance at 26 PCs reported; reconstruction verified
- [ ] PC-algorithm variant used is justified by the §3.3 pre-flight result
- [ ] T² PC-contributions sum to T² (verified numerically)
- [ ] Figures 1–23 produced or explicitly skipped with reasons
- [ ] Every figure has a CSV in `tables/`
- [ ] `RESULTS.md` written

## 8. `RESULTS.md` Contents (concise, factual)
1. Data summary (final shape, time range, split, dropped tags)
2. PCA results (variance table, dominant units per top PC)
3. PC-algorithm results (variant used and why, #edges, stable edges, key unit-level links)
4. T²/Q results (limits, #events, detection metrics, top contributing units per event)
5. Linked root-cause hypotheses (clearly labelled as hypotheses)
6. Limitations: PCA mixes variables (edges are between components, not tags); PC algorithm assumes i.i.d. samples while plant data is autocorrelated; linear CI test; no PFD ground truth
7. Figure index (number → one-line takeaway)
8. Suggested next step (one paragraph only — do not start it)
