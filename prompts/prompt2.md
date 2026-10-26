# program.md: PCA (26 PCs) + Flare Tags -> PC Algorithm -> Causation-Based T² Decomposition

You are an autonomous researcher/engineer working inside this repo. Run the whole pipeline below
end-to-end without asking me anything. You run every command yourself in the terminal, read the
output, fix errors, and rerun. Do not stop until the Definition of Done (Section 7) is met.

## Reference files you have been given
- `T_2_Decomposition.pdf`: Li, Jin & Shi (2008), *Causation-Based T² Decomposition for Multivariate
  Process Monitoring and Diagnosis*, J. Quality Technology 40(1), 46-58. This is the method spec for Stage 5.
  Read it fully before writing any T² code. Cite its equation numbers in code comments and in `SUMMARY.md`.
- A screenshot of the data root listing (folder names, see Section 2).
- The folder `causal_t2/` in this repo. It holds working code from an earlier attempt: data loading and cleaning,
  the combined-matrix builder, and a T² decomposition implementation. Files and functions are well named, so grep
  for what you need and reuse it.

## 0. Principles (follow strictly)
1. Think before coding: read the repo (especially `causal_t2/`) and the paper first. Write a short plan (about 10 lines)
   at the top of `outputs/LOG.md`, then execute.
2. Simplicity first: the smallest code that does the job. No new frameworks, no speculative abstractions.
3. Surgical changes: reuse existing code from `causal_t2/`. Edit only what you must. Do not refactor or tidy old code.
   If you must change reused logic, copy it into a new file under `src/` and edit the copy.
4. Goal-driven: every stage has a verifiable check (Section 7). A stage is not done until its check passes.

## 1. Hard constraints
- **Never run any git command** (no status, add, commit, diff, log, checkout, nothing). Plain file reads/writes only.
- **Paths live only in `.env`** (load with python-dotenv). No absolute or hardcoded path anywhere else
  (code, logs, docs, figure titles). Reuse existing `.env` keys; add new keys there.
- Raw data is read-only. All new artifacts go under `outputs/`.
- **Processing and matrix formation: reuse, don't reinvent.** Missing-value handling, cleaning and combined-matrix
  construction must come from the existing code in `causal_t2/`. Log which file/function you reused.
  Expected result: a matrix with about 376 tags. Verify the actual count and log it.
- **Flare tags (folder `flare_data`) can have no outgoing edges.** A flare cannot cause any process
  variable. Enforce it inside PC (forbidden edges flare -> X via the library's background-knowledge
  mechanism, e.g. causal-learn `BackgroundKnowledge`) and verify it afterwards (Section 7).
  Because this forbids every edge leaving a flare tag, edges between two flare tags are also forbidden;
  log that you applied this.
- Libraries are allowed for PCA (scikit-learn) and PC (causal-learn preferred, Fisher-Z test).
  Reuse the T² implementation from `causal_t2/`, checked against the paper.

## 2. Data
Data root and subfolder names come from `.env`. Folders seen in the screenshot (inspect them; don't assume which hold parquet):
`All_unit_tags, APS_NVPS_LEU, DAK, DIU, flare_data, ISOM, ISOM_3, ISOM_4, LOUP, NSU_NHT_CCR, PDA, PG, SEU1, SEU2, SEU3, unit_wise_control_data`.

Each parquet file is one tag. After processing there should be about 376 tags in total. Some are flare tags
(from `flare_data`), the rest are process tags. Check how many flare tags there are and whether the 376
includes them. Keep `outputs/tag_registry.csv` (tag, source folder/unit, is_flare). Deduplicate tags by name
and log every decision.

## 3. How you run things (this replaces any manual iteration)
Write each stage as a script and run it yourself, saving all terminal output to a log file:

```
python src/<stage_script>.py > outputs/logs/<stage>.log 2>&1
```

Then read the log (`tail -n 80 outputs/logs/<stage>.log`). If it errored, the traceback is in the log:
form a hypothesis, make the smallest fix, rerun the same command. Append each iteration to `outputs/LOG.md`
(stage, error, fix, result). If a command is slow, run it in the background, poll the log, and meanwhile write the next stage's script.

Rules for the loop:
- Never ask me for input or confirmation. If something is ambiguous, pick the most defensible option, log it, move on.
- A failed check (Section 7) counts as an error. Debug it the same way.
- If the same error survives 5 distinct fix attempts, take the simplest documented fallback (cheaper CI test,
  lower depth cap, subsample rows, drop offending columns), record it in `SUMMARY.md` under "Deviations", and continue.
- Cache intermediate outputs so reruns skip finished stages.

## 4. Pipeline

**Stage 1: Load, clean, build the matrix (reuse from `causal_t2/`).**
Produce the combined, aligned, cleaned matrix of all tags (about 376). Split it into two blocks:
`X_process` (all non-flare tags) and `X_flare` (flare tags). Save both with the registry.

**Stage 2: PCA on the process tags.**
- Standardize `X_process` (zero mean, unit variance per tag), then run PCA with exactly **26 components**.
- Fit PCA on `X_process` only. Flare tags do not enter PCA. Log the choice of fit rows (full cleaned matrix by default).
- Save: PC scores (rows x 26), loadings (26 x tags), explained variance ratio per PC and cumulative.
- Make loadings interpretable, since later results must be traced back to plant tags: for each PC, save a table of the
  top 15 tags by |loading| with their unit, and the share of squared loading per unit.
- Rename PC columns `PC1 ... PC26`. Standardize the scores to unit variance before Stage 3 (log it).

**Stage 3: PC algorithm on [26 PCs + flare tags].**
- Build the node matrix: the 26 PC scores plus all flare tags (standardized). Node count is 26 + number of flare tags.
- Run PC (Fisher-Z) with the flare constraint (Section 1). Handle singular or near-singular correlation matrices
  (drop one of each collinear pair and log it). With so few nodes, runtime should be small, so a sensible alpha sweep
  (e.g. 0.01, 0.05) with edge counts reported is fine. Pick one alpha for downstream use and log why.
- Expect PCs to be nearly uncorrelated with each other (PCA output), so PC-PC edges are likely sparse and most structure
  will be PC -> flare. This is not a bug, but report it explicitly in `SUMMARY.md`.
- Save the adjacency matrix, an edge list, and GraphML.

**Stage 4: CPDAG to DAG.**
The T² decomposition needs a parent set for every node, so convert the PC output to a DAG (a consistent extension,
e.g. Dor-Tarsi), keeping the flare constraint (any undirected edge touching a flare tag is oriented into the flare tag).
Assert it is acyclic.

**Stage 5: Causation-based T² decomposition (the paper), on the same nodes.**
Implement Definition 2 / Eq. 13 of the paper with the Stage 4 DAG as the Bayesian network. Start from the existing
implementation in `causal_t2/` and verify it against the paper's equations.
- Nodes are the 26 PCs plus the flare tags. p = number of nodes.
- Use the correlation matrix Σ from an in-control reference period (use the repo's definition of normal operation
  if it has one, otherwise flare-free periods; log the choice). Overall T² = ZᵀΣ⁻¹Z (Eq. 7), limit χ²_p.
- For each node j, with parents PA(j) from the DAG: T_{j·PA(j)} = (Z_j − Σ_k β_k Z_pa_k) / σ_{j·PA(j)}.
  Eqs. 9-11 give the form, with β from regressing Z_j on its parents using Σ and σ² = 1 − Σ_k β_k ρ(Z_j, pa_k).
  A node with no parents gets T_j = Z_j.
- Per-node significance limit: ±z_{α/2} with Bonferroni α = α₀/p, as in the paper's simulation setup.
- The identity T² = Σ_j T²_{j·PA(j)} (Proposition 1) holds exactly only for a correct graph. Do not assert it.
  Report the relative discrepancy per timestamp as a diagnostic of graph quality.
- Root-cause reading (paper, Section "Monitoring and Diagnosis"): a significant T_{j·PA(j)} means a mean shift
  originating at node j, as opposed to one propagated from its parents. Flag these nodes.
- Trace back to plant tags: for each flagged PC, report its top-loaded tags (from Stage 2) so the result reads in
  terms of real instruments and units. Do this as tables, not as extra theory.

**Stage 6: Figures and tables for a later PPT (do NOT build the PPT; its instructions will come later).**
- PCA: scree plot (explained variance per PC and cumulative), bar chart of unit composition per PC, loading
  plots for the top PCs.
- PC graph: node-link diagram of the final DAG (PCs in one colour, flare tags in another, edges directed),
  degree summary.
- One T_{j·PA(j)} chart per node (each of the 26 PCs and each flare tag): time series, significance limits,
  flare events shaded. Save as PNG in `outputs/figures/per_node/`, plus a contact-sheet grid of all of them.
- Overall T² chart with the χ² limit and flare events marked.
- Heatmap of |T_{j·PA(j)}| (nodes x time).
- Top root-cause nodes per flare event (bar charts and a CSV table), with the top-loaded tags for each flagged PC.
- `outputs/figures_index.csv` (filename, what it shows, key numbers) and `outputs/SUMMARY.md` (tag counts per stage,
  PCA variance retained, edge counts, runtime, discrepancy diagnostic, top findings, all assumptions).
- One consistent, readable style (large fonts, labelled axes, no clipped text) so figures drop straight into slides.

## 5. Notes on the data and method
- PCA is linear and unsupervised. The 26 PCs capture variance, not flare relevance, so a PC can matter for flaring
  while explaining little variance. Report the cumulative variance retained by 26 PCs and state this under "Limitations".
- PC ignores autocorrelation in time-series data. That is what was requested, so do not change the method, but state it
  under "Limitations" in `SUMMARY.md`.
- The learned graph is not guaranteed correct (the paper assumes it is; see its assumptions on a learning-error-free BN).
  Say so in `SUMMARY.md`.

## 6. Output layout
```
outputs/
  LOG.md  SUMMARY.md  figures_index.csv  tag_registry.csv
  logs/                  (one log per stage)
  data/                  (combined matrix, X_process, X_flare, PC scores, loadings)
  graph/                 (adjacency, edge list, GraphML, DAG)
  t2/                    (per-node T_{j·PA(j)}, overall T², flags)
  figures/  per_node/  ...
```

## 7. Definition of Done
- [ ] No hardcoded paths outside `.env`; no git command was run.
- [ ] Processing and matrix code came from `causal_t2/`; the tag count is logged and matches `tag_registry.csv`;
      the matrix has no NaN/inf and no constant columns.
- [ ] PCA has exactly 26 components; variance explained is reported; loadings tables exist; PC scores have no NaN.
- [ ] PC graph nodes = 26 PCs + flare tags. 0 directed edges out of any flare tag and 0 undirected edges touching
      a flare tag. The DAG is acyclic.
- [ ] T² terms on the reference period are about N(0,1) (report mean/std for a sample of nodes); the discrepancy
      diagnostic is computed; every node has a chart.
- [ ] `figures_index.csv`, `SUMMARY.md` and `LOG.md` exist and agree with each other.
- [ ] One command reruns the whole pipeline from `.env`; it is documented at the top of `SUMMARY.md`.
