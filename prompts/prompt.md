# program.md: Causal PC + Causation-Based T² Decomposition on Plant Tag Data

You are an autonomous researcher/engineer working inside this repo. Run the whole pipeline below
end-to-end without asking me anything. You run every command yourself in the terminal, read the
output, fix errors, and rerun. Do not stop until the Definition of Done (Section 6) is met.

## Reference files you have been given
- `T_2_Decomposition.pdf`: Li, Jin & Shi (2008), *Causation-Based T² Decomposition for Multivariate
  Process Monitoring and Diagnosis*, J. Quality Technology 40(1), 46-58. This is the method spec for Stage 4.
  Read it fully before writing any T² code. Cite its equation numbers in code comments and in `SUMMARY.md`.
- A screenshot of the data root listing (folder names, see Section 2).
- The earlier attempt in this repo. Its files and folders are well named: grep for what you need and
  reuse snippets selectively.

## 0. Principles (follow strictly)
1. Think before coding: read the repo and the paper first. Write a short plan (about 10 lines) at the top of `outputs/LOG.md`, then execute.
2. Simplicity first: the smallest code that does the job. No new frameworks, no speculative abstractions.
3. Surgical changes: reuse existing repo code. Edit only what you must. Do not refactor or tidy old code.
4. Goal-driven: every stage has a verifiable check (Section 6). A stage is not done until its check passes.

## 1. Hard constraints
- **Never run any git command** (no status, add, commit, diff, log, checkout, nothing). Plain file reads/writes only.
- **Paths live only in `.env`** (load with python-dotenv). No absolute or hardcoded path anywhere else
  (code, logs, docs, figure titles). Reuse existing `.env` keys; add new keys there.
- Raw data is read-only. All new artifacts go under `outputs/`.
- **Missing values: do not invent a method.** Find how the repo already handles them (the earlier
  data-processing stage) and reuse it as-is. Log which file/function you reused.
- **Flare tags (folder `flare_data`) can have no outgoing edges.** A flare cannot cause any process
  variable. Enforce it inside PC (forbidden edges flare -> X via the library's background-knowledge
  mechanism, e.g. causal-learn `BackgroundKnowledge`) and also verify it afterwards (Section 6).
- Libraries are allowed for PC (causal-learn preferred, Fisher-Z test). Write the T² decomposition yourself with numpy.

## 2. Data
Data root and subfolder names come from `.env`. Folders seen in the screenshot (inspect them; don't assume which hold parquet):
`All_unit_tags, APS_NVPS_LEU, DAK, DIU, flare_data, ISOM, ISOM_3, ISOM_4, LOUP, NSU_NHT_CCR, PDA, PG, SEU1, SEU2, SEU3, unit_wise_control_data`.

Each parquet file is one tag (350+ tags in total). `All_unit_tags` and `unit_wise_control_data` may duplicate tags
from the unit folders: deduplicate by tag name and log every decision. Maintain `outputs/tag_registry.csv`
(tag, source folder/unit, is_flare).

## 3. How you run things (this replaces any manual iteration)
Write each stage as a script and run it yourself, saving all terminal output to a log file:

```
python src/<stage_script>.py > outputs/logs/<stage>.log 2>&1
```

Then read the log (`tail -n 80 outputs/logs/<stage>.log`). If it errored, the traceback is in the log:
form a hypothesis, make the smallest fix, rerun the same command. Append each iteration to `outputs/LOG.md`
(stage, error, fix, result). If a command is slow, run it in the background, poll the log, and meanwhile work on the next stage's script.

Rules for the loop:
- Never ask me for input or confirmation. If something is ambiguous, pick the most defensible option, log it, move on.
- A failed check (Section 6) counts as an error. Debug it the same way.
- If the same error survives 5 distinct fix attempts, take the simplest documented fallback (cheaper CI test,
  lower depth cap, subsample rows, drop offending tags), record it in `SUMMARY.md` under "Deviations", and continue.
- Cache intermediate outputs so reruns skip finished stages.

## 4. Pipeline

**Stage 1: Load and clean.** Load every tag, align on timestamp, apply the repo's missing-value handling,
drop constant, near-constant and duplicate columns (log each drop), and build one combined matrix. Save it under `outputs/`.

**Stage 2: PC algorithm.** Run PC on the standardized combined matrix with the flare constraint (Section 1).
Roughly 350 variables is large, so handle runtime (depth cap, alpha, parallelism) and singular or near-singular
correlation matrices (drop one of each collinear pair, log it). Save the adjacency matrix, an edge list, and GraphML.

**Stage 3: CPDAG to DAG.** The T² decomposition needs a parent set for every tag, so convert the PC output to a DAG
(a consistent extension, e.g. Dor-Tarsi), keeping the flare constraint. Assert it is acyclic.

**Stage 4: Causation-based T² decomposition (the paper).**
Implement Definition 2 / Eq. 13 of the paper, with the graph from Stage 3 as the Bayesian network:
- Standardize the tags. Use the correlation matrix Σ from an in-control reference period (use the repo's definition
  of normal operation if it has one, otherwise flare-free periods; log the choice). Overall T² = ZᵀΣ⁻¹Z (Eq. 7), limit χ²_p.
- For each tag j, with parents PA(j) from the DAG: T_{j·PA(j)} = (Z_j − Σ_k β_k Z_pa_k) / σ_{j·PA(j)}.
  Eqs. 9-11 give the form, with β from regressing Z_j on its parents using Σ and σ² = 1 − Σ_k β_k ρ(Z_j, pa_k).
  A tag with no parents gets T_j = Z_j.
- Per-tag significance limit: ±z_{α/2} with Bonferroni α = α₀/p, as in the paper's simulation setup.
- The identity T² = Σ_j T²_{j·PA(j)} (Proposition 1) holds exactly only for a correct graph. Do not assert it.
  Report the relative discrepancy per timestamp as a diagnostic of graph quality.
- Root-cause reading (paper, Section "Monitoring and Diagnosis"): a significant T_{j·PA(j)} means a mean shift
  originating at tag j, as opposed to one propagated from its parents. Flag these tags.

**Stage 5: Figures and tables for a later PPT (do NOT build the PPT; its instructions will come later).**
- One T_{j·PA(j)} chart per tag: time series, significance limits, flare events shaded.
  Save as PNG in `outputs/figures/per_tag/`.
- Overall T² chart with the χ² limit and flare events marked.
- Heatmap of |T_{j·PA(j)}| (tags × time), tags grouped by unit.
- Top root-cause tags per flare event (bar charts and a CSV table).
- PC graph views: full graph coloured by unit, degree distribution, and zoomed subgraphs around the top root-cause tags.
- `outputs/figures_index.csv` (filename, what it shows, key numbers) and `outputs/SUMMARY.md` (tag counts per stage,
  edge counts, runtime, discrepancy diagnostic, top findings, all assumptions).
- One consistent, readable style (large fonts, labelled axes, no clipped text) so figures drop straight into slides.

## 5. Notes on the data and method
- PC ignores autocorrelation in time-series data. That is what was requested, so do not change the method, but state it under
  "Limitations" in `SUMMARY.md`.
- The learned graph is not guaranteed correct (the paper assumes it is; see its Section 3 assumptions). Say so in `SUMMARY.md`.

## 6. Definition of Done
- [ ] No hardcoded paths outside `.env`; no git command was run.
- [ ] Combined matrix has no NaN/inf and no constant columns; tag count is logged and matches `tag_registry.csv`.
- [ ] PC graph: 0 directed edges out of any flare tag and 0 undirected edges touching a flare tag
      (an undirected edge could still mean flare -> X; orient it X -> flare). The DAG is acyclic.
- [ ] T² terms on the reference period are about N(0,1) (report mean/std for a sample of tags); the discrepancy
      diagnostic is computed; every tag has a chart.
- [ ] `figures_index.csv`, `SUMMARY.md` and `LOG.md` exist and agree with each other.
- [ ] One command reruns the whole pipeline from `.env`; it is documented at the top of `SUMMARY.md`.
