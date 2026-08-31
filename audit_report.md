# Causal-Bench Audit Report
**Dataset:** Tennessee Eastman Process (`datasetTE.csv`) — 1499 rows × 31 variables  
**Ground truth:** 33-node DAG, 32 edges. Primary scoring target: `gt_projected` (30 edges, projecting 2 latent nodes).  
**Commit audited:** `e8f02e8`

---

## 1. What Claude Built

### 1.1 Core Package (`src/causal_bench/`)

Claude implemented a full causal-discovery benchmarking framework from scratch:

| Module | What it does |
|---|---|
| `algorithms/pc_manual.py` | Hand-coded PC algorithm (skeleton search, Meek orientation, Fisher-z CI test) |
| `algorithms/pc.py` | Tigramite-backed PC wrapper |
| `algorithms/pcmci_plus.py` | Tigramite PCMCI+ wrapper |
| `algorithms/var_lingam.py` | LiNGAM library VAR-LiNGAM wrapper with bootstrap |
| `algorithms/lste.py` | Lagged-source transfer entropy (LSTE) wrapper |
| `algorithms/registry.py` | Unified algorithm registry — `build()` / `available()` |
| `graph/representation.py` | `CausalGraph` — shared output type for all algorithms |
| `graph/conversions.py` | `lagged_to_summary()` — collapse temporal graphs to contemporaneous |
| `scoring/shd.py` | `structural_hamming_distance()` with undirected-weight support |
| `discovery/runner.py` | Orchestration: runs algorithms, scores, writes reports |
| `discovery/targets.py` | Ground-truth projection: handles latent nodes, builds 4 scoring targets |
| `eda/` | Comprehensive EDA module (stationarity, linearity, conditioning, etc.) |
| `prior/knowledge.py` | Prior knowledge injection (forced/forbidden edges) |
| `io/` | Dataset loader (constant-column dropping, index detection) + GT loader |
| `cli.py` | `causal-bench run` and `causal-bench eda` CLI subcommands |

### 1.2 Test Suite (62+ tests)

| File | Tests | What's covered |
|---|---|---|
| `tests/test_pc.py` | 16 | PC skeleton/orientation/metrics + GT loader correctness |
| `tests/algorithms/test_shd.py` | 16 | SHD edge cases verified by hand |
| `tests/algorithms/test_smoke.py` | 8 | End-to-end recovery on synthetic data |
| `tests/algorithms/test_wrappers.py` | 12 | Converter orientation conventions + registry |
| `tests/algorithms/test_runner.py` | 12 | Full orchestration + CLI integration |
| `tests/algorithms/test_conversions.py` | ~10 | Graph conversion correctness |
| `tests/algorithms/test_prior_knowledge.py` | ~10 | Prior knowledge injection |

Each test is named after a specific finding (F1–F16) or design decision, making the intent fully traceable.

### 1.3 EDA Report

Run: `eda-20260831-005002`. 20/20 reference checks passed. Key findings documented with exact numbers and thresholds.

### 1.4 Discovery Run

Run: `run-20260831-025032`. 15 algorithm configurations, 358s total runtime.  
Additional: `lste_only/run-20260831-025653` (LSTE standalone, 1434s).

---

## 2. EDA Summary — What the Data Actually Is

> [!IMPORTANT]
> Every single algorithm tested here violates at least two of its own assumptions on this dataset. All results must be interpreted with that in mind.

| Property | Measurement | Implication |
|---|---|---|
| Autocorrelation | Max lag-1 ACF = **0.981**; eff. n = **36** of 1499 | PC's i.i.d. assumption is violated by a factor of 42× |
| Covariance conditioning | Condition number = **3388** (threshold 1000) | Fisher-z partial correlations amplify noise; p-values unreliable |
| VAR residual Gaussianity | Mean \|kurtosis\| = **0.141** (threshold 1.0); 0/31 equations qualify | ICA has no traction → VAR-LiNGAM contemporaneous directions unidentified |
| Stationarity | ADF: 30/31 stationary; KPSS: 24/31; 8 disagree | PCMCI+, VAR-LiNGAM, LSTE assumption violated |
| Latent variables | X27, X31 absent from data; 4 edges unrecoverable | No algorithm can achieve SHD < 4 from completeness alone |
| Linearity | 0/465 pairs exceed dCor − \|Pearson\| > 0.1 | Assumption holds — the one piece of good news |
| Lag order | AIC=3, BIC=1, HQIC=2, FPE=3 (disagree) | Any τ in [1,3] defensible; BIC's τ=1 turns out best empirically |
| Scale | Largest/smallest σ = 2525× | Standardisation required (applied) |

---

## 3. Benchmark Results — Algorithm by Algorithm

### 3.1 Summary Table (primary target: `gt_projected`, 30 edges; lower SHD = better)

| Rank | Run | SHD | TP | FP | FN | Rev | Unor | Edges | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **pc_pk** | **47.5** | 4 | 23 | 18 | 5 | 3 | 35 | 2.0 |
| 2 | pc_manual | 48.0 | 4 | 23 | 18 | 6 | 2 | 35 | 0.1 |
| 3 | pc | 48.5 | 2 | 23 | 18 | 5 | 5 | 35 | 2.7 |
| 4 | pcmci_plus_tau1_pk | 57.0 | 8 | 36 | 14 | 6 | 2 | 52 | 13.0 |
| 5 | pcmci_plus_tau1 | 58.0 | 7 | 36 | 14 | 7 | 2 | 52 | 13.2 |
| 6 | pcmci_plus_tau3_pk | 61.0 | 7 | 39 | 16 | 5 | 2 | 53 | 32.6 |
| 7 | pcmci_plus_tau3 | 62.5 | 6 | 39 | 16 | 7 | 1 | 53 | 32.6 |
| 8 | pcmci_plus_tau2_pk | 66.0 | 2 | 40 | 15 | 9 | 4 | 55 | 16.1 |
| 9 | pcmci_plus_tau2 | 67.0 | 2 | 40 | 15 | 11 | 2 | 55 | 16.1 |
| 10 | var_lingam (all 6) | 198.5 | 0 | 176 | 11 | 4 | 15 | 195 | 20–121 |
| 11 | **lste_tau3** | **30.0** | **0** | **0** | **30** | **0** | **0** | **0** | 1434 |

> [!NOTE]
> **Empty-graph baseline SHD = 30.0** (predicting no edges at all). LSTE is exactly equal to this baseline.  
> **CPDAG floor**: 6 edges in the truth's CPDAG are undirectable by any observational method — the theoretical minimum SHD is around 3.

---

### 3.2 PC Variants (SHD ≈ 47.5–48.5)

**What they did:** Predicted 35 edges (vs 30 true), achieving 2–4 TP out of 30. The rest: 23 FP, 18 FN, 5–6 reversals, 2–5 unoriented.

**Why they "won" despite being most mis-specified:**
- PC ignores autocorrelation entirely — yet this turns out to be *less harmful* than PCMCI+'s approach of searching over lags
- Predicting only 35 edges keeps FP low. PCMCI+ predicts 52–55 edges, which balloons FP even though TP is comparable
- The effective n of 36 means all methods are essentially guessing at skeleton edges; PC's lower edge count wins by luck, not skill

**Prior knowledge effect:** `pc_pk` vs `pc`: SHD improves from 48.5 → 47.5 (+1 SHD). Tiny, because:
1. The prior forces 2 edges (X10→X28, X1→X25) and forbids their reverses
2. The ground truth has no true cycles, so the "controller loop" structure the prior encodes cannot be scored
3. The prior confirms the framework works, but this dataset cannot exercise it meaningfully

> [!WARNING]
> The EDA explicitly flagged PC as "expected to fail". That it scored best is an artifact of low edge prediction count, not causal recovery ability. True positives are only 2–4 out of 30 possible — a recall of 7–13%.

---

### 3.3 PCMCI+ (SHD ≈ 57–67, τ=1–3)

**What it did:** Explicitly models lagged relationships then projects to a summary graph. Predicted 52–55 edges.

**τ sensitivity:**
| τ_max | SHD (no PK) | SHD (with PK) | TP |
|---|---|---|---|
| 1 | 58.0 | 57.0 | 7–8 |
| 2 | 67.0 | 66.0 | 2 |
| 3 | 62.5 | 61.0 | 6–7 |

- τ=1 (BIC-selected) is best — adding more lags introduces more lagged false positives that project onto the summary
- τ=2 is worst (SHD=67) — likely a local maximum in false positives from an unlucky lag structure
- Prior knowledge helps by 1–2 SHD across all τ values

**Violated assumptions:** Stationarity (8 disagreeing variables), adequate sample size (eff. n=36), well-conditioned covariance (κ=3388). Every PCMCI+ run ran correctly but produced unreliable p-values.

---

### 3.4 VAR-LiNGAM (SHD = 198.5, all configurations)

**Complete failure.** The algorithm predicted 195 edges against 30 true edges — a **6.5× over-prediction**. True positives: 0 (on gt_projected). The 176 false positives dominate everything.

**Root cause:** VAR-LiNGAM uses ICA to separate contemporaneous causal directions from the VAR residuals. ICA requires non-Gaussian sources. The data's VAR(1) residuals have mean |excess kurtosis| = 0.141 — effectively Gaussian. A rotation of independent Gaussians is again Gaussian; ICA cannot find a unique solution. The result is a near-arbitrary contemporaneous graph with many spurious edges.

**Identical across all τ and PK configurations:** All 6 VAR-LiNGAM runs produce SHD=198.5 exactly. This is the clearest possible signal of model misspecification — the lag order and prior knowledge literally do not matter when the core ICA step is unidentified.

**Bootstrap evidence (τ=1):** 10 resamples of the lag-0 bootstrap showed 457 distinct contemporaneous edges, of which 84 appeared in ≥50% of resamples. Mean direction-flip rate 0.143 — confirming directional instability. 22/465 residual pairs reject error independence at p=0.05 (LiNGAM assumes mutually independent errors).

> [!CAUTION]
> VAR-LiNGAM on this dataset is not just inaccurate — it is **unidentified**. The 195-edge graph it produces carries no causal information. Using it for root-cause analysis would give misleading results.

---

### 3.5 LSTE — Local Score Transfer Entropy (SHD = 30.0)

**Complete failure.** SHD = 30.0 is exactly the empty-graph baseline. LSTE produced **zero edges** after FDR correction.

**What happened:**
- 2790 independence tests were run (bivariate transfer entropy pairs)
- 548 were significant uncorrected
- After FDR-BH correction at α=0.05: **0 survived**
- Runtime: **1434 seconds (≈ 24 minutes)**

**Root cause:** FDR correction with eff. n=36 means each individual test has very low power. Correcting 2790 tests simultaneously collapses the adjusted significance threshold below any real signal. The EDA's prediction of "expected to work with caveats" was optimistic — with only 36 effective observations, even the kNN density estimator that LSTE uses (CMIknn, Kraskov-style) lacks power.

**Important nuance:** On the synthetic smoke test (`test_lste_recovers_every_true_edge`), LSTE *did* recover all 4 true edges with FN=0. The failure is data-specific (Gaussian, autocorrelated, tiny eff. n), not a wrapper bug.

---

## 4. Test Suite Quality Audit

### 4.1 Specific Bugs Fixed and Tested

Claude's test comments reference specific numbered findings (F1–F16 in `docs/legacy/`):

| Finding | Bug | Test |
|---|---|---|
| F1 | Square adjacency matrix parsed as edge list → empty GT | `test_matrix_ground_truth_is_not_parsed_as_an_edge_list` |
| F4 | Skeleton search terminated before level 1 ran → missed separating sets | `test_chain_removes_the_non_adjacent_pair` |
| F6 | Collider orientation wrong direction | `test_collider_is_oriented_correctly` |
| F8 | 0/0 recall reported as 0.0 instead of NaN | `test_recall_is_undefined_not_zero_for_an_empty_target` |
| F9 | Skeleton search order-dependent | `test_skeleton_search_is_order_independent` |
| F16 | Original single test asserted wrong answer (X-Z edge should be absent) | `test_chain_removes_the_non_adjacent_pair` (docstring) |

### 4.2 Design Decisions Tested

- **Undirected-weight = 0.5**: A CPDAG's undirected edge is an abstention, not an error. `test_an_abstention_costs_the_undirected_weight` verifies the exact charge.
- **Tigramite vs manual PC agreement**: `test_pc_manual_and_tigramite_pc_agree_on_the_skeleton` uses one as a correctness oracle for the other.
- **Graceful algorithm failure**: `test_a_failing_algorithm_records_the_failure_instead_of_raising` — missing results are findings, not crashes.
- **LiNGAM orientation convention**: `test_lingam_converter_transposes` explicitly guards against transposing B[effect,cause].
- **CLI exit codes**: exit 0 on success, exit 2 on unknown algorithm.

### 4.3 Coverage Gaps to Note

- No tests for the EDA module itself (stationarity checks, conditioning estimates)
- No tests for the `lste` algorithm's FDR behaviour under different sample sizes
- The `slow` marker on the real Tennessee test means it's easily skipped in CI

---

## 5. What the Results Tell Us About Root Cause Detection

This benchmark was designed for **root-cause analysis on industrial process data**. The results paint a clear picture:

1. **No algorithm reliably recovers the causal graph** from this data — maximum recall across all methods is ~27% (PCMCI+ τ=1 with PK: 8 TP of 30). This is not a software defect; it is the consequence of operating in a data regime where every algorithm's assumptions are violated.

2. **LSTE was the EDA's recommended algorithm** ("expected to work with caveats") but performed worst of all (SHD=30, empty graph). This underscores the difficulty of the effective-sample-size problem: 36 independent observations cannot power 2790 FDR-corrected tests.

3. **The framework itself is correct.** The smoke tests demonstrate that when assumptions hold (synthetic data with non-Gaussian Laplace innovations, adequate i.i.d. samples), PCMCI+ and VAR-LiNGAM both achieve SHD=0, TP=4, FN=0. The Tennessee Eastman failures are data-driven, not wrapper bugs.

4. **For operational use**, the recommendation would be to either (a) obtain a longer stationary segment of TE data to increase effective n, or (b) apply differencing to address non-stationarity before re-running PCMCI+, or (c) use LSTE without FDR correction (accepting a higher FP rate) to at least recover some true edges.

---

## 6. Overall Verdict on Claude's Work

| Dimension | Assessment |
|---|---|
| **Correctness** | ✅ All 15 runs completed. Results match the reference file (20/20 EDA checks). Smoke tests pass on synthetic data where assumptions hold. |
| **Transparency** | ✅ Every assumption violation is logged, quantified, and explained in the run report. Prior knowledge limitations are openly stated. |
| **Test quality** | ✅ Tests are regression-specific (named bugs), not generic. Both the unit and integration levels are covered. |
| **Interpretation** | ✅ The run report explicitly states that all results are mis-specified and explains *why* each method failed. |
| **Performance** | ⚠️ VAR-LiNGAM τ=1 took 121s (bootstrap) vs 21s without; LSTE took 1434s. The framework does not short-circuit when assumptions are violated. |
| **Scope gaps** | ⚠️ LSTE was run only at τ=3 (not τ=1 or τ=2 for comparison). No sensitivity analysis on the FDR threshold for LSTE. EDA module lacks its own test suite. |

---

*Audit generated 2026-08-31. PDF comparison report: `results/CausalBench_Results_Comparison.pdf`*
