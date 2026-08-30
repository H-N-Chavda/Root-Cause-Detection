# Is Your PC Algorithm Ready for Industrial Time-Series Causal Discovery? A Literature-Grounded Review

## TL;DR
- **A standard, i.i.d.-oriented PC implementation is not, by itself, appropriate for process/industrial multivariate time-series like the Tennessee Eastman Process (TEP): autocorrelation violates the i.i.d. assumption underlying its conditional-independence (CI) tests, inflating false positives and depressing recall, and plain PC has no native notion of time lag or propagation delay.** Before trusting your code, you must at minimum add order-independence (PC-stable), temporal structure (lagged variables / time-order constraints), autocorrelation-robust CI testing, and stability/bootstrap validation.
- **The most directly relevant modern methods for your use case are PCMCI/PCMCI+ (constraint-based, autocorrelation-aware, lagged + contemporaneous), and — specifically for process root-cause/propagation problems — transfer-entropy variants including Lag-Specific Transfer Entropy (LSTE) and physics/connectivity-informed causal inference.** None "prove" causality; each establishes only what its assumptions permit.
- **When ground truth is unavailable, no metric establishes causal correctness.** Use graph-to-graph metrics (SHD for structure, SID for causal-ordering/intervention implications) to compare two estimates, and rely on bootstrap/subsampling stability, cross-regime replication, and — most powerfully in a process setting — agreement between estimated propagation delays and known process physics (flow direction, transport/residence times).

## Key Findings

1. **Plain PC is theoretically mis-specified for autocorrelated time series.** PC's CI tests assume i.i.d. sampling; time-series autocorrelation breaks this, so tests become miscalibrated. Runge, Nowack, Kretschmer, Flaxman & Sejdinovic (*Science Advances* 5(11):eaau4996, 2019) state that "the PC algorithm cannot be directly used for the time series case, in particular since autocorrelation" degrades it; Runge (UAI 2020) states that "existing CI-based methods such as the PC algorithm and also common methods from other frameworks suffer from low recall and partially inflated false positives for strong autocorrelation which is an ubiquitous challenge in time series." However, PC *can* be made consistent for certain stationary Gaussian dependent processes and extended with time-order — so the issue is fixable, not fatal.

2. **The essential PC upgrades since 2000 are well-established:** order-independence (PC-stable, Colombo & Maathuis 2014), collider-orientation robustness (Conservative-PC; PC-Max/majority rule), latent-confounder handling (FCI/RFCI), high-dimensional consistency (Kalisch & Bühlmann 2007), and error control (edge-specific p-values / FDR). A modern PC should be PC-stable at minimum.

3. **For time series, the field has largely moved to temporal extensions of PC** — chiefly PCMCI (Runge et al. 2019) and PCMCI+ (Runge 2020), which add momentary conditional independence (MCI) to recalibrate CI tests under autocorrelation and to recover both lagged and contemporaneous links.

4. **Alternative families make different trade-offs:** VAR-LiNGAM (linear, non-Gaussian, orients contemporaneous edges via ICA); TiMINo (restricted SEM, nonlinear, conservative — abstains rather than err); Granger causality (predictive, not structural; linear GC ≡ transfer entropy for Gaussians); transfer entropy (model-free, nonlinear, directional); DYNOTEARS (score-based continuous optimization for dynamic Bayesian networks).

5. **In the specific domain of plant-wide oscillation root-cause analysis** (Bauer, Thornhill, Yuan, Qin, Duan and colleagues), transfer entropy and Granger causality are the dominant data-driven tools, increasingly fused with process-connectivity knowledge. The 2024 AIChE Journal physics-informed sparse causal inference paper and the 2025 Lag-Specific Transfer Entropy paper are the two most directly relevant recent works for your TEP/process setting.

6. **Evaluation without ground truth cannot certify correctness.** SHD and SID quantify differences between two graphs but a small distance to another *estimate* means nothing about truth. The defensible evidence is stability under resampling plus consistency with independent knowledge (process physics, propagation delays, known engineering relationships).

## Details

### 1. Evolution of PC — historical foundations

The PC algorithm (named for **Peter Spirtes and Clark Glymour**) was introduced in Spirtes & Glymour (1991) and canonically in Spirtes, Glymour & Scheines, *Causation, Prediction, and Search*, 2nd ed., MIT Press, 2000. It is a constraint-based method: start from a complete undirected graph; delete edges via CI tests over conditioning subsets of adjacent nodes; orient v-structures (colliders) using separating sets; then apply Meek's orientation rules to propagate directions. The output is a CPDAG representing a Markov equivalence class, under three core assumptions: **causal Markov, faithfulness, and causal sufficiency** (no latent confounders), plus acyclicity and (classically) i.i.d. sampling.

Foundational developments (roughly 2000–2015), which define what a "modern" PC should contain:
- **High-dimensional consistency:** Kalisch & Bühlmann (2007), "Estimating high-dimensional directed acyclic graphs with the PC-algorithm," *JMLR* 8:613–636 — PC is uniformly consistent for sparse high-dimensional DAGs under a strengthened faithfulness condition; also the standard partial-correlation/Fisher-z Gaussian CI test.
- **Order-independence (PC-stable):** Colombo & Maathuis (2014), "Order-independent constraint-based causal structure learning," *JMLR* 15:3921–3962. Plain PC's output depends on variable ordering; this is minor in low dimensions but "very pronounced in high-dimensional settings." PC-stable defers edge removals to the end of each conditioning-set-size level, yielding an order-independent skeleton (at the cost of more CI tests). Implemented in `pcalg` (R) and `causal-learn` (Python, `stable=True` by default).
- **Orientation robustness:** Conservative-PC (Ramsey, Zhang & Spirtes 2006) orients a collider only if the middle node is in *none* of the separating sets; PC-Max (Ramsey 2016) uses the largest-p-value conditioning set. Both reduce orientation errors from faulty CI tests.
- **Latent confounders:** FCI and RFCI (Spirtes et al.; Colombo et al. 2012) relax causal sufficiency and output a PAG (with bidirected edges o–o / ↔ indicating possible latent common causes).
- **Error control:** Strobl, Spirtes & Visweswaran (2016/2019) — edge-specific p-values and FDR control for the PC algorithm.

### 2. Current state of the art (last ~5 years)

The dominant trend for time series is **CI-based temporal methods that fix autocorrelation calibration**, plus continuous-optimization (score-based) DAG learners and functional-model methods:

- **PCMCI** (Runge, Nowack, Kretschmer, Flaxman & Sejdinovic, "Detecting and quantifying causal associations in large nonlinear time series datasets," *Science Advances* 5(11):eaau4996, 2019; and the perspective Runge et al., "Inferring causation from time series in Earth system sciences," *Nature Communications* 10:2553, 2019). Two stages: **PC1** (a stable-PC-style condition-selection to find each variable's lagged parents) then **MCI** (momentary conditional independence), which conditions on parents of *both* the driver and the target to recalibrate the test under autocorrelation and control false positives.
- **PCMCI+** (Runge, "Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets," UAI 2020, PMLR 124:1388–1397). Extends PCMCI to **contemporaneous (instantaneous, lag-0)** links, separates lagged vs contemporaneous conditioning, is order-independent and consistent in the oracle case. Per the paper's own conclusions, PCMCI+ "yields much higher recall, well-controlled false positives, and faster runtime than the original PC algorithm for highly autocorrelated time series, while maintaining similar performance for low autocorrelation." Autocorrelation actually *helps* contemporaneous orientation — again per Runge (2020): "Autocorrelation is actually key to increase contemporaneous orientation recall since it creates triples X^i_{t-1} → X^i_t ∘–∘ X^j_t that can often be oriented while an isolated link X^i_t ∘–∘ X^j_t stays undirected in the Markov equivalence class."
- **LPCMCI** (Gerhardus & Runge, "High-recall causal discovery for autocorrelated time series with latent confounders," NeurIPS 2020) — relaxes causal sufficiency, outputs a PAG; higher recall than SVAR-FCI/SVAR-RFCI in autocorrelated, latent, nonlinear settings.
- **Bagged-PCMCI+** (Debeire et al., "Bootstrap aggregation and confidence measures to improve time series causal discovery," CLeaR 2024, PMLR 236) — a time-series bootstrap preserving temporal/lag structure that yields per-edge confidence and improves precision/recall.
- **Score-based / continuous:** DYNOTEARS (Pamfil et al., AISTATS 2020, PMLR 108) learns dynamic Bayesian networks (intra-slice + inter-slice) via a smooth acyclicity constraint; NTS-NOTEARS adds nonlinearity. These assume a fixed structure across time.
- All are implemented in **`tigramite`** (PCMCI family), **`lingam`** (VAR-LiNGAM), **`causal-learn`** and `pcalg` (PC/FCI), and `causalnex` (DYNOTEARS).

Note on hype/uncertainty: several 2025–2026 items retrieved (e.g., Adaptive-MCI/ECD-aMCI in *Mathematics*, assimilative causal inference in *Nature Communications*, various "causal audit" frameworks) are recent and not yet broadly validated; treat them as promising but unproven.

### 3. PC for time-series data — what breaks and what fixes it

**What goes wrong applying plain PC directly to process data:**
- **Autocorrelation → miscalibrated CI tests.** Analytical null distributions for partial-correlation/Fisher-z assume i.i.d. data; serial dependence reduces the effective degrees of freedom (Afyouni, Smith & Nichols 2019 quantify this bias for correlations), so tests are anti-conservative → spurious edges. Autocorrelation also lowers CI-test effect size via unfortunate conditioning sets → missed true links → cascading wrong orientations (the core motivation for MCI).
- **No temporal precedence / lags.** Plain PC has no time index; it cannot represent that X at t−τ causes Y at t, nor recover propagation delays — central to process root-cause analysis.
- **Contemporaneous ambiguity.** Instantaneous links often stay unoriented in the Markov equivalence class (a generic CPDAG limitation).
- **Non-stationarity, operating-regime changes** violate the single-distribution assumption; TEP fault scenarios are explicitly non-stationary.
- **Strong collinearity/oscillations** (plant-wide oscillations) induce near-unfaithful cancellations and spurious partial correlations; periodicity specifically breaks standard surrogate/significance tests.
- **Latent common causes** (unmeasured flows, ambient conditions) violate causal sufficiency → use FCI-type/PAG methods.

**Fixes / temporal adaptations:**
- Build a **time-lagged (window) graph**: replicate variables at lags 0…τmax and let time-order auto-orient lagged edges (Time-Aware PC; Entner & Hoyer's FCI-for-time-series; Malinsky & Spirtes SVAR-FCI).
- Use **MCI-style conditioning** (PCMCI) to recalibrate under autocorrelation.
- Choose **τmax** using partial-autocorrelation inspection / domain knowledge — too small misses long-delay links, too large inflates dimensionality and spurious associations.
- Consider that PC has been proven consistent for stationary Gaussian ρ-mixing processes (Time-Aware PC work, arXiv 2210.09038), so with time-order and autocorrelation-robust testing it is theoretically usable.

### 4. Modern alternatives — evaluated on the evidence

**PCMCI / PCMCI+ (constraint-based, temporal).** Problem solved: reliable lagged + contemporaneous discovery in high-dimensional, autocorrelated, possibly nonlinear time series. Differs from PC by the MCI step and by native lag handling. Assumptions: causal sufficiency (PCMCI/PCMCI+), stationarity, Markov + faithfulness; CI-test choice is pluggable (ParCorr for linear-Gaussian; GPDC/CMIknn for nonlinear). Verdict: **the most natural upgrade path from your existing PC code**, since PC1 is essentially stable-PC and the framework reuses CI-testing machinery. Better fit than plain PC for TEP-type data. Cannot resolve all edge directions (CPDAG-level for contemporaneous), and assumes no latent confounders (use LPCMCI if that fails).

**VAR-LiNGAM / LiNGAM.** LiNGAM (Shimizu et al. 2006) identifies a full DAG from non-Gaussian, linear, acyclic, causally sufficient data via ICA. VAR-LiNGAM (Hyvärinen, Zhang, Shimizu & Hoyer, "Estimation of a Structural Vector Autoregression Model Using Non-Gaussianity," *JMLR* 11:1709–1731, 2010) combines a VAR with LiNGAM on residuals to get **both lagged and instantaneous** effects, and — unlike CPDAG methods — can **fully orient** contemporaneous edges by exploiting non-Gaussianity. Assumptions: linearity, non-Gaussian noise, acyclicity, no hidden confounders. Limitation: Runge (2020, UAI) explicitly reports "we saw that LiNGAM suffers from large autocorrelation." Relevance: useful when process noise is clearly non-Gaussian and relationships approximately linear; orientation power is attractive, but sensitivity to autocorrelation is a concern for strongly dynamic loops.

**TiMINo** (Peters, Janzing & Schölkopf, "Causal Inference on Time Series using Restricted Structural Equation Models," NeurIPS 2013). A restricted-SEM (additive-noise) framework requiring **independent residual time series**; covers lagged and instantaneous, nonlinear, possibly unfaithful effects; **abstains** ("remains undecided") when assumptions fail rather than emitting wrong edges. Strength: conservative and honest — valuable in high-stakes process diagnosis. Limitations: relies on nonlinear independence tests (weak on small samples); no feedback loops. Relevance: attractive as a *conservative cross-check* rather than a primary high-recall method.

**Granger / Conditional Granger causality** (Granger 1969). Establishes **predictive** causality: X Granger-causes Y if X's past improves prediction of Y beyond Y's own past (and, in conditional/multivariate GC, beyond other series). It is *not* structural causality — it can be misled by latent common drivers, non-stationarity, and (critically for process data) periodic oscillations, which "always tend to produce spurious causation." For Gaussian variables, **linear GC is mathematically equivalent to transfer entropy** (Barnett, Barrett & Seth, "Granger causality and transfer entropy are equivalent for Gaussian variables," *Phys. Rev. Lett.* 103:238701, 2009). Widely used for process root-cause (Yuan & Qin 2014, "Root cause diagnosis of plant-wide oscillations using Granger causality," *J. Process Control* 24(2):450–459). Relevance: cheap, well-understood, but predictive-only; needs sparsity/regularization and oscillation-aware significance testing for process data.

**Transfer entropy (TE)** (Schreiber 2000). Model-free, nonparametric, directional information transfer; captures nonlinear relationships GC misses (except in the Gaussian case where they coincide). The seminal process application is Bauer, Cox, Caveness, Downs & Thornhill, "Finding the Direction of Disturbance Propagation in a Chemical Process Using Transfer Entropy," *IEEE Trans. Control Systems Technology* 15(1):12–21, 2007, and Bauer & Thornhill, "A practical method for identifying the propagation path of plant-wide disturbances," *J. Process Control* 18(7–8):707–719, 2008. Duan et al. (2014, *AIChE J.* 60:2019–2034) compare spectral envelope, adjacency matrix, GC, TE, and Bayesian-network methods on an industrial benchmark. Limitations: data-hungry, sensitive to estimation choices (embedding, bin/KNN, bias correction — "effective TE" subtracts a shuffled-surrogate bias), and hard in high-dimensional multivariate settings (curse of dimensionality). Relevance: strong for pairwise directional propagation, especially when flow/connectivity is unknown; needs significance testing and often connectivity pruning.

**Lag-Specific Transfer Entropy (LSTE)** — Chen, Liang, Wang, Yao, Su & Liu, "Lag-Specific Transfer Entropy for Root Cause Diagnosis and Delay Estimation in Industrial Sensor Networks," *Sensors* 25(13):3980, 2025, DOI 10.3390/s25133980. Method: (1) a **self-prediction optimization (SPO)** step removes each sensor's own information storage by conditioning Y's future on its immediately-preceding value (h0=1), preventing overestimation of TE from X; (2) TE is computed across candidate lags h up to hmax (KNN/Kraskov estimator) and the delay is chosen as **δ = argmax_h SPOTE_{X→Y}(h)**, so the method returns both the strength and the propagation delay of each link (when δ=1, LSTE reduces to classical TE); (3) significance via **time-shifted surrogates** — cyclically shifting the driver by a random offset to build a null, with M=100 surrogates, α=0.05, one-sided p-value. Validation (verified from the paper): a numerical simulation where **LSTE reached accuracy/F1 = 1.00 vs 0.84/0.78 for classical TE**, and identified the root cause down to SNR = −10 dB; **TEP** faults IDV5 and IDV8, where recovered delays matched process mechanism (e.g., condenser cooling-water flow X52 → separator temperatures within 3 min, reactor pressure after 9 min); the **Cranfield three-phase flow rig** (slugging), where LSTE correctly identified the top-riser flow as root while TE misidentified it, with 1 s vs 5 s delays tracking physical distance along the riser; and a **full-scale blast furnace**, where estimated lags between operating variables and molten-iron temperature (e.g., blast temperature → PT ≈ 115 min, p=0.0034) were "reviewed and validated by blast furnace ironmaking experts." Stated limitations: computationally intensive (needs candidate-variable pre-selection), assumes a **single disturbance**, developed under **stationary** conditions, and high-frequency data raises noise/false-positive risk. Relevance to your task: **highly relevant** — it is validated on TEP and provides a physics-checkable output (propagation delay) usable as a validation criterion even without a ground-truth graph.

**Physics-informed sparse causal inference** — Madhusoodanan, Chiplunkar, Puli & Huang, "Physics-informed sparse causal inference for source detection of plant-wide oscillations," *AIChE Journal* 70(4):e18362, 2024, DOI 10.1002/aic.18362. Problem: pure data-driven Granger causality is unreliable when industrial data carry sensor errors and when measurements are limited. Method: builds on **Granger causality and sparse Granger causality (SGC)** — sparsity imposed by an **L1/LASSO penalty** added to the GC objective, controlled by a tuning hyperparameter — and **amalgamates expert/process-connectivity knowledge** (flowsheet adjacency/topology) with the observed data to reconstruct causal maps, reducing over-dependence on data. It also introduces a **novel surrogate-data significance test specifically for oscillatory/periodic data**, because standard Fourier-transform and cycle-phase-permutation surrogates fail to destroy the causal property when data are periodic. Validation: a simulation plus an industrial case study (physics-informed causal map recovered correct stream/level connections while suppressing reverse/spurious links that pure GC produced). Relevance: **directly on point** for your stated situation where flow information may be partial and only measured trends are available — it shows how partial process knowledge can be injected to stabilize causal maps and how to test significance under oscillations. (Note: the exact mathematical fusion mechanism and quantitative case-study metrics could not be fully verified from the paywalled full text.)

### 5. Algorithm comparison table

Cells marked "n/e" = not established / depends on configuration. "Contemp." = contemporaneous (lag-0) effects.

| Method | Main idea | Temporal lags? | Contemp. effects? | Nonlinearity | Non-Gaussianity | Autocorrelation handling | Hidden confounders | Noise robustness | Output | Compute cost | Key limitations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Plain PC** | CI tests prune skeleton, orient colliders + Meek rules | No (native) | Only via CPDAG, often unoriented | Via nonparametric CI test | Not exploited | **Poor** (i.i.d. assumption violated) | No (assumes sufficiency) | Degrades with miscalibration | CPDAG | Moderate (sparse) | Order-dependent unless stable; mis-specified for autocorrelated TS |
| **PC-stable** | Order-independent skeleton | No native | CPDAG | Via CI test | Not exploited | Poor (same test issue) | No | Same as PC | CPDAG | Higher than PC | Still i.i.d.-oriented |
| **PCMCI** | PC1 condition-selection + MCI | **Yes** | No (lagged focus) | Yes (CI-test dependent) | n/e | **Good** (MCI recalibrates) | No | Good with right CI test | Time-series graph (lagged) | Moderate–high | Assumes sufficiency, stationarity |
| **PCMCI+** | PCMCI + contemporaneous phase | **Yes** | **Yes** (CPDAG for contemp.) | Yes | n/e | **Good/benefits** | No | Good | Time-series CPDAG | Moderate (faster than PC on autocorr.) | Contemp. edges may stay unoriented; sufficiency |
| **LPCMCI** | PCMCI + latent handling | Yes | Yes | Yes | n/e | Good | **Yes (PAG)** | Good | Time-series PAG | High | Faithfulness; complex |
| **VAR-LiNGAM** | VAR + ICA/LiNGAM on residuals | **Yes** | **Yes (fully oriented)** | No (linear) | **Required** | Reported to suffer under strong autocorr. | No | Moderate | Full DAG (lagged+contemp.) | Moderate | Linear; needs n>vars; autocorr-sensitive |
| **TiMINo** | Restricted (additive-noise) SEM | Yes | Yes | **Yes** | Exploited (ANM) | n/e | Partial (abstains) | Conservative | Summary graph or "undecided" | Moderate–high | Small-sample independence tests; no feedback |
| **Granger (cond.)** | Predictive: past improves forecast | **Yes** | No | No (linear) unless extended | Not required | Handled via VAR lags but spurious under oscillation | No | Poor with sensor error | Directed (predictive) graph | Low | Predictive≠structural; oscillation false positives |
| **Transfer entropy** | Info-theoretic directed transfer | **Yes** | No | **Yes** | Not required | Needs surrogate testing | No | Estimator-sensitive | Directed map | High (density est.) | Curse of dimensionality; tuning-sensitive |
| **LSTE** | SPO + lag-scanned TE | **Yes (recovers delay)** | No | **Yes** | Not required | Surrogate significance | No | Robust to ~−10 dB SNR (reported) | Directed map + delays | High | Single-disturbance, stationary assumptions |
| **DYNOTEARS** | Score-based continuous DAG opt. | **Yes** | **Yes** | No (linear; NTS-NOTEARS adds it) | n/e | n/e | No | n/e | DBN (intra+inter slice) | Scales to high-dim | Fixed structure over time; linear |
| **Physics-informed SGC** | GC+LASSO fused with connectivity | **Yes** | n/e | No (GC-based) | Not required | Oscillation-aware surrogates | Mitigated via priors | **Improved** under sensor error | Causal map | Moderate | Needs process knowledge; GC core |

### 6. PC implementation audit checklist

Use this to inspect `utils.py`/`main.py`/`test_pc.py`. For each item: [Classical PC?] / [Later improvement?] / [Time-series-critical?] / [Needed for your use case?]

1. **CI test present & appropriate** — Classical: yes. Confirm you use partial-correlation/Fisher-z for linear-Gaussian, and have a nonparametric/nonlinear option (KCI, CMIknn) for TEP nonlinearity. *TS-critical: yes; Needed: yes.*
2. **Autocorrelation-robust CI testing** — Not classical; later/temporal improvement. Standard Fisher-z is miscalibrated on autocorrelated data (reduced effective dof). *TS-critical: yes; Needed: yes* (this is the single most important gap for TEP).
3. **Significance threshold (α) handling** — Classical. Confirm α is configurable and you understand it is not a false-discovery rate. *Needed: yes.*
4. **Multiple-testing / FDR control** — Later improvement (Strobl et al.). PC runs thousands of tests. *Needed: yes for high-dimensional TEP.*
5. **Skeleton discovery correctness** — Classical. Confirm conditioning sets are drawn from adjacency sets of both endpoints. *Needed: yes.*
6. **Order-independence (PC-stable)** — Later improvement (Colombo & Maathuis 2014). Confirm edge deletions are deferred to end of each level. *TS-critical: yes (high-dim); Needed: yes.*
7. **Sepset storage & use** — Classical; required for correct collider orientation. *Needed: yes.*
8. **Collider orientation rule + robustness** — Classical (basic); Conservative-PC / PC-Max are improvements that reduce orientation errors from faulty tests. *Needed: recommended.*
9. **Meek orientation rules (R1–R4)** — Classical; confirm all rules and conflict handling. *Needed: yes.*
10. **Background knowledge / tiers** — Later improvement; lets you inject temporal tiers and known process constraints. *TS-critical: yes; Needed: yes* (encode time-order and known flow direction).
11. **Temporal/lag construction** — Not in classical PC. Confirm whether you build lagged variables and enforce time-order orientation. *TS-critical: yes; Needed: yes* (currently likely absent).
12. **Non-stationarity handling** — Not classical. TEP faults are non-stationary; consider regime segmentation. *Needed: recommended.*
13. **Missing-data handling** — Later improvement (test-wise deletion / MVPC). *Needed: if applicable.*
14. **Finite-sample behavior** — Confirm behavior when conditioning sets are large relative to n. *Needed: yes.*
15. **Numerical issues** — Near-singular covariance under collinearity/oscillation; regularize partial-correlation inversion. *Needed: yes for oscillatory process data.*
16. **Computational optimization** — Later (Parallel-PC, GPU). *Needed: if scaling to full TEP variable set.*
17. **High-dimensional robustness** — Kalisch–Bühlmann consistency requires sparsity + strong faithfulness. *Needed: yes.*
18. **Output representation** — Classical: CPDAG. Confirm you distinguish directed/undirected edges and don't over-claim orientation; consider PAG output if latents suspected. *Needed: yes.*
19. **Stability/bootstrap wrapper** — Not classical; essential for validation without ground truth. *Needed: yes.*
20. **Ground-truth comparison metrics** — Confirm `test_pc.py` uses SHD and ideally SID against your TEP ground-truth file, not just edge accuracy.

### 7. Evaluating causal graphs without ground truth

**A. Graph-to-graph comparison (comparing two estimates).**
- **Structural Hamming Distance (SHD)** — counts edge insertions/deletions/reversals to turn one graph into another; measures structural similarity. Treats all edge errors equally; good for skeleton/adjacency fidelity.
- **Structural Intervention Distance (SID)** — Peters & Bühlmann, "Structural Intervention Distance for Evaluating Causal Graphs," *Neural Computation* 27(3):771–799, 2015. Counts pairs of variables for which the two graphs imply different intervention (adjustment) distributions; prioritizes **causal ordering** over individual edges, and is well-suited when the graph will be used to compute interventions. Can compare CPDAGs (via bounds). Empirically PC can score well on SHD but poorly on SID — i.e., good skeleton, unreliable orientations.
- **Adjustment Identification Distance (AID)** (Henckel et al. 2024) generalizes SID; separation-based distances (2024) are further alternatives.
- **Crucial caveat:** these are *distances between graphs*. If neither graph is ground truth, a small SHD/SID tells you the two estimates *agree*, not that either is *correct*. Agreement across methods is corroboration, not proof — all methods can share the same bias (e.g., from autocorrelation).

**B. Validation without ground truth (credibility of a discovered link).**
- **Bootstrap / subsampling stability:** re-run the algorithm on many resamples; an edge's inclusion frequency is a confidence score. Stability-selection theory (Meinshausen & Bühlmann, "Stability Selection," *J. R. Statist. Soc. B* 72(4):417–473, 2010) requires the selection threshold π_thr to lie in (½,1); the authors report that results were insensitive to π for **π ∈ [0.6, 0.9]**, a reasonable default band for calling an edge "stable." For time series, use a **temporal bootstrap that preserves lag structure** (Bagged-PCMCI+, Debeire et al. 2024). *Caveat (Glymour et al.): stable output is not necessarily correct — stability is necessary, not sufficient.*
- **Replication across operating regimes / cross-dataset validation:** links that persist across independent normal-operation windows or plants are more credible.
- **Predictive validation:** does the implied structure improve out-of-sample forecasting or invariant prediction across regimes?
- **Interventional/perturbation validation where possible:** the gold standard — even a few known set-point changes or historical interventions can confirm/refute a direction.
- **Physics/process-knowledge consistency (strongest in your setting):** does the edge respect flow direction, known unit connectivity, and mass/energy balances?
- **Propagation-delay consistency:** if a method recovers lags (LSTE, PCMCI), check that estimated delays match known transport/residence times — this is a *physically checkable, ground-truth-free* criterion that LSTE and physics-informed methods exploit explicitly.
- **Lag consistency & significance:** genuine links show consistent, significant lags across surrogate tests (time-shifted/oscillation-aware surrogates).

**Bottom line for "two graphs, no ground truth":** prefer the graph that is (1) more stable under temporal resampling, (2) more consistent with process physics and known connectivity, (3) recovers propagation delays matching transport/residence times, and (4) better predicts held-out data or known interventions. SHD/SID alone cannot decide it.

### 8. Industrial/process-data implications

For temperature, pressure, level, flow, and composition variables with strong dynamics:
- **Delays matter and carry physical meaning** — methods that recover propagation lags (LSTE, PCMCI, Bauer–Thornhill TE) give directly checkable outputs against transport/residence times. This is your best validation lever.
- **Plant-wide oscillations** demand oscillation-aware significance testing; standard surrogates fail on periodic data (physics-informed SGC paper).
- **Sensor noise and errors** degrade GC/TE; physics/connectivity priors materially help (physics-informed SGC). LSTE reports robustness down to −10 dB SNR in simulation.
- **When flow information is unavailable** (your stated concern): purely data-driven directional methods (TE/LSTE, PCMCI) remain usable because they infer direction from data + time-order rather than requiring the flowsheet; connectivity knowledge, when partially available, should be injected as background constraints rather than treated as mandatory.
- **Missing/irregular sampling** breaks the regular-lag assumption of PCMCI+/VAR-LiNGAM; interpolation distorts delays — flag as a real risk if your TEP export has gaps.
- **Non-stationary operating regimes / faults** — segment by regime or use regime-aware methods (Regime-PCMCI); TEP fault scenarios are explicitly non-stationary.
- **Latent/unmeasured causes** (ambient, unmeasured flows) — if suspected, prefer PAG-output methods (FCI/LPCMCI) over sufficiency-assuming PC.

## Recommendations

**Stage 1 — Audit and harden your existing PC (do first, low effort).**
1. Confirm PC-stable (order-independent) skeleton; if absent, add it.
2. Replace/augment the CI test: keep Fisher-z for a linear baseline but recognize it is **miscalibrated on autocorrelated data**. Add FDR control across the many CI tests.
3. Add background-knowledge/tiers to encode **time-order** and any known flow direction.
4. Ensure `test_pc.py` scores against the TEP ground truth with **both SHD and SID** (not just edge accuracy), and treat SID as the more decision-relevant metric.
- *Threshold to advance:* if PC's SHD/SID against ground truth is poor or bootstrap edge-stability is low (below the ~0.6 stability band for most edges), plain PC is insufficient — proceed to Stage 2.

**Stage 2 — Move to a temporal method (recommended primary).**
5. Adopt **PCMCI+** (via `tigramite`) as your main discovery method: it reuses your PC/CI intuition, handles lagged + contemporaneous links, and is autocorrelation-robust. Choose the CI test to match the data (ParCorr if approximately linear-Gaussian; CMIknn/GPDC for nonlinearity). Tune τmax from partial-autocorrelation and process time-constants.
6. Wrap it in **Bagged-PCMCI+** for per-edge confidence.
- *Threshold:* if latent confounders are suspected (edges that flip across regimes, unmeasured flows), switch to **LPCMCI** (PAG output).

**Stage 3 — Cross-check with complementary paradigms.**
7. Run **VAR-LiNGAM** if residuals are clearly non-Gaussian and you need full orientation of contemporaneous edges; run **TiMINo** as a conservative confirmer (it abstains rather than err).
8. For the specific **root-cause / propagation** question, run a **transfer-entropy / LSTE** analysis: it yields propagation delays you can validate against process physics, and (per the 2025 paper) is validated on TEP itself.
9. If you have partial connectivity/flowsheet knowledge, apply the **physics-informed sparse-GC** approach to stabilize the map and use oscillation-aware surrogates for significance.

**Stage 4 — Validate without ground truth (always).**
10. Report bootstrap edge-stability; keep only edges stable across temporal resamples and across operating regimes.
11. Check **propagation delays against transport/residence times** and **edges against flow direction/connectivity** — the most defensible evidence in a process setting.
12. Where any historical set-point change/intervention exists, use it to confirm key directions.

**Do not** simply switch to the newest algorithm. PCMCI+ is the best-supported primary choice for your data; LSTE and physics-informed GC are the best-supported *root-cause* tools; VAR-LiNGAM/TiMINo are complements, not replacements. The decisive gain comes from (a) temporal + autocorrelation-aware discovery and (b) physics-based validation — not from any single "best" method.

## Research Gaps and Open Questions
- **Faithfulness under oscillation/collinearity** remains fragile; adjacency-faithfulness relaxations in the latent + autocorrelated case are noted as an open problem (Gerhardus & Runge 2020).
- **Non-stationary and multi-disturbance settings** are under-served: LSTE assumes single-disturbance stationarity; regime-aware CI methods are still maturing.
- **Irregular/mixed-frequency sampling** for CI-based methods is an active area (e.g., 2026 arXiv work on causal discovery for irregular time series).
- **Uncertainty quantification** for constraint-based causal discovery is not fully solved; bootstrap gives heuristics, not calibrated guarantees.
- **Ground-truth-free certification** of causal correctness remains fundamentally unsolved — physics-consistency and stability are the best available proxies, not proofs.
- **Verification limits in this review:** the exact knowledge-fusion mechanism and quantitative case-study metrics of the 2024 physics-informed AIChE paper could not be confirmed from the paywalled full text (abstract, SGC section, and figure captions were verified); all LSTE (2025) details were verified from the open-access paper.

## Caveats
- **No observational method proves causality.** The association → prediction → temporal precedence → causal-discovery → causal-effect ladder requires escalating assumptions. PC/PCMCI recover a Markov-equivalence class under Markov+faithfulness+sufficiency; Granger/TE establish *predictive*/*information-transfer* directionality, not mechanism; LiNGAM/TiMINo need functional-form + noise assumptions; only interventions establish causal effects.
- **Ground-truth-free evaluation cannot certify correctness** — stability and physics-consistency are necessary, not sufficient; all methods can share biases (autocorrelation, non-stationarity, latent confounding).
- **Some cited 2025–2026 works are recent and not broadly replicated** (adaptive-MCI variants, assimilative causal inference, causal-audit frameworks); treat as promising, not settled.
- **The TEP is a simulation benchmark**; its "ground truth" causal file reflects design intent/flowsheet, and real plant data will be noisier, with more missingness and regime change than TEP.

## Bibliography (verified)
- Spirtes, Glymour & Scheines. *Causation, Prediction, and Search*, 2nd ed. MIT Press, 2000.
- Kalisch & Bühlmann. "Estimating high-dimensional directed acyclic graphs with the PC-algorithm." *JMLR* 8:613–636, 2007. https://www.jmlr.org/papers/v8/kalisch07a.html
- Colombo & Maathuis. "Order-independent constraint-based causal structure learning." *JMLR* 15:3921–3962, 2014. https://jmlr.org/papers/v15/colombo14a.html (arXiv:1211.3295)
- Ramsey. "Improving accuracy and scalability of the PC algorithm by maximizing p-value." arXiv:1610.00378, 2016.
- Peters & Bühlmann. "Structural Intervention Distance for Evaluating Causal Graphs." *Neural Computation* 27(3):771–799, 2015. https://doi.org/10.1162/NECO_a_00708 (arXiv:1306.1043); R package: https://cran.r-project.org/package=SID
- Shimizu, Hoyer, Hyvärinen & Kerminen. "A linear non-Gaussian acyclic model for causal discovery." *JMLR* 7:2003–2030, 2006.
- Hyvärinen, Zhang, Shimizu & Hoyer. "Estimation of a Structural Vector Autoregression Model Using Non-Gaussianity." *JMLR* 11:1709–1731, 2010. https://www.jmlr.org/papers/v11/hyvarinen10a.html
- Peters, Janzing & Schölkopf. "Causal Inference on Time Series using Restricted Structural Equation Models" (TiMINo). NeurIPS 26, 2013. https://proceedings.neurips.cc/paper_files/paper/2013/hash/47d1e990583c9c67424d369f3414728e-Abstract.html (arXiv:1207.5136)
- Granger. "Investigating causal relations by econometric models and cross-spectral methods." *Econometrica* 37(3):424–438, 1969.
- Schreiber. "Measuring information transfer." *Phys. Rev. Lett.* 85(2):461, 2000.
- Barnett, Barrett & Seth. "Granger causality and transfer entropy are equivalent for Gaussian variables." *Phys. Rev. Lett.* 103:238701, 2009. https://doi.org/10.1103/PhysRevLett.103.238701 (arXiv:0910.4514)
- Bauer, Cox, Caveness, Downs & Thornhill. "Finding the Direction of Disturbance Propagation in a Chemical Process Using Transfer Entropy." *IEEE Trans. Control Systems Technology* 15(1):12–21, 2007.
- Bauer & Thornhill. "A practical method for identifying the propagation path of plant-wide disturbances." *J. Process Control* 18(7–8):707–719, 2008.
- Yuan & Qin. "Root cause diagnosis of plant-wide oscillations using Granger causality." *J. Process Control* 24(2):450–459, 2014. https://doi.org/10.1016/j.jprocont.2013.11.009
- Duan, Yang, Shah & Chen. "Methods for root cause diagnosis of plant-wide oscillations." *AIChE J.* 60:2019–2034, 2014. https://doi.org/10.1002/aic.14391
- Runge, Nowack, Kretschmer, Flaxman & Sejdinovic. "Detecting and quantifying causal associations in large nonlinear time series datasets" (PCMCI). *Science Advances* 5(11):eaau4996, 2019. https://doi.org/10.1126/sciadv.aau4996
- Runge et al. "Inferring causation from time series in Earth system sciences." *Nature Communications* 10:2553, 2019. https://doi.org/10.1038/s41467-019-10105-3
- Runge. "Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets" (PCMCI+). UAI 2020, PMLR 124:1388–1397. https://proceedings.mlr.press/v124/runge20a.html (arXiv:2003.03685)
- Gerhardus & Runge. "High-recall causal discovery for autocorrelated time series with latent confounders" (LPCMCI). NeurIPS 33, 2020. https://proceedings.neurips.cc/paper/2020/hash/94e70705efae423efda1088614128d0b-Abstract.html (arXiv:2007.01884)
- Runge, Gerhardus, Varando, Eyring & Camps-Valls. "Causal inference for time series." *Nature Reviews Earth & Environment* 4:487–505, 2023. https://doi.org/10.1038/s43017-023-00431-y
- Pamfil et al. "DYNOTEARS: Structure Learning from Time-Series Data." AISTATS 2020, PMLR 108. https://proceedings.mlr.press/v108/pamfil20a.html (arXiv:2002.00498)
- Debeire et al. "Bootstrap aggregation and confidence measures to improve time series causal discovery" (Bagged-PCMCI+). CLeaR 2024, PMLR 236. https://proceedings.mlr.press/v236/debeire24a.html (arXiv:2306.08946)
- Meinshausen & Bühlmann. "Stability Selection." *J. R. Statist. Soc. B* 72(4):417–473, 2010.
- Afyouni, Smith & Nichols. "Effective degrees of freedom of the Pearson's correlation coefficient under autocorrelation." *NeuroImage*, 2019. https://doi.org/10.1016/j.neuroimage.2019.05.011
- Madhusoodanan, Chiplunkar, Puli & Huang. "Physics-informed sparse causal inference for source detection of plant-wide oscillations." *AIChE Journal* 70(4):e18362, 2024. https://doi.org/10.1002/aic.18362
- Chen, Liang, Wang, Yao, Su & Liu. "Lag-Specific Transfer Entropy for Root Cause Diagnosis and Delay Estimation in Industrial Sensor Networks." *Sensors* 25(13):3980, 2025. https://doi.org/10.3390/s25133980
- "Consistent Causal Inference from Time Series with PC Algorithm and its Time-Aware Extension." arXiv:2210.09038.
- Software: `tigramite` (https://github.com/jakobrunge/tigramite), `causal-learn` (https://causal-learn.readthedocs.io), `pcalg` (R), `lingam` (https://lingam.readthedocs.io), `causalnex` (DYNOTEARS).