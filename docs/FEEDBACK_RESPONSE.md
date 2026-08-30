# Response to Week-1 Review Feedback

**Project:** Root-Cause Detection with constraint-based causal discovery
**Data:** CIPCaD-Bench — Ultra-Processed Food (UF), UF with Internal Machine
Dependencies (UFIMD), Tennessee Eastman (TE)
**Prepared:** 19 August 2026

This document answers each of the nine points raised in the review. For every point it
states the question, what the literature review established, the options that were
available, which option is recommended and why, and — where the question cannot yet be
answered — what specifically is blocking it.

---

## Contents

- [How the review was carried out](#how-the-review-was-carried-out)
- [Summary of conclusions](#summary-of-conclusions)
- **Review items**
  - [R1 — Library and implementation audit: stability mechanisms](#r1--library-and-implementation-audit-stability-mechanisms)
  - [R2 — High-performance C++ implementations of PC](#r2--high-performance-c-implementations-of-pc)
- **Action items**
  - [1 — Benchmarking and validation against established baselines](#1--benchmarking-and-validation-against-established-baselines)
  - [2 — SOTA review: PC adaptations for time series](#2--sota-review-pc-adaptations-for-time-series)
  - [3 — Time-series modelling and lag analysis](#3--time-series-modelling-and-lag-analysis)
  - [4 — Comparative empirical evaluation](#4--comparative-empirical-evaluation)
  - [5 — Evaluation without ground truth](#5--evaluation-without-ground-truth)
  - [6 — Error attribution: data scarcity vs algorithmic limits](#6--error-attribution-data-scarcity-vs-algorithmic-limits)
  - [7 — Domain-knowledge sanity checks](#7--domain-knowledge-sanity-checks)
- [Five findings we did not expect](#five-findings-we-did-not-expect)
- [What we could not resolve, and why](#what-we-could-not-resolve-and-why)
- [Recommended next steps](#recommended-next-steps)
- [Bibliography](#bibliography)

---

## How the review was carried out

**Search strategy.** The review was seeded from two directions: the source paper for our
own datasets, and the standard surveys of the field. Citations were then chased in both
directions — backward into what a seed paper cites, forward into what cites it — until a
thread stopped producing papers that changed its conclusion.

**Coverage.** 65 distinct sources were engaged. They are honestly tiered, because "read"
means different things:

| Tier | What it means | Count |
|---|---|---|
| A | Full text read cover to cover | 1 |
| B | Source fetched directly; abstract verbatim plus targeted sections extracted | 10 |
| C | Metadata and specific claims verified; full text not opened | 54 |

No recommendation in this document rests on a Tier C source alone. Every load-bearing
claim is either quoted verbatim from a Tier A/B source or is a number measured from our
own data. All 19 cited URLs were checked and resolve.

**Own measurements.** Several of the review's questions — particularly items 3, 6 and 7 —
cannot be answered from the literature alone; they require measuring our own data. Those
measurements were made read-only, without altering the existing pipeline, and are
reported inline where they answer a question. They are reproducible from the datasets in
the repository.

**A note on scope.** A working prototype was also built to test some of these conclusions
empirically. It has been archived in the local git history (commit `ba59c8f`) and removed
from the working tree, so this document stands on its own as an analysis. Where prototype
results are quoted below, they are labelled as such.

---

## Summary of conclusions

| # | Question | Short answer | Confidence |
|---|---|---|---|
| R1 | What stability mechanism do the libraries use, and which do we lack? | Our skeleton is already correct (PC-stable). Our **orientation** stage is the defect: no maxP/conservative rule, and Meek R4 missing | High |
| R2 | Which C++ PC implementations should we use? | None. They change speed, not answers, and our problem is ~4 orders of magnitude too small | High |
| 1 | How do we benchmark against established baselines? | Against `causal-learn` (what the benchmark itself used), scored under the benchmark's own metric protocol — which is **not** the standard one | High |
| 2 | What is SOTA for time-series PC? | PCMCI+ as primary; LPCMCI for UF, whose causal sufficiency the data authors document as violated | High |
| 3 | How do we analyse lag structure? | Full temporal battery, but `tau_max` must come from plant knowledge — we measured that the data **cannot** resolve its own lags | High on diagnosis |
| 4 | How do we compare candidate algorithms fairly? | Fixed protocol, summary-graph projection, and SID alongside SHD | High |
| 5 | How do we evaluate without ground truth? | Three-layer battery: bootstrap stability, self-compatibility, permutation falsification | High |
| 6 | Is low precision data scarcity or algorithmic limits? | **Different for each dataset** — and we can now give a number rather than an opinion | High |
| 7 | How do we sanity-check edges against physics? | Formal background-knowledge specification; the audit found a real problem in the ground truth | High |

---

## R1 — Library and implementation audit: stability mechanisms

### The question
What do the established libraries actually implement for order-independence and
orientation stability, and which of those mechanisms are we missing?

### What the literature establishes

The canonical reference is **Colombo & Maathuis (JMLR 15:3741–3782, 2014)**, which
identifies the problem precisely:

> "This algorithm is known to be order-dependent, in the sense that the output can depend
> on the order in which the variables are given… We show, however, that it can be very
> pronounced in high-dimensional settings, where it can lead to highly variable results."

The paper proposes a family of fixes, which the libraries then implement:

1. **PC-stable** — freeze each node's adjacency set at the start of a conditioning level
   so that edge deletions within the level cannot change other pairs' conditioning
   candidates. This removes order-dependence from the *skeleton*.
2. **Conservative PC (CPC)** — Ramsey, Zhang & Spirtes, UAI 2006. Tests *every* separating
   set of a triple, and marks the triple "unfaithful" unless the vote is unanimous.
3. **Majority-rule PC (MPC)** — Colombo & Maathuis. Orients as a collider when the middle
   node appears in fewer than half of the separating sets; exactly half marks it
   ambiguous. Introduced because CPC is too strict: the authors note it
   "can be very conservative, in that very few unshielded triples are unambiguous in the
   sample version".
4. **maxP / PC-Max** — Ramsey, arXiv:1610.00378, 2016. Chooses the conditioning set with
   the **highest p-value**, so collider classification is unambiguous by construction. The
   paper explicitly describes it as "avoiding bidirected edges in the orientation of
   colliders", with "no risk of bidirected edges", and reports accuracy "far more
   accurately than any of the compared constraint-based competitors".

Separately, **Perković, Kalisch & Maathuis (UAI 2017, arXiv:1707.02171)** establish that
**Meek rule R4** is required to obtain a *maximally* oriented PDAG once background
knowledge is added — R4 is frequently omitted from implementations.

The reference Python library, `causal-learn` (Zheng et al., JMLR 2024), exposes exactly
these choices: `stable=True` by default, `uc_rule` ∈ {`uc_sepset`, `maxP`,
`definiteMaxP`}, and `uc_priority` for conflict resolution — whose **default is
"prioritize existing colliders"**.

### Audit of our implementation

| Mechanism | Ours | Field standard | Verdict |
|---|---|---|---|
| Order-independent skeleton | Yes — adjacency frozen per level | PC-stable | **Correct** |
| Conditioning sets from both endpoints | Yes | Standard | **Correct** |
| Collider rule | Plain sepset rule | maxP / CPC / MPC available | **Weakest link** |
| Conflict handling | Mark, then orient *neither* direction | Libraries resolve by priority | **Non-standard** |
| Meek rules | R1–R3 only | R1–R4 | **Incomplete** |
| Independence test | Fisher-z on partial correlation | Standard | Correct (but see item 3) |

The consequence is measurable and severe. On UF, **49 of 55 discovered edges end in
orientation conflict**, so the algorithm claims a direction for only 5. Our policy of
abandoning both directions is not the field convention, and it is the worst possible
choice under the benchmark's scoring, which treats a non-directed edge as a missing edge.

### Options and recommendation

| Option | Trade-off |
|---|---|
| Keep plain PC | Status quo; leaves 49/55 UF edges undirected |
| Conservative PC | Most rigorous, but produces an emptier output on an already-conflicted skeleton |
| Majority-rule PC | Order-independent v-structures, explicit ambiguity marking; a sound second choice |
| **maxP (recommended)** | Designed precisely to eliminate this failure mode; available in the reference library as `uc_rule=1` |

**Recommendation:** adopt maxP as the primary collider rule, report majority-rule
alongside it rather than silently choosing one, add Meek R4, and replace the
"claim neither direction" conflict policy with the library convention.

**Caveat carried forward.** A prototype confirmed that maxP resolves the UF pathology but
gives **no benefit on TE**, which has only 12 conflicted edges against UF's 49. The fix
addresses our largest defect, but it is dataset-specific, and that should be stated
whenever it is reported.

---

## R2 — High-performance C++ implementations of PC

### The question
What established high-performance C++ implementations of PC exist, and should we adopt one?

### What the literature establishes

| Implementation | Language | Reported speed-up | Requires |
|---|---|---|---|
| **cuPC-E / cuPC-S** — Zarebavani et al., IEEE TPDS 31(3), 2020 | CUDA | 500× / 1300× over serial CPU; "11 hours to about 4 seconds" | NVIDIA GPU |
| **GPUCSL** — HPI-EPIC, IEEE ICDMW 2022 | Python API over CUDA | 9.5× vs `pcalg`, 19.8× vs `bnlearn` | NVIDIA GPU |
| **PEPC** — HP3C 2023 | C++ multicore | 577× vs Stable, 116× vs Stable.fast, 134× vs Tetrad | CPU only |
| **ParallelPC** — Le et al., arXiv:1502.02454 | R/C++ | 6.7× vs Stable | Multicore CPU |
| `pcalg` | R with C++ core | reference baseline | R |
| Tetrad | Java | reference baseline | JDK 21 |

### Recommendation: survey only, do not adopt

Two independent reasons.

**First, they do not change the answer.** cuPC's own abstract states that it executes
"an order-independent version of PC" — that is, PC-stable, the same statistics we already
implement. These are *scheduling* optimisations, not statistical ones. There is therefore
no accuracy argument for adopting them, and correctness benchmarking can legitimately be
done against a Python reference implementation.

**Second, our problem is far too small to benefit.** Measured on the project machine:

| Case | Variables | Samples | Runtime |
|---|---|---|---|
| Tennessee Eastman, k=2 | 31 | 1,499 | **0.13 s** |
| Ultra-Processed Food, k=2 | 17 | 23,132 | **0.29 s** |
| UF, run to convergence | 17 | 23,132 | **0.43 s** |

The literature above optimises problems that take *hours*. A 1300× speed-up on 0.13 s is
not a contribution.

### When this changes

Two realistic triggers, and they should be stated because the industrial case is real:

1. **Moving to a window causal graph.** A 31-variable process with `tau_max = 20` becomes
   a 651-node problem. Conditional-independence test counts grow combinatorially: measured
   on synthetic data of comparable density, going from 31 to 150 variables took the test
   count from ~5,900 to ~1,970,000 and the runtime from 0.02 s to 16.5 s.
2. **A real plant tag list.** Industrial deployments carry hundreds to thousands of tags,
   where the same curve puts a single run into hours.

For either case, **GPUCSL is the recommended entry point** — it exposes a Python API, so
it can be dropped in without abandoning the existing toolchain, whereas cuPC is a CUDA
codebase to be integrated. The architectural point worth noting for a future handover is
that PC's inner loop is a very large number of small, identical matrix inversions, which
is exactly the workload a GPU absorbs well.

**Blocker.** None of this can be executed or verified on the current machine: it has no
NVIDIA GPU, no R, and only Java 8 (Tetrad 7.x requires JDK 21).

---

## 1 — Benchmarking and validation against established baselines

### The question
How do we benchmark our implementation against established library baselines?

### The most important finding in this review

Our datasets come from **CIPCaD-Bench** (Menegozzo, Dall'Alba & Fiorini, IEEE CASE 2022,
arXiv:2208.01529). That paper publishes PC baselines **on our exact datasets**, and we had
not been comparing against them. More seriously, the numbers are not directly comparable
to ours for two separate reasons.

**Reason one — their metric set is direction-aware, ours is not.** Their equations define

```
FDR = (R + FP)/(TP + FP)     TPR = TP/(TP + FN)     FPR = (R + FP)/(TN + FP)
SHD = UE + UM + R            PR  = TP/(TP + FP)     RE  = TP/(TP + FN)
```

where TP means "detected **with correct direction**" and R means reversed. Their
conversion rules are explicit:

> "We decompose each bidirectional edge into two edges with opposite direction such as
> A ↔ B = A ← B , A → B."
> "Non-directed edges are reported as missing edges since the method cannot distinguish
> the correct causal direction."

We report *skeleton* precision and recall, which ignore direction entirely. These are
different quantities.

**Reason two — their published precision and recall are transposed.** This is not in the
paper; it was found by reconciling their table arithmetically. Multiplying each published
precision by the size of the corresponding **ground truth** yields an integer
true-positive count in all three datasets:

| Case | Precision × \|ground truth\| | ⇒ TP | Implied \|estimate\| = TP / Recall |
|---|---|---|---|
| TE | 0.2500 × 32 | 8.000 → **8** | 58.997 → **59** |
| UF | 0.1566 × 83 | 12.998 → **13** | 54.992 → **55** |
| UFIMD | 0.1955 × 133 | 26.002 → **26** | 55.003 → **55** |

Six integers to three decimal places is not coincidence. Their precision denominator is
the **true-graph** size and their recall denominator is the **estimate** size — the
reverse of the standard definitions their own equations state. Independent corroboration:
the implied estimate size is 55 for both UF and UFIMD, which is correct because those two
cases share a dataset and therefore a skeleton — and our own PC produces a 55-edge
skeleton on UF.

**Any comparison against their table that does not transpose these two quantities will be
wrong.**

### Their published baselines — the numbers to beat

| Dataset | Their PC: Precision | Recall | F1 | SHD | Best method overall |
|---|---|---|---|---|---|
| TE | 0.2500 | 0.1356 | **0.1758** | 69 | **PC** (best F1 on TE) |
| UF | 0.1566 | 0.2364 | 0.1884 | 95 | GES (F1 0.3171) |
| UFIMD | 0.1955 | 0.4727 | 0.2766 | 94 | GES (F1 0.4466) |

Their own conclusion is worth carrying to the review, because it is a direct endorsement
of our algorithm choice for this application:

> "If detecting causal relationships when they are not present in the real process is
> particularly harmful … the PC algorithm can be considered the most suitable due to its
> high precision."

### Recommendation: a two-track benchmark

- **Track A — correctness.** Run `causal-learn`'s PC against ours under identical settings
  and compare **edge by edge**, not by aggregate score. Two implementations can produce
  identical precision while disagreeing on every edge; only edge-level agreement is a real
  correctness check.
- **Track B — comparability.** Re-score every output under CIPCaD-Bench's own metric
  definitions, with the transposition applied, so our numbers can sit beside theirs.

**Why `causal-learn` rather than `pcalg` or Tetrad:** it is what CIPCaD-Bench itself used
for its PC, FCI and GES rows. Matching their toolchain is worth more than matching the
historical reference implementation — and neither R nor JDK 21 is available here anyway.
The residual risk is that we cannot independently verify `causal-learn` against `pcalg`,
so a shared bug in the Python lineage would be invisible to us.

### Preliminary result from the archived prototype

The skeleton replicates **exactly**: our PC on UF produces a 55-edge skeleton, matching
the 55 implied by their published numbers. The determinable part of their TE row — TP = 8
against a 32-edge ground truth, giving Precision = 0.2500 — reproduces exactly.

**Exact replication of their full row is not achievable**, for reasons that are worth
reporting because they reflect on the benchmark rather than on us:

1. Neither their significance level nor their conditioning depth is stated in the paper.
2. Their published table is not self-consistent with their own equations:
   - TPR (0.1695) differs from Recall (0.1356) although both are defined identically.
   - FDR + Precision = 0.7500 + 0.2500 = **1.0000 exactly**, which by their formulas forces
     R = 0, i.e. zero reversed edges. No real PC run on this data produces zero reversals.
   - UF and UFIMD are reported with an **identical FDR of 0.8434** despite having different
     ground truths; that value only reconciles with the UF denominator.

---

## 2 — SOTA review: PC adaptations for time series

### The question
What is the state of the art for constraint-based causal discovery on time series, and
which family fits autocorrelated industrial process data?

### Recommendation
**PCMCI+** as the primary method, **LPCMCI** for UF, and plain PC retained as the i.i.d.
baseline for comparison.

### Why PCMCI+

Runge (UAI 2020) states the problem in exactly our terms:

> "Existing CI-based methods such as the PC algorithm and also common methods from other
> frameworks suffer from low recall and partially inflated false positives for strong
> autocorrelation which is an ubiquitous challenge in time series."

and the method's properties:

> "PCMCI+ improves the reliability of CI tests by optimizing the choice of conditioning
> sets and even benefits from autocorrelation. The method is order-independent and
> consistent in the oracle case."

It also targets our specific situation:

> "…where often time resolutions are too coarse to resolve time delays and strong
> autocorrelation is present."

This matters because our measurements (item 3) show TE's sampling resolution is too coarse
to resolve its delays, and UF's autocorrelation is extreme.

### Why LPCMCI for UF specifically

CIPCaD-Bench's own dataset table records UF as **Sufficiency = No, Un-cofounders = No** —
the dataset authors state that UF has unobserved confounders. PC assumes causal
sufficiency. **On UF we are running an algorithm whose core assumption the data's own
authors documented as false.**

LPCMCI (Gerhardus & Runge, NeurIPS 2020) is the appropriate response, and specifically
beats FCI in our regime:

> "We show that existing causal discovery methods such as FCI and variants suffer from low
> recall in the autocorrelated time series case and identify low effect size of conditional
> independence tests as the main reason."
> "This performance gain grows with stronger autocorrelation."

With UF's lag-1 autocorrelation at 0.98, this is the strongest possible fit.

### Alternatives considered

| Method | Verdict |
|---|---|
| Granger causality / VAR | **Rejected** — cannot express contemporaneous effects, and UF's External stage has a declared 0–1 instance delay |
| FCI | Weaker than LPCMCI under autocorrelation by LPCMCI's own experiments; already benchmarked by CIPCaD (TE F1 0.0909) |
| VARLiNGAM | **Retain as candidate** — non-Gaussian identifiability could orient edges PC must leave undirected; scales well |
| DYNOTEARS | Retain as candidate, with the caveat below |
| CD-NOD | Deferred — addresses non-stationarity, which we have not yet established is present |

### A caveat that must travel with any gradient-based result

Reisach, Seiler & Weichwald (NeurIPS 2021, "Beware of the Simulated DAG!") show that
continuous/gradient methods can exploit **varsortability** — marginal variance increasing
along the causal order — and that on *standardised* data "the same algorithms fail to
identify the ground-truth DAG or its Markov equivalence class". CIPCaD-Bench standardised
UF with a robust scaler. This is a plausible explanation for why the entire gradient family
(NOTEARS, GOLEM, CORL, MCSL) scores poorly on UF in their published table, and it is a
reason to treat DYNOTEARS results with care rather than at face value.

---

## 3 — Time-series modelling and lag analysis

### The question
Implement lag-correlation analysis to capture the dataset's temporal dependencies.

### What we measured

| Dataset | n | lag-1 ACF (median) | Effective sample size | ESS / n | Decorrelation time |
|---|---|---|---|---|---|
| **UF** | 23,132 | 0.981 | **121.5** | **0.53 %** | 346 samples (≈29 h) |
| **TE** | 1,499 | 0.571 | 529.8 | 35.3 % | 5 samples |

Effective sample size is computed with the Bartlett correction,
`n_eff = n / (1 + 2 Σ_k ρ_x(k) ρ_y(k))`, as formalised by **Afyouni, Smith & Nichols
(NeuroImage, 2019)**, who state the consequence directly: ignoring this variance inflation
"will inflate Z-scores and produce excess false positives", and "Fisher's transformation
fails to stabilise the variance".

**The headline number: UF's 23,132 samples carry roughly 120 independent observations.**

### The lag structure is documented but not recoverable

CIPCaD-Bench's production-flow figure annotates every UF causal link with a transport
delay in 5-minute instances — from 0–1 (External) up to 48–49 (Source). For example:

> "an intervention on variable X2 will have an effect on variable X17 with a delay between
> 32 and 48 temporal instances (i.e., 160 and 240 minutes)."

We tested whether those delays can be recovered from the data:

| Method | Upstream variables whose peak \|CCF\| falls inside the declared range |
|---|---|
| Raw cross-correlation | 4 / 16 |
| Prewhitened AR(5) | 5 / 16 |
| Prewhitened AR(20) | 5 / 16 |
| Prewhitened AR(50) | 5 / 16 |

**Why it fails, precisely.** UF's decorrelation time (~346 samples) is about **7× the
largest declared causal delay (49 samples)**. The cross-correlation function is therefore
smeared far wider than the effect it is meant to localise — across all 272 ordered variable
pairs, the raw |CCF| peaks at lag 0 in 45% of cases. Box–Jenkins prewhitening, the textbook
remedy, sharpens only the mid-plant stages and reduces surviving correlations to
|r| ≈ 0.03–0.10, which at ESS ≈ 120 are not significant.

For TE the situation is different but no better: CIPCaD-Bench states plainly that
"it was not possible to determine the delay between the various cause-effect
relationships", so there are no declared lags at all.

### The second, larger problem: the test is mis-calibrated

At α = 0.01, the smallest partial correlation the Fisher-z test can detect:

| Dataset | \|Z\| | Naive (using n) | ESS-corrected | Ratio | Our edges below the honest threshold |
|---|---|---|---|---|---|
| UF | 0 | 0.0339 | **0.4408** | **13.0×** | **42 / 55 = 76 %** |
| UF | 2 | 0.0339 | 0.4441 | 13.1× | |
| TE | 0 | 0.1324 | 0.2208 | 1.7× | 7 / 35 = 20 % |

**Three quarters of the edges we report on UF sit below the threshold an honestly
calibrated test could detect.** They are artefacts of using n = 23,132 where the evidence
supports n ≈ 120.

### Options and recommendation

| Option | Assessment |
|---|---|
| Thin the series | **Rejected as the primary fix.** Our existing sweep shows thinning costs recall heavily (UF recall 0.361 → 0.181 at thin=50) while barely moving precision (0.545 → 0.500) — it discards data instead of correcting the test |
| **ESS-correct the Fisher-z degrees of freedom** | **Recommended.** Cheap, principled, and makes our own algorithm honest. Report both calibrations side by side so the difference is visible |
| **Move to PCMCI+** | **Recommended in parallel.** Conditions the problem away by construction rather than patching the test |

**On `tau_max`:** set it from the benchmark's declared plant delays (`tau_max ≈ 50` for
UF), *not* from the data, because we measured that the data cannot resolve them. For TE,
with no declared delays, set it from the measured decorrelation time (median 5, max 53) →
`tau_max ≈ 20`, and report sensitivity across the choice rather than claiming a value.
Tigramite's own guidance is to "choose a rather large value that includes peaks in the
`get_lagged_dependencies` function".

**The recommended battery:** per-variable ACF/PACF; pairwise cross-correlation; ADF and
KPSS stationarity tests; Ljung-Box; effective sample size; and Box–Jenkins prewhitening
before any CCF is interpreted.

---

## 4 — Comparative empirical evaluation

### The question
Evaluate candidate algorithms on the dataset to determine the most effective method.

### Recommended candidate set

Our PC (current), our PC with the R1 corrections, `causal-learn` PC, GES, FCI, PCMCI+,
LPCMCI, and VARLiNGAM — across all three cases.

### The fairness protocol

Fixed **before** any run, so that nothing is tuned to the answer: an identical α grid, an
identical conditional-independence test family, an identical conditioning cap, identical
preprocessing, and a declared `tau_max` per dataset. Any method-specific parameter is
swept and the sweep reported, never hand-picked.

This matters more than it sounds. In a prototype sweep of 300 configurations on TE, 31 of
them beat the published baseline — so a single favourably-chosen configuration proves
nothing. Robustness across the sweep is the claim worth making.

### The projection rule — easy to get wrong

Time-series methods return a **window graph**; our ground truths are **summary graphs**.
Assaad, Devijver & Gaussier (JAIR, 2022) formalise the distinction between the full-time,
window and summary causal graph, the last of which "gives an overview and can be deduced
from the window causal graph".

The convention to adopt: **an edge X → Y exists in the summary graph if and only if the
window graph contains X(t−τ) → Y(t) for at least one τ in 0…tau_max.** This must be stated
in any results table, because applied asymmetrically it makes time-series methods look
spuriously denser than i.i.d. ones.

### Metrics

Skeleton precision/recall/F1; arrowhead precision/recall; standard edge-level SHD;
CIPCaD-Bench's metric set (transposed, per item 1); and **SID** — Peters & Bühlmann
(*Neural Computation* 27:771–799, 2015) — which counts wrongly-estimated *causal effects*
rather than wrong edges:

> "Instead of DAGs it is also possible to compare CPDAGs, completed partially directed
> acyclic graphs that represent Markov equivalence classes. Since it differs significantly
> from the popular Structural Hamming Distance (SHD), the SID constitutes a valuable
> additional measure."

For a CPDAG, SID yields a lower and an upper bound corresponding to the best and worst DAG
in the equivalence class — which is exactly our situation, since our output is a CPDAG and
not a DAG.

---

## 5 — Evaluation without ground truth

This item contains two distinct questions with different answers.

### 5a — What validates a graph when the truth is unknown?

**Recommendation: a three-layer battery, ordered by what each can falsify.**

**Layer 1 — Bootstrap edge stability.** Debeire, Runge, Gerhardus & Eyring (CLeaR 2024)
provide "a novel bootstrap approach designed for time series causal discovery that
preserves the temporal dependencies and lag structure", aggregating by majority vote.
Their result:

> "Bagged-PCMCI+ improves in precision and recall as compared to its base algorithm
> PCMCI+, at the cost of higher computational demands. These statistical performance
> improvements are especially pronounced in the more challenging settings (short time
> sample size, large number of variables, high autocorrelation)."

That describes our data exactly. Banerjee, Andrews & Kummerfeld (arXiv:2503.15436)
independently find resampling ensembles "protect against adding erroneous edges".

**This is the highest-value single addition available to us**, because it converts every
edge into an edge *with a confidence* — which is what a root-cause user actually needs. An
operator asked to act on a fault trace needs to know which links are solid.

**Layer 2 — Self-compatibility.** Faller, Chennuru Vankadara, Mastakouri, Locatello &
Janzing (AISTATS 2024):

> "while statistical learning seeks stability across subsets of data points, causal
> learning should seek stability across subsets of variables."

Run the algorithm on subsets of variables and check the resulting graphs are mutually
compatible. The authors are explicit about the limit: "passing such compatibility tests is
only a necessary criterion for good performance". It falsifies; it never confirms.

**Layer 3 — Permutation-based falsification.** Eulig, Mastakouri, Blöbaum, Hardt & Janzing
(AAAI 2025). They identify the gap the other methods leave:

> "Existing metrics provide an absolute number of inconsistencies between the graph and the
> observed data, and without a baseline, practitioners are left to answer the hard question
> of how many such inconsistencies are acceptable or expected."

and close it by building a baseline through node permutation:

> "we derive an interpretable metric that captures whether the graph is significantly
> better than random."

Their validation is exactly the property we need: "the true graph is not falsified by our
metric, whereas the wrong graphs given by a hypothetical user are likely to be falsified."

**Rejected alternatives.** Predictive accuracy (MSE) alone — a purely associational model
predicts well while being causally wrong. Raw counts of conditional-independence
violations without a permutation baseline — uninterpretable, per Eulig et al. above.

### 5b — Does our graph explain the data better than the assumed ground truth?

**Answer: yes, this is answerable — with one precise caveat that must be stated first.**

**The caveat is a theorem, not a tuning problem.** Any score computed purely from
observational data is Markov-equivalence-invariant. Chickering (JMLR 3, 2002) and the
score-equivalence literature establish that two DAGs in the same equivalence class induce
the same likelihood and therefore receive the same BIC. **BIC can never adjudicate within
a Markov equivalence class.** Any proposal to "just use BIC" to compare two graphs must
survive this objection.

**But that is not the question being asked.** Our predicted CPDAG and the benchmark's
assumed ground truth lie in *different* equivalence classes — they have different
skeletons entirely; on TE, 24 of our 35 edges are not in the ground truth at all. **Across
equivalence classes, penalised likelihood is a valid comparison.**

The recommended protocol:

1. Fit both graphs as linear-Gaussian structural equation models — each node regressed on
   its parents.
2. Compare **held-out log-likelihood** on a **temporally blocked** split — never a random
   split, because the autocorrelation would leak between train and test — and compare BIC.
3. Run the Eulig permutation test on both graphs and compare their p-values.
4. Report the ESS-corrected count of graph-implied conditional independences that the data
   violates, for both graphs.

If our graph wins on all four, that is a defensible claim that the benchmark's ground
truth is not the best explanation of its own data. Given that the UFIMD ground truth is
cyclic and the UF ground truth omits within-machine links by construction, this is a
realistic outcome rather than a fishing expedition.

A recent framing worth citing in support: "Effect-Level Validation for Causal Discovery"
(arXiv:2602.08340, 2026) argues that "graph-level metrics alone are inadequate proxies for
causal reliability" and evaluates by identifiability, stability and falsification instead.

---

## 6 — Error attribution: data scarcity vs algorithmic limits

### The question
Is our low precision/recall caused by sample-size constraints or by algorithmic
limitations?

This is the item where we can give a numeric verdict rather than an opinion — **and the
answer is different for each dataset.**

### The method

The literature supplies the ingredients but not the recipe, so we constructed one:

1. **Identifiability ceiling.** Compute the true CPDAG by applying v-structure detection
   and Meek rules R1–R4 to the ground-truth DAG. With a perfect conditional-independence
   oracle, PC provably returns exactly this — so it is an upper bound **no amount of data
   can beat**. Anything below it is our problem; the gap to 100% is not. (Using an oracle
   CI test as a diagnostic is established practice — PyWhy's `dodiscover` ships an
   `Oracle` test for this purpose.)
2. **Effective sample size and minimum detectable effect** (item 3).
3. **Fraction of reported edges below the honest detection threshold.**
4. **Bootstrap edge stability** — high stability with a wrong graph implies a structural or
   assumption failure; low stability implies finite-sample variance.

### The results

| Case | Orientation ceiling | We achieve | Gap attributable to us | ESS / n | Edges below honest threshold |
|---|---|---|---|---|---|
| **UF** | **91.6 %** (76 of 83 edges orientable) | 3.6 % | **87.9 points** | 0.53 % | 76 % |
| **TE** | **78.6 %** (22 of 28 orientable) | 14.3 % | 64.3 points | 35.3 % | 20 % |
| **UFIMD** | *undefined* — ground truth is cyclic | — | — | 0.53 % | 76 % |

### Verdict per dataset

**UF — dominated by test mis-calibration, then by our orientation stage.** The effective
sample size is 121 against a nominal 23,132, and three quarters of the reported edges could
not be detected by an honestly calibrated test. Separately, only 8.4% of the true edges are
unorientable *in principle*, so the orientation collapse is a defect we can fix (item R1),
not a limit we must accept. **Both causes are ours to address; neither is "we need more
data" in the naive sense — we need more *independent* data, which is a different problem.**

**TE — not an autocorrelation problem.** Only 20% of its edges fall below the honest
threshold. Its real problem is *what* it finds: **9 of 11 true positives (82%) are
actuator/sensor pairs on the same physical stream** — a valve and its own flow meter, which
are near-deterministically related and therefore trivial to detect. The 17 missed edges are
the actual process topology: reactor feed rate → temperature, pressure → temperature,
stripper level → pressure/underflow/temperature, separator temperature → level/pressure/
underflow. **We are recovering the instrumentation, not the chemistry.** That is a far more
useful diagnosis than "precision is low".

**UFIMD — the estimand does not exist.** See the blockers section.

---

## 7 — Domain-knowledge sanity checks

### The question
Validate both directed and undirected edges against physical ground realities.

### Method

Meek (1995) provides the framework for incorporating required and forbidden edges;
Perković et al. (2017) show the result must then be refined to a maximally oriented PDAG
using all four Meek rules. For tiered knowledge specifically, Bang & Didelez
(arXiv:2306.01638) establish that tiered background knowledge "only affects directions,
and only where there exists an adjacency between nodes in two different tiers".

We can apply this concretely because CIPCaD-Bench's variable table gives the full TE
semantics: 22 measurements (XMEAS, X1–X22) and 11 manipulated variables (XMV, X23–X33),
with descriptions and units. That yields an unambiguous physical tier: **manipulated
variables are actuators; measured variables are sensors.**

### The audit — and a real problem in the ground truth

We checked the nine actuator/sensor pairs on the same physical stream (e.g. XMV(1)
"D Feed Flow" valve against XMEAS(2) "D Feed" measured flow):

| Actuator | Sensor | Stream | Found in our skeleton | Ground-truth direction |
|---|---|---|---|---|
| X23 | X2 | D Feed (stream 2) | yes | X2 → X23 |
| X24 | X3 | E Feed (stream 3) | yes | X3 → X24 |
| X25 | X1 | A Feed (stream 1) | yes | X1 → X25 |
| X26 | X4 | A & C Feed (stream 4) | yes | X4 → X26 |
| X28 | X10 | Purge (stream 9) | yes | X10 → X28 |
| X29 | X14 | Sep Pot Liquid (stream 10) | yes | X14 → X29 |
| X30 | X17 | Stripper Liquid Product (11) | yes | X17 → X30 |
| X32 | X21 | Reactor Cooling Water | yes | X21 → X32 |
| X33 | X22 | Condenser Cooling Water | yes | X22 → X33 |

Three conclusions follow.

**First, a genuine strength: our skeleton finds all nine** — 100% recall on the subset
where the physical relationship is certain.

**Second, the ground truth orients every one of them sensor → actuator.** Physically, a
valve position causes the flow through it. But in a *closed-loop* plant the controller
reads the measurement and sets the valve, so sensor → actuator is the **control**
direction. Both directions are real: **these pairs are feedback loops.** The benchmark
resolves them one way; PC has no way to infer that convention from data. The ground truth
is internally consistent about this — it uses sensor → actuator for same-stream pairs but
actuator → sensor for cross-effects (X32 "Reactor Cooling Water Flow" → X9 "Reactor
Temperature").

**Third, these nine pairs dominate TE's entire orientation score.** TE has only 11 true
positives and nine of them are these pairs. Declaring the control convention as required
background knowledge would take correct arrowheads from **4 to 10** — arrowhead recall
0.143 → 0.357, precision 0.444 → 0.909 — without changing the skeleton at all.

A prototype confirmed the asymmetry starkly: the control direction agrees with the ground
truth on **9 of 9** pairs; the physical process direction agrees on **0 of 9**.

**This must be reported honestly.** Injecting that convention improves the score, but eight
of the resulting nine true positives are *forced by the knowledge* rather than discovered.
It is what a deployed system that knows the plant's control layout would achieve; it is not
a measure of causal-discovery skill. Both numbers belong in any report.

### A second domain check that vindicates our pipeline

The two TE variables absent from our CSV — X27 = XMV(5) "Compressor Recycle Valve" and
X31 = XMV(9) "Stripper Steam Valve" — are missing because CIPCaD-Bench "removed the
variables with null variance": they are constant in the normal steady state. The four
ground-truth edges our loader drops (X5→X27, X11→X27, X19→X31, X27→X20) are therefore
genuinely unrecoverable rather than a data-handling bug. Our existing alignment logic is
correct.

### For UF

The equivalent tier structure comes from the published production flow:
Source (X1) → Mixer (X2–X8) → Dryer_1 (X9,X10) → Dryer_2 (X11,X12) → Dryer_3 (X13,X14) →
Product (X17), with External (X15,X16) uncontrolled. This is a clean tiered
background-knowledge specification: edges may not run backwards through the production flow.

---

## Five findings we did not expect

1. **The benchmark's published precision and recall are transposed.** Established by
   arithmetic that resolves to integers on all three datasets. Any comparison that does
   not account for it is invalid.
2. **UF's 23,132 samples are worth about 120 independent observations** — an effective
   sample size of 0.53% of nominal.
3. **UF's orientation failure is ours, not an identifiability limit.** The ceiling is
   91.6%; we achieve 3.6%.
4. **Tennessee Eastman recovers its instrumentation, not its chemistry** — 82% of true
   positives are valve/flow-meter pairs.
5. **The benchmark's own results table is not internally self-consistent**, in three
   separate ways, which caps how precisely anyone can replicate it.

---

## What we could not resolve, and why

These are the places where the literature does not settle the question for our setting.
They are the honest limits of this review.

**B1 — The UFIMD ground truth is cyclic, so PC's estimand does not exist.**
It contains 25 reciprocal pairs among 108 skeleton pairs (23%). A DAG cannot represent a
2-cycle, so no acyclic method can exceed the resulting ceiling — and unlike UF and TE we
cannot even compute an identifiability ceiling, because d-separation is undefined on a
cyclic graph. *Proposal:* score UFIMD against the acyclic subgraph as the honest target,
and report the 25 cyclic pairs separately as structurally out of scope. *Blocker:* we found
no maintained implementation of a cyclic constraint-based method (CCD, LLC) comparable in
quality to `causal-learn` or `tigramite`. **If the review knows of one, that unblocks this
case directly.**

**B2 — UF violates causal sufficiency by the benchmark's own admission, and the fix breaks
the comparison.** The correct response is FCI/LPCMCI, but those return PAGs containing
bidirected edges, and CIPCaD-Bench's protocol collapses them: "Edges arising from
un-cofounded variables (as in the case of the FCI algorithm) were approximated as directed
edges." That approximation discards exactly the information a latent-variable method exists
to produce. Their FCI row on UF scores F1 = 0.0217, the worst of all ten methods, which we
suspect is an artefact of the collapse rather than a property of FCI. **We currently cannot
score a latent-variable method fairly against this benchmark.** Resolving it needs a
PAG-aware metric.

**B3 — The UF time lags are not statistically identifiable from the UF data.** Measured
above: neither raw nor prewhitened cross-correlation recovers more than 5 of 16 declared
ranges. We must take `tau_max` on the benchmark's plant authority rather than validate it.
*Risk:* if the declared delays are wrong, we have no way to detect it.

**B4 — TE has no declared lags at all.** Any `tau_max` we choose is a judgement call
justified only by the measured decorrelation time. We will report sensitivity across the
choice rather than claim a value.

**B5 — The control-versus-process direction is a modelling choice we cannot make from
data.** For the nine TE actuator/sensor pairs, both directions are physically real; the
plant contains the loop. Encoding the control convention as background knowledge sharply
improves our score, but it is fitting to the benchmark's convention, and we should say so
plainly if we do it. **This needs a decision from the review or a plant engineer, not from
us.** The methodologically cleaner alternative — modelling them as 2-cycles — returns us to
B1.

**B6 — UF may be underpowered for this task at any resolution.** With ESS ≈ 120, the
smallest partial correlation detectable at α = 0.01 is 0.44. Most genuine process couplings
across a multi-stage plant are weaker than that. If so, **no choice of algorithm rescues
UF**, and the honest conclusion is that this dataset cannot support edge-level causal
discovery at the confidence the application needs — only the strongest couplings are
recoverable. *What would resolve it:* longer records at the same resolution, or data
containing genuine interventions or faults, which the TE simulator can generate but the
supplied steady-state extract does not contain. **This is the single largest open risk to
the project's headline objective.**

**B7 — We cannot execute the canonical reference implementations on this hardware.**
No NVIDIA GPU (cuPC, GPUCSL), no R (`pcalg`, `bnlearn`), Java 8 only (Tetrad needs JDK 21).
We mitigate by benchmarking against `causal-learn`, which is what CIPCaD-Bench itself used —
but a shared bug in the Python lineage would be invisible to us.

---

## Recommended next steps

Ordered by evidence-weighted value rather than effort.

| # | Action | Justification |
|---|---|---|
| 1 | ESS-corrected Fisher-z, reporting both calibrations | Addresses the single largest measured defect — 76% of UF edges |
| 2 | Replace the conflict policy; add maxP colliders and Meek R4 | Closes a measured 87.9-point orientation gap on UF |
| 3 | Score everything under CIPCaD-Bench's protocol, transposed | Makes every number we quote comparable to a published baseline |
| 4 | Background-knowledge module (TE actuator tiers, UF production flow) | Takes TE correct arrowheads from 4 to 10 |
| 5 | Bootstrap edge stability with a time-series-appropriate bootstrap | Turns every edge into an edge with a confidence — what the application needs |
| 6 | PCMCI+ and LPCMCI with summary-graph projection | Addresses the assumption violations directly |
| 7 | Validation battery: self-compatibility, permutation falsification, held-out likelihood | Answers item 5 in production, where no ground truth exists |
| 8 | SID alongside SHD | Measures wrong *causal effects*, not just wrong edges |

**Two decisions we are requesting from the review**, because they are not ours to make:

- **B5** — should same-stream actuator/sensor pairs be encoded in the control direction
  (matching the benchmark, improving the score) or treated as the feedback loops they
  physically are (methodologically cleaner, unscorable)?
- **B1** — is there a maintained cyclic causal-discovery implementation we should be using
  for UFIMD?

---

## Bibliography

Sources are grouped by the question they bear on. Depth tier in brackets.

**Our datasets**
- **[A]** Menegozzo, G., Dall'Alba, D., Fiorini, P. "CIPCaD-Bench: Continuous Industrial Process datasets for benchmarking Causal Discovery methods." *IEEE CASE*, 2022. arXiv:2208.01529
- **[C]** Downs, J., Vogel, E. "A plant-wide industrial process control problem." *Computers & Chemical Engineering* 17(3):245–255, 1993
- **[C]** Chen, X., Wang, J., Ding, S. X. "Complex system monitoring based on distributed least squares method." *IEEE T-ASE* 18(4):1892–1900, 2021 — source of the TE ground-truth graph
- **[C]** Menegozzo, G. et al. "Causal interaction modeling on ultra-processed food manufacturing." *IEEE CASE*, 2020 — the UF plant description

**Order-independence and orientation (R1)**
- **[B]** Colombo, D., Maathuis, M. H. "Order-independent constraint-based causal structure learning." *JMLR* 15:3741–3782, 2014. arXiv:1211.3295
- **[B]** Ramsey, J. "Improving Accuracy and Scalability of the PC Algorithm by Maximizing P-value." arXiv:1610.00378, 2016
- **[C]** Ramsey, J., Zhang, J., Spirtes, P. "Adjacency-Faithfulness and Conservative Causal Inference." *UAI*, 2006
- **[C]** Meek, C. "Causal Inference and Causal Explanation with Background Knowledge." *UAI*, 1995
- **[C]** Perković, E., Kalisch, M., Maathuis, M. H. "Interpreting and using CPDAGs with background knowledge." *UAI*, 2017. arXiv:1707.02171
- **[B]** Zheng, Y. et al. "causal-learn: Causal Discovery in Python." *JMLR*, 2024. arXiv:2307.16405
- **[C]** Spirtes, P., Glymour, C., Scheines, R. *Causation, Prediction, and Search.* MIT Press, 2000

**High-performance implementations (R2)**
- **[B]** Zarebavani, B., Jafarinejad, F., Hashemi, M., Salehkaleybar, S. "cuPC: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU." *IEEE TPDS* 31(3), 2020. arXiv:1812.08491
- **[C]** "GPUCSL: GPU-Based Library for Causal Structure Learning." *IEEE ICDMW*, 2022
- **[C]** "PEPC: Parallel and Extensible PC Implementation for Causal Structure Learning." *HP3C*, 2023. doi:10.1145/3606043.3606054
- **[C]** Le, T. et al. "A fast PC algorithm for high dimensional causal discovery with multi-core PCs." arXiv:1502.02454

**Time-series causal discovery (items 2, 3)**
- **[B]** Runge, J. "Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets." *UAI*, 2020. arXiv:2003.03685 — PCMCI+
- **[B]** Gerhardus, A., Runge, J. "High-recall causal discovery for autocorrelated time series with latent confounders." *NeurIPS*, 2020. arXiv:2007.01884 — LPCMCI
- **[B]** Runge, J. et al. "Detecting and quantifying causal associations in large nonlinear time series datasets." *Science Advances* 5(11):eaau4996, 2019 — PCMCI
- **[C]** Assaad, C. K., Devijver, E., Gaussier, E. "Survey and Evaluation of Causal Discovery Methods for Time Series." *JAIR*, 2022
- **[C]** Runge, J. et al. "Causal inference for time series." *Nature Reviews Earth & Environment* 4:487–505, 2023
- **[C]** "Causal Discovery from Temporal Data: An Overview and New Perspectives." *ACM Computing Surveys*, 2025. doi:10.1145/3705297
- **[C]** Pamfil, R. et al. "DYNOTEARS: Structure Learning from Time-Series Data." *AISTATS*, 2020. arXiv:2002.00498

**Autocorrelation and independence testing (item 3)**
- **[C]** Afyouni, S., Smith, S., Nichols, T. "Effective degrees of freedom of the Pearson's correlation coefficient under autocorrelation." *NeuroImage*, 2019
- **[C]** Averin, P., Moysiadis, T., Katakis, I. "Conditional Independence Tests for Constraint-Based Causal Discovery: A Survey." arXiv:2608.11156, 2026

**Evaluation without ground truth (item 5)**
- **[B]** Faller, P. M., Chennuru Vankadara, L., Mastakouri, A. A., Locatello, F., Janzing, D. "Self-Compatibility: Evaluating Causal Discovery without Ground Truth." *AISTATS*, 2024. arXiv:2307.09552
- **[B]** Eulig, E., Mastakouri, A. A., Blöbaum, P., Hardt, M., Janzing, D. "Toward Falsifying Causal Graphs Using a Permutation-Based Test." *AAAI*, 2025. arXiv:2305.09565
- **[B]** Debeire, K., Runge, J., Gerhardus, A., Eyring, V. "Bootstrap aggregation and confidence measures to improve time series causal discovery." *CLeaR*, 2024. arXiv:2306.08946
- **[C]** Chickering, D. M. "Optimal Structure Identification With Greedy Search." *JMLR* 3, 2002 — score equivalence
- **[C]** Banerjee, R., Andrews, B., Kummerfeld, E. "An extensive simulation study evaluating the interaction of resampling techniques across multiple causal discovery contexts." arXiv:2503.15436, 2025
- **[C]** "Effect-Level Validation for Causal Discovery." arXiv:2602.08340, 2026

**Benchmarking and metrics (items 1, 4)**
- **[B]** Peters, J., Bühlmann, P. "Structural Intervention Distance (SID) for Evaluating Causal Graphs." *Neural Computation* 27:771–799, 2015. arXiv:1306.1043
- **[C]** Reisach, A., Seiler, C., Weichwald, S. "Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game." *NeurIPS*, 2021. arXiv:2102.13647
- **[C]** Stein, G., Shadaydeh, M., Blunk, J., Penzel, N., Denzler, J. "CausalRivers — Scaling up benchmarking of causal discovery for real-world time-series." *ICLR*, 2025. arXiv:2503.17452

**Background knowledge and industrial root-cause analysis (item 7)**
- **[C]** Bang, C. W., Didelez, V. "Do we become wiser with time? On causal equivalence with tiered background knowledge." arXiv:2306.01638
- **[C]** "Constraint-based causal discovery with tiered background knowledge and latent variables in single or overlapping datasets." arXiv:2503.21526, 2025
- **[C]** Vuković, M., Thalmann, S. "Causal Discovery in Manufacturing: A Structured Literature Review." *JMMP* 6(1):10, 2022
- **[C]** Wadhwa, S., Dong, R. "On the Sample Complexity of Causal Discovery and the Value of Domain Expertise." arXiv:2102.03274

---

*One correction for the record: an earlier draft of this analysis asserted that our UF
output "would score close to zero" under the benchmark's protocol. That holds only under
one of two possible readings of a conflicted edge — measured, the two readings give
F1 = 0.0682 and F1 = 0.3118. The benchmark does not specify which applies, so both should
be reported.*
