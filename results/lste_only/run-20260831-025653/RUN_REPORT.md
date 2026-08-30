# Causal discovery run report

Generated 2026-08-31T03:20:47 from commit `e8f02e8` by `causal-bench run`. Seed 0. Dataset `datasetTE.csv` (1499 rows x 31 variables).

## Results

SHD against **gt_projected**, the primary target: the 30-edge latent projection onto the 31 observed variables. Lower is better; an empty graph would score 30.

| Run | SHD | TP | FP | FN | Rev | Unor | Edges | Seconds | Assumptions met |
|---|---|---|---|---|---|---|---|---|---|
| lste_tau3 | 30 | 0 | 0 | 30 | 0 | 0 | 0 | 1434.2 | **no** |

An undirected edge scored against a directed one costs 0.5 rather than 1, because a CPDAG's undirected edge is an abstention, not an error.


## Scoring targets

| Target | Nodes | Edges | What it is |
|---|---|---|---|
| gt_full | 33 | 32 | raw ground truth; reference only, nothing is scored against it |
| gt_induced | 31 | 28 | induced subgraph; every edge touching a latent node dropped |
| gt_projected | 31 | 30 | **primary**; latent projection, the graph the data can support |

Latent nodes, derived from the data-versus-ground-truth reconciliation rather than configured: **X27, X31**. Induction drops `X11 -> X27`, `X19 -> X31`, `X27 -> X20`, `X5 -> X27`; the projection recovers `X11 -> X20`, `X5 -> X20` because a latent mediator leaves its parent connected to its child.


**CPDAG floor.** The full ground truth has 9 v-structures, so its CPDAG has 26 directed and 6 undirected edges. Those 6 are undirectable from observational data by any method, which is why the `_cpdag` targets exist: scoring a CPDAG-returning algorithm against a DAG charges it for a limit it cannot beat. On the projected target the CPDAG is 24 directed and 6 undirected.


## Full score table

| Run | Target | SHD | TP | FP | FN | Reversed | Unoriented |
|---|---|---|---|---|---|---|---|
| lste_tau3 | gt_projected | 30 | 0 | 0 | 30 | 0 | 0 |
| lste_tau3 | gt_projected_cpdag | 30 | 0 | 0 | 30 | 0 | 0 |
| lste_tau3 | gt_induced | 28 | 0 | 0 | 28 | 0 | 0 |
| lste_tau3 | gt_induced_cpdag | 28 | 0 | 0 | 28 | 0 | 0 |

## Assumptions and failures

Every requested run completed.


**Completed but mis-specified.** These ran to completion under assumptions the data violates. The graph exists; whether it means anything is what the note says.

| Run | Violated assumption and the number |
|---|---|
| lste_tau3 | stationarity: ADF calls only 30/31 columns stationary; a kNN density estimate over a drifting series mixes two regimes |
| lste_tau3 | adequate sample size: binding effective n is 36 of 1499 rows; a k-nearest-neighbour estimator needs more effective samples than a parametric test, not fewer |

## Per-algorithm notes


**`lste_tau3`** -- estimator: CMIknn (Kraskov-style k-nearest-neighbour CMI); 2790 independence tests; 548 significant uncorrected, 0 after fdr_bh.

## Prior knowledge

No prior knowledge was supplied.


## Parameters and provenance

| Setting | Value | Source |
|---|---|---|
| seed | 0 | config discovery.seed |
| tau values | 1, 2, 3 | EDA VAR order selection, swept over the full range 1..3 because the criteria disagree (AIC=3, BIC=1, HQIC=2, FPE=3) |
| standardize | yes | EDA preprocessing advice: largest/smallest column standard deviation = 2525.2, above 10.0: an unscaled distance or kernel computation would be dominated by the widest variable |
| undirected weight | 0.5 | config discovery.undirected_weight |
| EDA report | /Users/gyan/Documents/Root-Cause-Detection/results/eda/eda-20260831-005002/eda_report.json | phase 2 handoff |
| config | /Users/gyan/Documents/Root-Cause-Detection/configs/default.yaml | --config |

**Library versions**

| Package | Version |
|---|---|
| tigramite | 5.2.10.1 |
| lingam | 1.13.0 |
| networkx | 3.6.1 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| statsmodels | 0.15.0 |

## Reproducing this run

```bash
causal-bench run --algorithms lste --out results/lste_only --no-prior-knowledge --no-figures
```

Total runtime 1434s.

