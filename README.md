# Root-Cause Detection with the PC Algorithm

Recovering causal structure from industrial process data, so that a detected
fault can be traced back to the variable that caused it.

A causal graph is the middle stage of that objective:

```
process data  →  causal graph  →  fault detected  →  trace upstream  →  root cause
                 ^^^^^^^^^^^^
                 what this repository produces
```

## Method

The PC (Peter–Clark) algorithm recovers a causal graph from observational data
in two phases.

**Phase 1 — skeleton.** Start from a complete graph and remove the edge between
every pair of variables that some conditioning set renders conditionally
independent. Under joint normality this reduces to a partial correlation of
zero, tested with Fisher's z-transform:

```
z = ½ · √(n − |Z| − 3) · ln((1 + ρ̂) / (1 − ρ̂))        p = 2(1 − Φ(|z|))
```

The edge is removed when `p ≥ α`. The separating set is recorded for phase 2.

**Phase 2 — orientation.** For every unshielded triple `a — b — c` where `b` is
absent from the set that separated `a` and `c`, orient `a → b ← c`. Meek's rules
then propagate the consequences, giving a CPDAG. Edges that remain undirected
are genuinely undetermined by the data; where two triples imply opposite
directions on one edge, no direction is claimed.

Parameters: `α = 0.01`, conditioning sets up to size 2.

## Datasets

| Case | Variables | Samples | Reference graph |
|---|---|---|---|
| Ultra-processed food | 17 | 23,132 | 83 relationships |
| …with internal machine dependencies | 17 | 23,132 | 133, including feedback loops |
| Tennessee Eastman | 31 | 1,499 | 32 over 33 variables, 28 recoverable |

The Tennessee reference graph covers two variables that are not present in the
data, so its edges are aligned to the dataset by variable name and recall is
scored against the 28 that are recoverable.

## Results

Skeleton recovery, at `α = 0.01` and conditioning size 2:

| Case | Found | Precision | Recall | F1 | SHD |
|---|---|---|---|---|---|
| Ultra-processed food | 55 | 0.55 | 0.36 | 0.43 | 105 |
| …with machine dependencies | 55 | 0.80 | 0.41 | 0.54 | 102 |
| Tennessee Eastman | 35 | 0.31 | 0.39 | 0.35 | 48 |

Skeleton and orientation are scored separately, since recovering *which*
variables are related and *which way* the influence runs are different tasks.
`SHD` is the edge-level structural Hamming distance.

`Results.txt` carries the full output, including per-case orientation counts and
sweeps over `α`, conditioning size and sample thinning.

## Limitations

- **The samples are not independent.** Both datasets are time series with
  lag-one autocorrelation up to 0.98, so the effective sample size is far below
  `n` and the tests are more confident than the evidence warrants. This is the
  largest single constraint on the results, and the reason thinning the series
  *improves* precision.
- **Tennessee Eastman has converged** at conditioning size 2; the
  ultra-processed-food cases have not, so their scores still depend on where the
  search stops.
- **PC assumes acyclicity**, so the feedback loops in the third reference graph
  cannot be recovered by construction.
- **No unmeasured confounders** are admitted, which a method such as FCI would
  be needed for.

## Layout

```
PC_Algorithm/
  utils.py                    loading, independence testing, PC, evaluation
  main.py                     runs all three cases, writes Results.txt
  test_pc.py                  test suite
  Results.txt                 current output
  Results_previous.txt        earlier output, retained for reference
  *.csv, *GroundTruth.txt     datasets and reference graphs
presentations/
  build_presentation.py       generates PC_Algorithm.pptx
  theme.py content.py         visual system and figures
  analysis.py slides.py       analysis run at build time, slide definitions
Research Papers/              background reading
```

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r PC_Algorithm/requirements.txt

cd PC_Algorithm
../.venv/bin/python main.py              # writes Results.txt
../.venv/bin/python -m pytest test_pc.py -q
```

## Presentation

A 20-slide deck covering the algorithm, its mathematics, and the Tennessee
Eastman case walked end to end — preparation, graph construction, the recovered
graph, its assessment, and next steps.

```bash
cd presentations
../.venv/bin/python build_presentation.py    # writes PC_Algorithm.pptx
```

The build runs the analysis itself rather than carrying transcribed numbers, so
the deck cannot fall out of step with the results.
