# causal-bench

Reusable implementations of causal-discovery algorithms for continuous
process data. The package is algorithm code only: no datasets, no analysis
pipeline, no run harness.

## Algorithms

| Module | What it is |
| --- | --- |
| `algorithms/pc_manual.py` | PC (Peter-Clark), from scratch -- CI tests, skeleton, orientation |
| `algorithms/pc.py` | PC via tigramite |
| `algorithms/pcmci_plus.py` | PCMCI+ via tigramite |
| `algorithms/var_lingam.py` | VAR-LiNGAM via lingam |
| `algorithms/lste.py` | Lagged symbolic transfer entropy, built on tigramite's CMIknn |

All five sit behind the `CausalDiscoveryAlgorithm` interface in
`algorithms/base.py` and are looked up by name through `algorithms/registry.py`.

## Supporting modules

```
src/causal_bench/
  algorithms/     the five implementations, base class, registry
  graph/          CausalGraph representation, conversions, JSON io
  prior/          prior-knowledge YAML -> tigramite link_assumptions / lingam matrix
  scoring/        structural Hamming distance and related scores
  io/             dataset and ground-truth loaders
  utils/logging.py
  paths.py        filesystem paths, overridable by CAUSAL_BENCH_* env vars
  metrics/        placeholder
```

## Install

Python 3.11+ (developed on 3.14, see `.python-version`).

```bash
python3 -m venv .venv
# lingam pins scipy<=1.13.1, which has no wheel for python 3.14. The pin is
# conservative -- both estimators fit and run against scipy 1.18.1 -- so lingam
# and its import-time extras go in without their own dependency resolution.
.venv/bin/pip install --no-deps lingam==1.13.0 graphviz semopy psy pygam autograd
.venv/bin/pip install -e ".[dev]"
```

## Use

```python
from causal_bench.algorithms import available, build

available()                       # the registered names
algo = build("pcmci_plus", tau_max=3)
result = algo.run(data, variable_names=names)
```

`data` is a `(n_samples, n_variables)` float array. `result.graph` is a
`CausalGraph`; `causal_bench.scoring` scores it against a ground-truth matrix.

## Check the install

`smoke_check.py` generates a small synthetic SCM with two planted lagged edges,
runs every algorithm on it, and confirms each one completes and recovers what
was planted. No repository data is needed. Run it after installing, and before
pointing anything at a new dataset.

```bash
.venv/bin/python smoke_check.py          # all five, ~25s
.venv/bin/python smoke_check.py --fast   # skips LSTE, ~7s
```

Exit status is 0 only if every check passes. It also covers the support
modules: graph JSON round-trip, SHD scoring, prior-knowledge YAML loading.

Note that `algorithm.run(...)` never raises -- a failure comes back as an empty
graph with `meta["completed"] == False` -- so any check of your own should
inspect that flag rather than assume a returned graph means success.

## Lint and types

```bash
.venv/bin/ruff check src
.venv/bin/mypy
```
