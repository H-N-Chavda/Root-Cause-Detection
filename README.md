# causal-bench

A from-scratch implementation of the **PC (Peter–Clark)** constraint-based
causal discovery algorithm, run against the three
[CIPCaD-Bench](https://arxiv.org/abs/2208.01529) industrial process cases:
Ultra-Processed Food (UF), UF with Internal Machine Dependencies (UFIMD), and
Tennessee Eastman (TE).

The goal is root-cause analysis on continuous industrial process data: recover
the causal graph over process variables from observational data alone, and score
the recovery against the benchmark's published ground truth. The PC
implementation is the subject under test, not a wrapper around a library.

## Layout

```
src/causal_bench/       the package
  paths.py              every filesystem path; nothing else builds one
  config.py             typed settings loaded from configs/
  algorithms/pc.py      the PC implementation (skeleton, orientation, metrics)
  io/                   dataset and ground-truth loaders
  metrics/, eda/        placeholders for a later phase
  utils/logging.py      logging setup
  cli.py                the causal-bench entry point
configs/default.yaml    alpha, conditioning-set cap, seed, cases, sweeps
data/raw/               input CSVs           (read only)
data/ground_truth/      adjacency matrices   (read only)
results/                run output, git-ignored
tests/                  pytest suite
docs/                   DATA.md, feedback response, working notes, legacy output
references/             papers, presentation build scripts, insight report
notebooks/              empty
```

`data/` is read only. Every generated file goes into a timestamped directory
under `results/`.

## Install

Requires Python 3.11+ (developed on 3.14, see `.python-version`).

```bash
git clone <repo-url> && cd Root-Cause-Detection
make install          # creates .venv, installs -e ".[dev]", sets up pre-commit
```

Equivalent by hand:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run

```bash
make run                        # all three cases plus the sensitivity sweeps
.venv/bin/causal-bench          # the same thing directly
```

Useful flags:

```bash
causal-bench --no-sensitivity                    # headline cases only (~2s)
causal-bench --dataset datasetTE.csv \
             --ground-truth TEGroundTruth.txt    # one ad-hoc case
causal-bench --config configs/default.yaml       # a different config
causal-bench --output-dir /tmp/out               # a different results root
causal-bench --dry-run                           # stdout only, writes nothing
```

Each run creates `results/run-<timestamp>/` containing `results.txt` (the
report) and `run.log`. Progress goes to stderr; the report goes to stdout.

Bare file names given to `--dataset` / `--ground-truth` resolve under
`data/raw/` and `data/ground_truth/`. Paths can be overridden with the
`CAUSAL_BENCH_ROOT`, `CAUSAL_BENCH_DATA_DIR`, `CAUSAL_BENCH_CONFIG_DIR` and
`CAUSAL_BENCH_RESULTS_DIR` environment variables.

Every tunable — significance level, maximum conditioning set size, seed, case
list, sweep grids — lives in `configs/default.yaml`, not in the code.

## Tests and checks

```bash
make test        # pytest with coverage (fails under 75%)
make lint        # ruff check + ruff format --check
make typecheck   # mypy, non-strict
make check       # all three
```

Tests read fixture slices, not the full datasets; the one test that needs the
complete Tennessee variable set is marked `slow`. Run `pytest -m "not slow"` to
skip it.

## Data

Datasets and ground-truth graphs come from Menegozzo, Dall'Alba & Fiorini,
*CIPCaD-Bench*, IEEE CASE 2022. See **[docs/DATA.md](docs/DATA.md)** for shapes,
column names, separators, and the known problems — notably that the Tennessee
ground truth is 33×33 while `datasetTE.csv` has only 31 columns.

## License

See [LICENSE](LICENSE).
