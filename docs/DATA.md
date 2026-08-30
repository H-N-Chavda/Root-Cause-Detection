# Datasets and ground truth

Everything under `data/` is read only. Paths are resolved by
`causal_bench.paths`; no other module builds one.

## Provenance

All five files come from **CIPCaD-Bench** (Menegozzo, Dall'Alba & Fiorini,
*CIPCaD-Bench: Continuous Industrial Process datasets for benchmarking Causal
Discovery methods*, IEEE CASE 2022, [arXiv:2208.01529](https://arxiv.org/abs/2208.01529)).
Two of the three benchmark cases share one CSV: UF and UFIMD differ only in
which ground-truth graph they are scored against.

## `data/raw/`

| File | Rows | Cols | Separator | Columns | Notes |
|---|---|---|---|---|---|
| `datasetTE.csv` | 1499 | 32 | `,` | unnamed row counter, then `X1`–`X26`, `X28`, `X29`, `X30`, `X32`, `X33` | 31 usable variables; all `float64` except the counter (`int64`). No NaNs, no constant columns. |
| `DatasetUF.csv` | 23132 | 18 | `,` | `DateTime`, then `X1`–`X17` | 17 usable variables, all `float64`. `DateTime` is `YYYY-MM-DD HH:MM:SS` at 1-minute resolution. No NaNs, no constant columns. |

`load_dataset` drops the leading counter (`Unnamed: 0`) and the `DateTime`
column before analysis and records both in its `dropped` return value.

## `data/ground_truth/`

All three are square binary adjacency matrices, **tab separated**, values in
`{0, 1}`, no header row. `[i, j] == 1` means `X(i+1) -> X(j+1)`.

| File | Shape | Edges | Reciprocal pairs | Used with |
|---|---|---|---|---|
| `TEGroundTruth.txt` | 33×33 | 32 | 0 | `datasetTE.csv` |
| `UFGroundTruth.txt` | 17×17 | 83 | 0 | `DatasetUF.csv` |
| `UFIMDGroundTruth.txt` | 17×17 | 133 | 25 | `DatasetUF.csv` |

## Known problems

**TE ground truth is wider than the TE data.** `TEGroundTruth.txt` is 33×33 but
`datasetTE.csv` carries only 31 variables: **`X27` and `X31` are absent from the
CSV**, and both carry edges in the ground truth. The four unrecoverable edges are
`X5 -> X27`, `X11 -> X27`, `X27 -> X20` and `X19 -> X31`. `load_ground_truth`
aligns by variable name (`X<k>` → row/column `k-1`), takes the 31×31 sub-matrix,
and reports the dropped edges rather than scoring against edges that cannot
possibly be found: **32 edges in the file, 28 usable.** Recall computed against
the full 32 is wrong by construction.

**UFIMD contains feedback loops.** 25 of its 133 edges are reciprocal pairs
(`i -> j` and `j -> i`). PC assumes acyclicity and has no CPDAG representation
for a 2-cycle, so these are unrecoverable in principle. `compute_metrics` scores
an undirected edge over such a pair as correct (SHD 0) and charges 1 for either
a single orientation or a missing adjacency.

**Both datasets are autocorrelated time series.** Lag-1 autocorrelation reaches
0.96, which violates the i.i.d. assumption behind the Fisher-z test and makes
nominal p-values anti-conservative. The test is not changed; the CLI's
sensitivity sweep measures the effect by thinning rows (`sensitivity.thin_sweep`
in `configs/default.yaml`).

**Published baselines are not directly comparable.** The CIPCaD-Bench paper uses
a direction-aware protocol in which an undirected edge counts as missing, so the
skeleton precision/recall reported here is not the same quantity. `UNVERIFIED`
for any comparison not made under that protocol.
