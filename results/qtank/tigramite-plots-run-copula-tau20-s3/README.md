# tigramite.plotting (5.2.10.1) on qtank — what works, what does not (2026-10-02)

Script: `datasets/tigramite_plots.py`. Inputs: stride-3 data (raw + copula), graphs from `run-copula-tau20-s3`.

| Function | Verdict | Limitation observed |
|---|---|---|
| plot_timeseries | fine | no units/legend control beyond `var_units`; one colour for all panels |
| plot_scatterplots | fine | lag-0 pairs only unless `matrix_lags` given per pair; useless for binary inputs (two vertical lines) |
| plot_densityplots | fine | needs seaborn (not in deps; installed now). Binary inputs give 4-blob KDEs |
| plot_lagfuncs | useful | needs `PCMCI.get_lagged_dependencies` first (ParCorr only here); axis label prints `lag τ []`; dots only, no CI bands |
| plot_graph | good | node colour = auto-MCI (tigramite-specific, meaningless for VAR-LiNGAM); lag labels overlap at >3 lags; fixed circular layout, arrows cross heavily with 7 nodes; `vmin/vmax` must be set by hand |
| plot_time_series_graph | unreadable at tau_max=20 | 21 columns x 7 rows, every repeated lag drawn; only usable for tau_max <= ~5 |
| plot_tsg | unusable | low-level; no var names, no sizing args, 21 lags squashed into 350px; meant for internal path plots |
| plot_mediation_graph / _time_series_graph | partial | requires fitting `LinearMediation` on lagged parents, so contemporaneous links are dropped; path label renders `np.int64(1)` (numpy-2 repr bug); empty figure when no path exists (LSTE graph, 0 edges) |
| write_csv | fine | plain edge list, no scores |
| Ground truth | workaround | lag-0 `-->` must be mirrored `<--` or it raises; drew GT at lag 1 because GT carries no lags |

Not available in tigramite at all (covered today by matplotlib/seaborn figures in `results/qtank/figures`):
- side-by-side comparison of several algorithms, or algorithm vs ground truth
- precision/recall/F1/SHD tables, heatmaps, radar, runtime bars
- edge-level TP/FP/FN colouring
- normality / Q-Q / Hotelling T² EDA plots
- any plot for non-tigramite graphs without first converting to its `(N,N,tau+1)` string array (done here via saved `links`)

Status of every call: see `status.md` in this folder.
