| plot | status | error |
|---|---|---|
| ground_truth plot_graph (drawn at lag 1; no lags known) | ok | |
| P_minus/raw plot_timeseries | ok | |
| P_minus/raw plot_scatterplots | ok | |
| P_minus/raw plot_densityplots | ok | |
| P_minus/raw plot_lagfuncs (ParCorr lagged deps) | ok | |
| P_minus/copula plot_timeseries | ok | |
| P_minus/copula plot_scatterplots | ok | |
| P_minus/copula plot_densityplots | ok | |
| P_minus/copula plot_lagfuncs (ParCorr lagged deps) | ok | |
| P_minus/lste plot_graph | ok | |
| P_minus/lste plot_time_series_graph | ok | |
| P_minus/lste write_csv | ok | |
| P_minus/lste plot_tsg (v1 -> h2 path) | ok | |
| P_minus/lste plot_mediation_graph + _time_series_graph | FAIL | RuntimeError: no lagged path between any pair; nothing to draw |
