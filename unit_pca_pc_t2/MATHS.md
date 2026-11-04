# MATHS: unit-wise PCA → PC → causation-based T²

Paper: Li, Jin & Shi (2008), J. Quality Technology 40(1) 46-58 (`references/Literature/T^2 Decomposition.pdf`). Report: `references/Previous Project Reports/Work so far (report).pdf`. TA code: `references/CodeByTA/`. "Project" = in none of these. Window: Jul–Oct 2025 (`config.py`).

| # | Quantity | Equation | Symbols | Source | Code |
|---|---|---|---|---|---|
| 0a | Time grid | outer join of all tags on time, reindex to a complete 1-min grid (no resampling); repeated timestamps keep the first | `GRID_FREQ` = 1min | TA notebook, `load_parquet_folder` | [helpers.py:436](helpers.py#L436) |
| 0b | Gap filling | drop tags > 10 % missing; then `interpolate(method='spline', order=3, limit_direction='both')`, ffill, bfill | `MAX_MISSING`, `SPLINE_ORDER` | TA notebook, `clean_and_impute` | [helpers.py:730](helpers.py#L730), [:696](helpers.py#L696) |
| 1 | flare_net | $F_t = f_{1,t} + f_{2,t} - r_t$, each input gap-filled by row 0b first (recovery included) | $f_1, f_2$: flare mass flows; $r$: recovered gas (t/h), tags in `.env` | TA notebook (Flare_Flowrate) | [helpers.py:475](helpers.py#L475), [:476](helpers.py#L476) |
| 2 | Flare threshold | `FLARE_USE_MAD_THRESHOLD` = False: $\tau$ = `FLARE_THRESHOLD` (1.5 t/h); True: $\tau = \operatorname{med}(F) + K \cdot 1.4826\,\operatorname{med}\lvert F - \operatorname{med}(F)\rvert$ | $K$ = `FLARE_MAD_K` = 3 | Project | [helpers.py:509](helpers.py#L509) |
| 3 | Runs above $\tau$ | minutes with $F_t > \tau$ at most $g\Delta$ apart form one run | $g$ = `EVENT_GAP` = 5; $\Delta$: grid step | Project | [helpers.py:514](helpers.py#L514) |
| 4 | Flare event | a run with $t^{end} - t^{start} + \Delta \ge 30$ min; onset $t_e$ = first minute; shorter runs are kept out of NOC only | `FLARE_MIN_DURATION_MIN` = 30 | Project | [helpers.py:519](helpers.py#L519) |
| 5 | Pre-flare window | $W_e = \{t : t_e - n\Delta \le t < t_e\} \setminus \text{flare rows}$, $n = \operatorname{round}(30\,\text{min}/\Delta)$ | $t_e$: onset of event $e$ | Project | [helpers.py:612](helpers.py#L612), [:636](helpers.py#L636) |
| 6 | NOC rows | $t < t_0 + 0.6\,(t_N - t_0)$, $t \notin [t_e - 2\text{h},\, t^{end}_e + 2\text{h}]$, $t \notin$ short runs | $t_0, t_N$: first/last row | Project | [helpers.py:585](helpers.py#L585) |
| 7 | Tag standardization | $z_{t,i} = (x_{t,i} - \bar{x}_i)/s_i$ | $\bar{x}_i, s_i$: NOC mean, std (ddof 1) of tag $i$ | p. 50 ($\mu = 0$, unit variance) | [helpers.py:761](helpers.py#L761) |
| 8 | Unit-wise PCA | $Z_u^{NOC} - \mathbf{1}\bar{z}_u^\top = U \Lambda^{1/2} V^\top$; $y_{t} = V_{:,1:x}^\top (z_{t,u} - \bar{z}_u)$ | $u$: unit; $x$ = `N_PCS_PER_UNIT` or `VAR_EXPLAINED_THRESHOLD`, capped at rank $\lambda_k > 10^{-6}\lambda_1$ | Project | [helpers.py:858](helpers.py#L858), [:806](helpers.py#L806) |
| 9 | PC-score standardization | $\tilde{y}_{t,k} = (y_{t,k} - \bar{y}_k)/s_k$ | NOC mean/std of score $k$ | p. 50 | [helpers.py:863](helpers.py#L863) |
| 10 | PC algorithm | Fisher-Z CI tests (= pcalg `gaussCItest`), stable PC, on the PCs only | $\alpha$ = `PC_ALPHA` = 0.01 | p. 49; TA R script | [helpers.py:988](helpers.py#L988), [:1012](helpers.py#L1012) |
| 11 | CPDAG → DAG | Dor-Tarsi: remove a sink $x$ whose undirected neighbours are adjacent to all of $\operatorname{adj}(x)$; orient into $x$ | | Dor & Tarsi (1992) (= pcalg `pdag2dag`) | [helpers.py:1021](helpers.py#L1021), [:1078](helpers.py#L1078) |
| 12 | Final graph | DAG among PCs $\cup \{Y_j \to F\ \forall j\}$ (added, not learned) | $F$: flare_net | Project | [helpers.py:1100](helpers.py#L1100) |
| 13 | NOC correlation | $R = \frac{1}{m - 1}\tilde{Y}_0^\top \tilde{Y}_0$ | $\tilde{Y}_0$: NOC rows of the $p$ PCs; $m$: NOC rows | p. 49 (sample correlation) | [helpers.py:1134](helpers.py#L1134) |
| 14 | Path coefficients | $\beta_j = R_{PP}^{-1} R_{Pj}$ | $P = \mathbf{PA}(Z_j)$: parents of PC $j$ among PCs | p. 49 | [helpers.py:1200](helpers.py#L1200) |
| 15 | Conditional mean | $\mu_{j\cdot PA(j)} = \sum_k \beta_{jk} Z_{pa_k}$ | $Z_{pa_k}$: $k$-th parent | Eq. 10 | [helpers.py:1227](helpers.py#L1227) |
| 16 | Conditional variance | $\sigma^2_{j\cdot PA(j)} = 1 - \sum_k \beta_{jk}\,\rho(Z_j, Z_{pa_k})$ | $\rho$: entry of $R$ | Eqs. 5, 11 | [helpers.py:1205](helpers.py#L1205) |
| 17 | Causal term | $T_{j\cdot PA(j)} = (Z_j - \mu_{j\cdot PA(j)})/\sigma_{j\cdot PA(j)}$; root PC: $T_j = Z_j$ | | Eq. 9 | [helpers.py:1227](helpers.py#L1227) |
| 18 | Overall T² (monitored) | $T^2 = Z^\top \operatorname{pinv}(R)\, Z$ on the PCs only (no flare data) | $Z$: one row of standardized PCs | Eq. 7; TA Main.py lines 243, 315 | [helpers.py:1171](helpers.py#L1171) |
| 19 | Sum of terms (check) | $\sum_j T^2_{j\cdot PA(j)} = Z^\top \Sigma_G^{-1} Z$, $\Sigma_G = (I - B^\top)^{-1} D (I - B^\top)^{-\top}$; its gap to row 18 is logged | $B_{kj} = \beta_{jk}$; $D = \operatorname{diag}(\sigma^2_{j\cdot PA(j)})$ | Eq. 13, Proposition 1; Eqs. 1-4 | [helpers.py:1262](helpers.py#L1262), [:1240](helpers.py#L1240) |
| 20 | Threshold | $T^2_{lim} = \dfrac{p\,(m+1)(m-1)}{m\,(m-k-p)}\,F_{1-\alpha;\,p,\,m-k-p}$ | $m$: NOC rows; $p$: variables; $k$: conditioning variables | Report Eq. 13 (corrected; see TA_clarifications Q2) | [helpers.py:1140](helpers.py#L1140) |
| 21 | Overall limit / per-PC limit | overall: row 20 with $p$ = PCs, $k = 0$, $\alpha$ = `ALPHA0`; per PC: $p = 1$, $k = \lvert PA(j)\rvert$, $\alpha$ = `ALPHA_TERM` (both 0.05) | | Report Eq. 13 | [main.py:71](main.py#L71), [:72](main.py#L72) |
| 22 | NOC reference | $T_{j\cdot PA(j)}$ has mean 0, std 1 on NOC (checked) | | p. 52 | [helpers.py:1282](helpers.py#L1282) |
| 23 | Flare found | event $e$ found $\iff \exists\, t \in W_e: T^2(t) > T^2_{lim}$ | | Project (user rule) | [helpers.py:1327](helpers.py#L1327) |
| 24 | Diagnosis | decompose the first $T^2(t) > T^2_{lim}$ in $W_e$; terms above their own limit = root-cause PCs | | Fig. 5 | [helpers.py:1330](helpers.py#L1330) |
| 25 | Share of the terms | $s_j(t) = T^2_{j\cdot PA(j)}(t) / \sum_k T^2_{k\cdot PA(k)}(t)$ | | Project | [helpers.py:1761](helpers.py#L1761) |
| 26 | Cause ranking per event | PCs above their own limit in $W_e$, ranked by minutes above, then earliest first crossing | | Project | [helpers.py:1983](helpers.py#L1983) |
| 27 | Ratio to own limit (deck) | $r_j(t) = T^2_{j\cdot PA(j)}(t) / T^2_{j,lim}$; weekly charts show $\max r_j$ per 15-min bin | $T^2_{j,lim}$: row 21 | Project | [helpers.py:2122](helpers.py#L2122), [:2136](helpers.py#L2136) |

- Row 18 follows the TA (sample $R$). The terms add up to it exactly only if the learned DAG is the true network (Proposition 1, p. 50); the gap is logged and in `limits.csv`.
- Report Eq. 13 is printed as $\frac{(m+p)(m-p)}{m(m-k-p)} F_{1-\alpha,p,m-p}$, with no factor $p$. Row 20 is the Phase II form (Tracy et al. 1992; Mason et al. 1995). For $p = 1$ the two differ only slightly. Open question to the TA: `TA_clarifications.pdf` Q2.
