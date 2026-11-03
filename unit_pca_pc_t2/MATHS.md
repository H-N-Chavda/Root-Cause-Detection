# MATHS: unit-wise PCA → PC → causation-based T²

Paper: Li, Jin & Shi (2008), J. Quality Technology 40(1) 46-58 (`references/papers/T^2 Decomposition.pdf`). Report: `references/papers/Work so far (report).pdf`. "Project" = in neither. Window: Jul–Oct 2025 (`config.py`).

| # | Quantity | Equation | Symbols | Source | Code |
|---|---|---|---|---|---|
| 1 | flare_net | $F_t = f_{1,t} + f_{2,t} - r_t$, $r_t := 0$ if missing | $f_1, f_2$: flare mass flows; $r$: recovered gas (t/h), tags in `.env` | Project | [helpers.py:464](helpers.py#L464) |
| 2 | Flare threshold | `FLARE_USE_MAD_THRESHOLD` = False: $\tau$ = `FLARE_THRESHOLD` (1.5 t/h); True: $\tau = \operatorname{med}(F) + K \cdot 1.4826\,\operatorname{med}\lvert F - \operatorname{med}(F)\rvert$ | $K$ = `FLARE_MAD_K` = 3 | Project | [helpers.py:496](helpers.py#L496) |
| 3 | Runs above $\tau$ | minutes with $F_t > \tau$ at most $g\Delta$ apart form one run | $g$ = `EVENT_GAP` = 5; $\Delta$: sampling interval | Project | [helpers.py:501](helpers.py#L501) |
| 4 | Flare event | a run with $t^{end} - t^{start} + \Delta \ge 30$ min; onset $t_e$ = first minute; shorter runs are kept out of NOC only | `FLARE_MIN_DURATION_MIN` = 30 | Project | [helpers.py:506](helpers.py#L506) |
| 5 | Pre-flare window | $W_e = \{t : t_e - n\Delta \le t < t_e\} \setminus \text{flare rows}$, $n = \operatorname{round}(30\,\text{min}/\Delta)$ | $t_e$: onset of event $e$ | Project | [helpers.py:599](helpers.py#L599), [:623](helpers.py#L623) |
| 6 | NOC rows | $t < t_0 + 0.6\,(t_N - t_0)$, $t \notin [t_e - 2\text{h},\, t^{end}_e + 2\text{h}]$, $t \notin$ short runs | $t_0, t_N$: first/last row | Project | [helpers.py:572](helpers.py#L572) |
| 7 | Tag standardization | $z_{t,i} = (x_{t,i} - \bar{x}_i)/s_i$ | $\bar{x}_i, s_i$: NOC mean, std (ddof 1) of tag $i$ | p. 50 ($\mu = 0$, unit variance) | [helpers.py:752](helpers.py#L752) |
| 8 | Unit-wise PCA | $Z_u^{NOC} - \mathbf{1}\bar{z}_u^\top = U \Lambda^{1/2} V^\top$; $y_{t} = V_{:,1:x}^\top (z_{t,u} - \bar{z}_u)$ | $u$: unit; $x$ = `N_PCS_PER_UNIT` or `VAR_EXPLAINED_THRESHOLD`, capped at rank $\lambda_k > 10^{-6}\lambda_1$ | Project | [helpers.py:849](helpers.py#L849), [:797](helpers.py#L797) |
| 9 | PC-score standardization | $\tilde{y}_{t,k} = (y_{t,k} - \bar{y}_k)/s_k$ | NOC mean/std of score $k$ | p. 50 | [helpers.py:854](helpers.py#L854) |
| 10 | PC algorithm | Fisher-Z CI tests, stable PC, on the PCs only (flare_net not included) | $\alpha$ = `PC_ALPHA` = 0.01 | p. 49 (PC) | [helpers.py:979](helpers.py#L979), [:1003](helpers.py#L1003) |
| 11 | CPDAG → DAG | Dor-Tarsi: remove a sink $x$ whose undirected neighbours are adjacent to all of $\operatorname{adj}(x)$; orient into $x$ | | Dor & Tarsi (1992); paper needs a DAG (p. 48) | [helpers.py:1012](helpers.py#L1012), [:1069](helpers.py#L1069) |
| 12 | Final graph | DAG among PCs $\cup \{Y_j \to F\ \forall j\}$ (added, not learned) | $F$: flare_net | Project | [helpers.py:1091](helpers.py#L1091) |
| 13 | NOC correlation | $\hat{\Sigma} = \frac{1}{m - 1}\tilde{Y}_0^\top \tilde{Y}_0$ | $\tilde{Y}_0$: NOC rows of the $p$ PCs; $m$: NOC rows | p. 49 (sample correlation) | [helpers.py:1125](helpers.py#L1125) |
| 14 | Path coefficients | $\beta_j = \hat{\Sigma}_{PP}^{-1}\hat{\Sigma}_{Pj}$ | $P = \mathbf{PA}(Z_j)$: parents of PC $j$ among PCs | p. 49 | [helpers.py:1196](helpers.py#L1196) |
| 15 | Conditional mean | $\mu_{j\cdot PA(j)} = \sum_k \beta_{jk} Z_{pa_k}$ | $Z_{pa_k}$: $k$-th parent | Eq. 10 | [helpers.py:1223](helpers.py#L1223) |
| 16 | Conditional variance | $\sigma^2_{j\cdot PA(j)} = 1 - \sum_k \beta_{jk}\,\rho(Z_j, Z_{pa_k})$ | $\rho$: entry of $\hat{\Sigma}$ | Eqs. 5, 11 | [helpers.py:1201](helpers.py#L1201) |
| 17 | Causal term | $T_{j\cdot PA(j)} = (Z_j - \mu_{j\cdot PA(j)})/\sigma_{j\cdot PA(j)}$; root PC: $T_j = Z_j$ | | Eq. 9 | [helpers.py:1223](helpers.py#L1223) |
| 18 | BN covariance | $Z = B^\top Z + E$, $E \sim N(0, D)$ ⇒ $\Sigma_G = (I - B^\top)^{-1} D (I - B^\top)^{-\top}$ | $B_{kj} = \beta_{jk}$; $D = \operatorname{diag}(\sigma^2_{j\cdot PA(j)})$ | Eqs. 1-4 | [helpers.py:1236](helpers.py#L1236) |
| 19 | Overall T² | $T^2 = Z^\top \Sigma_G^{-1} Z$ on the PCs only (no flare data); $T^2_S$ with $\hat{\Sigma}$ reported | $Z$: one row of standardized PCs | Eq. 7 | [helpers.py:1150](helpers.py#L1150) |
| 20 | Sum identity | $T^2 = \sum_{j=1}^{p} T^2_{j\cdot PA(j)}$, since $\Sigma_G^{-1} = (I - B) D^{-1} (I - B^\top)$ | | Definition 2, Eq. 13; Proposition 1 | [helpers.py:1258](helpers.py#L1258) |
| 21 | Threshold | $T^2_{lim} = \dfrac{p\,(m+1)(m-1)}{m\,(m-k-p)}\,F_{1-\alpha;\,p,\,m-k-p}$ | $m$: NOC rows; $p$: variables; $k$: conditioning variables | Report Eq. 13 (corrected) | [helpers.py:1131](helpers.py#L1131) |
| 22 | Overall limit / per-PC limit | overall: row 21 with $p$ = PCs, $k = 0$, $\alpha$ = `ALPHA0`; per PC: $p = 1$, $k = \lvert PA(j)\rvert$, $\alpha$ = `ALPHA_TERM` (both 0.05) | | Report Eq. 13 | [main.py:71](main.py#L71), [:72](main.py#L72) |
| 23 | NOC reference | $T_{j\cdot PA(j)}$ has mean 0, std 1 on NOC (checked) | | p. 52 | [helpers.py:1278](helpers.py#L1278) |
| 24 | Flare found | event $e$ found $\iff \exists\, t \in W_e: T^2(t) > T^2_{lim}$ | | Project (user rule) | [helpers.py:1323](helpers.py#L1323) |
| 25 | Diagnosis | decompose the first $T^2(t) > T^2_{lim}$ in $W_e$; terms above their own limit = root-cause PCs | | Fig. 5 | [helpers.py:1326](helpers.py#L1326) |
| 26 | Share of T² | $s_j(t) = T^2_{j\cdot PA(j)}(t) / T^2(t)$, $\sum_j s_j(t) = 1$ by row 20 | | Project | [helpers.py:1757](helpers.py#L1757) |
| 27 | Cause ranking per event | PCs above their own limit in $W_e$, ranked by minutes above, then earliest first crossing | | Project | [helpers.py:1979](helpers.py#L1979) |

- Paper vs request: Eq. 13 of the paper (the sum identity) holds exactly for the sample $\hat{\Sigma}$ only if the learned DAG is the true BN (Proposition 1, p. 50). The code monitors $T^2$ with $\Sigma_G$, where it is exact, and logs the gap to $T^2_S$.
- Report Eq. 13 is printed as $\frac{(m+p)(m-p)}{m(m-k-p)} F_{1-\alpha,p,m-p}$. It has no factor $p$, so the overall limit would be about 1.2 against a normal T² of about $p$. Row 21 is the Phase II form (Tracy et al. 1992; Mason et al. 1995). For $p = 1$ the two differ only slightly.
- Every function's inputs, outputs and data sources are in its docstring in `helpers.py`.
