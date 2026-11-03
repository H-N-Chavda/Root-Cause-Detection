# MATHS: unit-wise PCA → PC → causation-based T²

Paper: Li, Jin & Shi (2008), J. Quality Technology 40(1) 46-58 (`references/papers/T^2 Decomposition.pdf`). "Project" = not in the paper.

| # | Quantity | Equation | Symbols | Source | Code |
|---|---|---|---|---|---|
| 1 | flare_net | $F_t = f_{1,t} + f_{2,t} - r_t$, $r_t := 0$ if missing | $f_1, f_2$: flare mass flows; $r$: recovered gas (t/h), tags in `.env` | Project | [helpers.py:285](helpers.py#L285) |
| 2 | Flare threshold | $\tau = \operatorname{med}(F) + K \cdot 1.4826\,\operatorname{med}\lvert F - \operatorname{med}(F)\rvert$ | $K$ = `FLARE_MAD_K` = 3 | Project | [helpers.py:303](helpers.py#L303) |
| 3 | Flare events | flags $F_t > \tau$ with gap $\le g\Delta$ form one event; onset $t_e$ = first flag | $g$ = `EVENT_GAP` = 5; $\Delta$ = measured sampling interval | Project | [helpers.py:308](helpers.py#L308) |
| 4 | Pre-flare window | $W_e = \{t : t_e - n\Delta \le t < t_e\} \setminus \text{flare rows}$, $n = \operatorname{round}(30\,\text{min}/\Delta)$ | $t_e$: onset of event $e$ | Project | [helpers.py:360](helpers.py#L360), [:371](helpers.py#L371) |
| 5 | NOC rows | $t < t_0 + 0.6\,(t_N - t_0)$ and $t \notin [t_e - 2\text{h},\, t^{end}_e + 2\text{h}]\ \forall e$ | $t_0, t_N$: first/last row | Project | [helpers.py:344](helpers.py#L344) |
| 6 | Tag standardization | $z_{t,i} = (x_{t,i} - \bar{x}_i)/s_i$ | $\bar{x}_i, s_i$: NOC mean, std (ddof 1) of tag $i$ | p. 50 ($\mu = 0$, unit variance) | [helpers.py:462](helpers.py#L462) |
| 7 | Unit-wise PCA | $Z_u^{NOC} - \mathbf{1}\bar{z}_u^\top = U \Lambda^{1/2} V^\top$; $y_{t} = V_{:,1:x}^\top (z_{t,u} - \bar{z}_u)$ | $u$: unit; $x$ = `N_PCS_PER_UNIT` or `VAR_EXPLAINED_THRESHOLD`, capped at rank $\lambda_k > 10^{-6}\lambda_1$ | Project | [helpers.py:512](helpers.py#L512), [:489](helpers.py#L489) |
| 8 | PC-score standardization | $\tilde{y}_{t,k} = (y_{t,k} - \bar{y}_k)/s_k$ | NOC mean/std of score $k$; flare_net uses row 6, same rule | p. 50 | [helpers.py:517](helpers.py#L517), [:538](helpers.py#L538) |
| 9 | PC algorithm | Fisher-Z CI tests, stable PC; flare_net in tier 1, so $F \to Y$ is forbidden | $\alpha$ = `PC_ALPHA` = 0.01 | p. 49 (PC) | [helpers.py:658](helpers.py#L658), [:677](helpers.py#L677) |
| 10 | Flare is a sink | every flare edge oriented $Y \to F$ | | Project | [helpers.py:634](helpers.py#L634) |
| 11 | CPDAG → DAG | Dor-Tarsi: remove a sink $x$ whose undirected neighbours are adjacent to all of $\operatorname{adj}(x)$; orient into $x$ | | Dor & Tarsi (1992); paper needs a DAG (p. 48) | [helpers.py:689](helpers.py#L689), [:731](helpers.py#L731) |
| 12 | NOC correlation | $\hat{\Sigma} = \frac{1}{n_0 - 1}\tilde{Y}_0^\top \tilde{Y}_0$ | $\tilde{Y}_0$: NOC rows of the $p$ PCs in the DAG; $n_0$: NOC rows | p. 49 (sample correlation) | [helpers.py:768](helpers.py#L768) |
| 13 | Path coefficients | $\beta_j = \hat{\Sigma}_{PP}^{-1}\hat{\Sigma}_{Pj}$ | $P = \mathbf{PA}(Z_j)$: parents of PC $j$ in the DAG | p. 49 | [helpers.py:803](helpers.py#L803) |
| 14 | Conditional mean | $\mu_{j\cdot PA(j)} = \sum_k \beta_{jk} Z_{pa_k}$ | $Z_{pa_k}$: $k$-th parent | Eq. 10 | [helpers.py:820](helpers.py#L820) |
| 15 | Conditional variance | $\sigma^2_{j\cdot PA(j)} = 1 - \sum_k \beta_{jk}\,\rho(Z_j, Z_{pa_k})$ | $\rho$: entry of $\hat{\Sigma}$ | Eqs. 5, 11 | [helpers.py:808](helpers.py#L808) |
| 16 | Causal term | $T_{j\cdot PA(j)} = (Z_j - \mu_{j\cdot PA(j)})/\sigma_{j\cdot PA(j)}$; root node: $T_j = Z_j$ | | Eq. 9 | [helpers.py:820](helpers.py#L820) |
| 17 | BN covariance | $Z = B^\top Z + E$, $E \sim N(0, D)$ ⇒ $\Sigma_G = (I - B^\top)^{-1} D (I - B^\top)^{-\top}$ | $B_{kj} = \beta_{jk}$; $D = \operatorname{diag}(\sigma^2_{j\cdot PA(j)})$ | Eqs. 1-4 | [helpers.py:825](helpers.py#L825) |
| 18 | Overall T² | $T^2 = Z^\top \Sigma^{-1} Z$, monitored with $\Sigma = \Sigma_G$; $T^2_S$ with $\Sigma = \hat{\Sigma}$ reported | $Z$: one row of standardized PCs | Eq. 7 | [helpers.py:779](helpers.py#L779) |
| 19 | Decomposition / sum identity | $T^2 = Z^\top \Sigma_G^{-1} Z = \sum_{j=1}^{p} T^2_{j\cdot PA(j)}$, since $\Sigma_G^{-1} = (I - B) D^{-1} (I - B^\top)$ | | Definition 2, Eq. 13; Proposition 1 | [helpers.py:834](helpers.py#L834) |
| 20 | T² limit | $T^2 > \chi^2_p(1 - \alpha)$ is out of control | $\alpha$ = `ALPHA0` = 0.05; $p$: PCs in the T² | Eq. 6; p. 52 step 1(b) | [helpers.py:776](helpers.py#L776) |
| 21 | Per-term limit | $\lvert T_{j\cdot PA(j)}\rvert > z_{\alpha'/2}$, $\alpha' = \alpha/p$ (Bonferroni) | $z_q$: upper $q$ point of $N(0,1)$ | p. 52 (Fig. 5, step 1(c)) | [main.py:63](main.py#L63) |
| 22 | NOC reference | $T_{j\cdot PA(j)} \sim N(0,1)$ on NOC; checked as mean 0, std 1 | | p. 52 | [helpers.py:854](helpers.py#L854) |
| 23 | Diagnosis | decompose the first out-of-control $T^2$ in $W_e$; significant terms = root-cause PCs | | Fig. 5 | [helpers.py:872](helpers.py#L872) |
| 24 | Share of T² | $s_j(t) = T^2_{j\cdot PA(j)}(t) / T^2(t)$, $\sum_j s_j(t) = 1$ by row 19 | | Project (uses Eq. 13) | `plot_event_zooms` in helpers.py |
| 25 | Cause ranking per event | PCs with $\lvert T_{j\cdot PA(j)}\rvert > z_{\alpha'/2}$ in $W_e$, ranked by: ancestor of flare_net, minutes significant, earliest first significance, shortest DAG path $d(j \to F)$ | $d$: directed path length in the DAG | Project (extends Fig. 5 with the DAG) | [helpers.py:1351](helpers.py#L1351), [:1378](helpers.py#L1378) |

- Paper vs request: Eq. 13 is exact for the sample $\hat{\Sigma}$ only if the learned DAG is the true BN (Proposition 1; p. 50 assumes no learning or sampling error). A DAG alone is not enough. The code monitors $T^2$ with $\Sigma_G$, where the identity is exact, and logs the gap to $T^2_S$.
- Limits are $\chi^2$ with known $\Sigma$, as in the paper. They are not the finite-sample Phase II $F$ limit.
