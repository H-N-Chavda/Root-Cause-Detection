# MATHS: unit-wise PCA → PC → causation-based T²

Paper: Li, Jin & Shi (2008), J. Quality Technology 40(1) 46-58 (`references/papers/T^2 Decomposition.pdf`). Report: `references/papers/Work so far (report).pdf`. "Project" = in neither.

| # | Quantity | Equation | Symbols | Source | Code |
|---|---|---|---|---|---|
| 1 | flare_net | $F_t = f_{1,t} + f_{2,t} - r_t$, $r_t := 0$ if missing | $f_1, f_2$: flare mass flows; $r$: recovered gas (t/h), tags in `.env` | Project | [helpers.py:286](helpers.py#L286) |
| 2 | Flare threshold | `FLARE_USE_MAD_THRESHOLD` = False: $\tau$ = `FLARE_THRESHOLD` (1.5 t/h); True: $\tau = \operatorname{med}(F) + K \cdot 1.4826\,\operatorname{med}\lvert F - \operatorname{med}(F)\rvert$ | $K$ = `FLARE_MAD_K` = 3 | Project | [helpers.py:304](helpers.py#L304) |
| 3 | Flare events | flags $F_t > \tau$ with gap $\le g\Delta$ form one event; onset $t_e$ = first flag | $g$ = `EVENT_GAP` = 5; $\Delta$ = measured sampling interval | Project | [helpers.py:309](helpers.py#L309) |
| 4 | Pre-flare window | $W_e = \{t : t_e - n\Delta \le t < t_e\} \setminus \text{flare rows}$, $n = \operatorname{round}(30\,\text{min}/\Delta)$ | $t_e$: onset of event $e$ | Project | [helpers.py:361](helpers.py#L361), [:372](helpers.py#L372) |
| 5 | NOC rows | $t < t_0 + 0.6\,(t_N - t_0)$ and $t \notin [t_e - 2\text{h},\, t^{end}_e + 2\text{h}]\ \forall e$ | $t_0, t_N$: first/last row | Project | [helpers.py:345](helpers.py#L345) |
| 6 | Tag standardization | $z_{t,i} = (x_{t,i} - \bar{x}_i)/s_i$ | $\bar{x}_i, s_i$: NOC mean, std (ddof 1) of tag $i$ | p. 50 ($\mu = 0$, unit variance) | [helpers.py:463](helpers.py#L463) |
| 7 | Unit-wise PCA | $Z_u^{NOC} - \mathbf{1}\bar{z}_u^\top = U \Lambda^{1/2} V^\top$; $y_{t} = V_{:,1:x}^\top (z_{t,u} - \bar{z}_u)$ | $u$: unit; $x$ = `N_PCS_PER_UNIT` or `VAR_EXPLAINED_THRESHOLD`, capped at rank $\lambda_k > 10^{-6}\lambda_1$ | Project | [helpers.py:513](helpers.py#L513), [:490](helpers.py#L490) |
| 8 | PC-score standardization | $\tilde{y}_{t,k} = (y_{t,k} - \bar{y}_k)/s_k$ | NOC mean/std of score $k$; flare_net uses row 6, same rule | p. 50 | [helpers.py:518](helpers.py#L518), [:539](helpers.py#L539) |
| 9 | PC algorithm | Fisher-Z CI tests, stable PC; flare_net in tier 1, so $F \to Y$ is forbidden | $\alpha$ = `PC_ALPHA` = 0.01 | p. 49 (PC) | [helpers.py:659](helpers.py#L659), [:678](helpers.py#L678) |
| 10 | Flare is a sink | every flare edge oriented $Y \to F$ | | Project | [helpers.py:635](helpers.py#L635) |
| 11 | CPDAG → DAG | Dor-Tarsi: remove a sink $x$ whose undirected neighbours are adjacent to all of $\operatorname{adj}(x)$; orient into $x$ | | Dor & Tarsi (1992); paper needs a DAG (p. 48) | [helpers.py:690](helpers.py#L690), [:732](helpers.py#L732) |
| 12 | NOC correlation | $\hat{\Sigma} = \frac{1}{m - 1}\tilde{Y}_0^\top \tilde{Y}_0$ | $\tilde{Y}_0$: NOC rows of the $p$ PCs in the DAG; $m$: NOC rows | p. 49 (sample correlation) | [helpers.py:770](helpers.py#L770) |
| 13 | Path coefficients | $\beta_j = \hat{\Sigma}_{PP}^{-1}\hat{\Sigma}_{Pj}$ | $P = \mathbf{PA}(Z_j)$: parents of PC $j$ in the DAG | p. 49 | [helpers.py:806](helpers.py#L806) |
| 14 | Conditional mean | $\mu_{j\cdot PA(j)} = \sum_k \beta_{jk} Z_{pa_k}$ | $Z_{pa_k}$: $k$-th parent | Eq. 10 | [helpers.py:823](helpers.py#L823) |
| 15 | Conditional variance | $\sigma^2_{j\cdot PA(j)} = 1 - \sum_k \beta_{jk}\,\rho(Z_j, Z_{pa_k})$ | $\rho$: entry of $\hat{\Sigma}$ | Eqs. 5, 11 | [helpers.py:811](helpers.py#L811) |
| 16 | Causal term | $T_{j\cdot PA(j)} = (Z_j - \mu_{j\cdot PA(j)})/\sigma_{j\cdot PA(j)}$; root node: $T_j = Z_j$ | | Eq. 9 | [helpers.py:823](helpers.py#L823) |
| 17 | BN covariance | $Z = B^\top Z + E$, $E \sim N(0, D)$ ⇒ $\Sigma_G = (I - B^\top)^{-1} D (I - B^\top)^{-\top}$ | $B_{kj} = \beta_{jk}$; $D = \operatorname{diag}(\sigma^2_{j\cdot PA(j)})$ | Eqs. 1-4 | [helpers.py:828](helpers.py#L828) |
| 18 | Overall T² | $T^2 = Z^\top \Sigma^{-1} Z$, monitored with $\Sigma = \Sigma_G$; $T^2_S$ with $\Sigma = \hat{\Sigma}$ reported | $Z$: one row of standardized PCs | Eq. 7 | [helpers.py:782](helpers.py#L782) |
| 19 | Decomposition / sum identity | $T^2 = Z^\top \Sigma_G^{-1} Z = \sum_{j=1}^{p} T^2_{j\cdot PA(j)}$, since $\Sigma_G^{-1} = (I - B) D^{-1} (I - B^\top)$ | | Definition 2, Eq. 13; Proposition 1 | [helpers.py:837](helpers.py#L837) |
| 20 | Threshold (both limits) | $T^2_{lim} = \dfrac{p\,(m+1)(m-1)}{m\,(m-k-p)}\,F_{1-\alpha;\,p,\,m-k-p}$ | $m$: NOC rows; $p$: variables; $k$: conditioning variables | Report Eq. 13 (corrected, see below) | [helpers.py:776](helpers.py#L776) |
| 21 | Overall T² limit | row 20 with $p$ = PCs in the T², $k = 0$, $\alpha$ = `ALPHA0` = 0.05 | | Report Eq. 13 | [main.py:60](main.py#L60) |
| 22 | Per-PC limit | $T^2_{j\cdot PA(j)} >$ row 20 with $p = 1$, $k = \lvert PA(j)\rvert$, $\alpha$ = `ALPHA_TERM` = 0.05 | | Report Eq. 13 | [main.py:61](main.py#L61) |
| 23 | NOC reference | $T_{j\cdot PA(j)}$ has mean 0, std 1 on NOC (checked) | | p. 52 | [helpers.py:857](helpers.py#L857) |
| 24 | Diagnosis | decompose the first out-of-control $T^2$ in $W_e$; terms above their limit = root-cause PCs | | Fig. 5 | [helpers.py:875](helpers.py#L875) |
| 25 | Share of T² | $s_j(t) = T^2_{j\cdot PA(j)}(t) / T^2(t)$, $\sum_j s_j(t) = 1$ by row 19 | | Project (uses Eq. 13 of the paper) | `plot_event_zooms` in helpers.py |
| 26 | Cause ranking per event | PCs above their limit in $W_e$, ranked by: ancestor of flare_net, minutes above, earliest first, shortest DAG path $d(j \to F)$ | $d$: directed path length | Project (extends Fig. 5 with the DAG) | [helpers.py:1358](helpers.py#L1358), [:1385](helpers.py#L1385) |

- Paper vs request: the paper's Eq. 13 sum identity is exact for the sample $\hat{\Sigma}$ only if the learned DAG is the true BN (Proposition 1, p. 50). The code monitors $T^2$ with $\Sigma_G$, where the identity is exact, and logs the gap to $T^2_S$.
- Report Eq. 13 is printed as $\frac{(m+p)(m-p)}{m(m-k-p)} F_{1-\alpha,p,m-p}$. That has no factor $p$, so the overall limit would be about 1.2 against a normal T² of about $p$. Row 20 is the Phase II form (Tracy et al. 1992; Mason et al. 1995) with $p$, $(m+1)(m-1)$ and $m-k-p$ degrees of freedom. For $p = 1$ the two differ only slightly.
- The paper itself uses $\chi^2_p(1-\alpha)$ and a Bonferroni $z_{\alpha/(2p)}$; both are replaced by row 20 at the user's request (2026-11-03).
