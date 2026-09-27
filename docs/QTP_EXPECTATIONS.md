# Quadruple-Tank: expected behaviour

Johansson, IEEE TCST 8(3):456-465, 2000 + datasets/*.m. Predictions written 2026-09-07 before any
run, unchanged since; facts corrected same day after reading the .m files.

## Ground truth (QTank_Dynamics.m, verified against eq. 1)
| Target | Parents |
|---|---|
| h1 | h1 (self), h3, v1 |
| h2 | h2 (self), h4, v2 |
| h3 | h3 (self), v2, d |
| h4 | h4 (self), v1, d |

8 cross-edges: v1->h1, v1->h4, v2->h2, v2->h3, d->h3, d->h4, h3->h1, h4->h2. Acyclic.
d is the .m files' disturbance (gam3=0.4 split across tanks 3,4), NOT in the paper. v1, v2, d are
all recorded -> causally sufficient; omitting any leaves a confounder.

## Data properties (measured on the generated data)
| Property | Value |
|---|---|
| Inputs v1,v2,d | PRBS, kurtosis -1.97/-1.99/-2.00, JB p ~1e-105 -> strongly non-Gaussian |
| Levels h1..h4 | near-Gaussian (linear filtering of PRBS), JB p 0.008-0.35; noise Gaussian white sigma=0.05 measurement + 0.05 input |
| Stationarity | ADF p < 0.0001 all 7 columns, no unit root |
| Autocorrelation | lag-1: levels 0.93-0.99, inputs 0.83-0.90 |
| P+ (from .m) | T = 63.2, 91.4, 39.0, 56.1 s (paper 63,91,39,56); gam sum 0.77, nonmin phase, RHP zero +0.013, RGA -0.64 |
| P- (paper, NOT .m) | T = 62.7, 90.3, 23.9, 30.0 s (paper 62,90,23,30); gam sum 1.30, min phase, RGA 1.40 |

## Run config
Ts = 5 s, 4 h -> 2880 rows, 600 s burn-in discarded. Dominant constant ~90 s = ~18 samples.
tau_max 20 (pcmci_plus, var_lingam), 8 (lste, cost); pc is contemporaneous-only. standardize=True.
PRBS: pumps +/-0.30 V hold 30 s, disturbance +/-0.50 hold 50 s.

## Expectations (unchanged since first written)
| Algorithm | Expected | Reason |
|---|---|---|
| pcmci_plus | best; most or all 8 edges | built for autocorrelated series |
| var_lingam | good lagged, weak contemporaneous | v/d residuals non-Gaussian -> identifiable; h residuals near-Gaussian -> not |
| lste | strong edges only, noisy | sample-hungry, TE saturates under autocorrelation; tau_max 8 < 18 may miss slow edges |
| pc | poor, dense, spurious | contemporaneous-only vs a purely dynamic system |

Ranking: pcmci_plus > var_lingam > lste >> pc. A bad `pc` result is expected, not a bug. At P+ the
RHP zero inverts the short-lag sign, so orientations may flip vs P-.
