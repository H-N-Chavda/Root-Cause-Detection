# Plan — CMIknn comparison (2026-09-27)

## Goal
Run PC, PCMCI+, VAR-LiNGAM on the existing square-wave qtank data with identical
settings and a nonlinear independence test. LSTE deferred.

## Shared config
| Setting | Value |
|---|---|
| Datasets | P_minus, P_plus (existing, unchanged) |
| Rows | every 3rd → 960 rows, Ts=15s, same 4h span |
| tau_max | 20 (= 300s window) |
| Independence test | CMIknn, 500 shuffles, block length from tigramite's own estimator (=12) |
| pc_alpha | 0.01 |
| Output | `results/qtank/run-cmiknn/` (run-fast untouched) |

VAR-LiNGAM has no CI test to swap; runs at lags=20, bootstrap 100, threshold 0.9,
as in run-fast.

## Code changes
| File | Change |
|---|---|
| `src/causal_bench/utils/compat.py` | NEW. Shim stripping dead `ddof`/`bias` from `np.corrcoef` (NumPy 2 removed them; they were already no-ops). Imported from `causal_bench/__init__.py` |
| `src/causal_bench/algorithms/pc.py` | Add `tau_max` (default 0) and `independence_test` (default "parcorr"); pass to existing `run_pcalg` call |
| `datasets/run_algorithms.py` | Add `--independence-test` and `--cmiknn-sig` flags; plumb to pc + pcmci_plus |

No change to `pcmci_plus.py` (already exposes `independence_test`) or `lste.py`.
Defaults preserve current behaviour.

## Runtime — UNVERIFIED, extrapolated from measured 49 tests/lag @ 0.4s/test
| Algorithm | per dataset |
|---|---|
| PCMCI+ | ~67 min |
| PC | ~10 min |
| VAR-LiNGAM | ~5 min |

Total both datasets ≈ 2.7h. One dataset per invocation.

## Risks
- Lagged PC is PC-stable on time series, not PCMCI+: no autocorrelation
  correction. Gap between them is informative, not a bug.
- LSTE deferred; `run-lste-quick` remains the only LSTE artifact (0 edges).
