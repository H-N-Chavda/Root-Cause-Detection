# Copula-transformed runs vs raw (2026-10-02)

Transform: per-column Gaussian copula (`causal_bench.eda.copula`), then standardize. Same tau_max=20, pc_alpha=0.01, ParCorr, VAR-LiNGAM bootstrap 100 / thr 0.9 as the raw baselines.

| run | rows | ds | algo | F1 raw | F1 copula | SHD raw | SHD copula | FP raw→cop |
|---|---|---|---|---|---|---|---|---|
| tau20-s3 | 960 | P_minus | pc | 0.67 | 0.71 | 6 | 5 | 2→1 |
| tau20-s3 | 960 | P_minus | pcmci_plus | 0.88 | 0.86 | 2 | 2 | 1→0 |
| tau20-s3 | 960 | P_minus | var_lingam | 0.67 | 0.63 | 5 | 8 | 3→6 |
| tau20-s3 | 960 | P_plus | pc | 0.77 | 0.77 | 3 | 3 | 0→0 |
| tau20-s3 | 960 | P_plus | pcmci_plus | 0.89 | 0.82 | 2 | 2 | 2→1 |
| tau20-s3 | 960 | P_plus | var_lingam | 0.89 | 0.80 | 2 | 5 | 2→5 |
| tau20-full | 2880 | P_minus | pc | 0.57 | 0.67 | 6 | 4 | 2→0 |
| tau20-full | 2880 | P_minus | pcmci_plus | 0.82 | 0.78 | 3 | 4 | 2→3 |
| tau20-full | 2880 | P_minus | var_lingam | 0.63 | 0.71 | 8 | 5 | 6→3 |
| tau20-full | 2880 | P_plus | pc | 0.55 | 0.62 | 5 | 5 | 0→1 |
| tau20-full | 2880 | P_plus | pcmci_plus | 0.82 | 0.82 | 2 | 3 | 1→1 |
| tau20-full | 2880 | P_plus | var_lingam | 0.76 | 0.74 | 5 | 6 | 5→5 |
| lste-s3 | 960 | P_minus | lste (tau 8, 100 shuffles) | 0.00 (run-lste-quick) | 0.00 | 8 | 8 | 0→0 |

- LSTE copula: 0 edges, 1313 s, P_minus only (P_plus skipped on request).
- VAR-LiNGAM runtime roughly doubled under copula (135 s → 240 s stride-3; 300 s → 500 s full).
- Dirs: `run-copula-tau20-s3`, `run-copula-tau20-full`, `run-copula-lste-s3` (each has `scores.json`).
- Raw baselines: `run-parcorr-tau20-s3`, `run-parcorr-tau20-full`.
