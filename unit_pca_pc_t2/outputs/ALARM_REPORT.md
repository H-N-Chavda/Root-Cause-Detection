# Alarm report: overall T² detection, per-PC root cause (2026-11-04)

Source: commit `0ae2182` (lab run, pulled 2026-11-04), `summary.py` output, counts from `t2/` and `tables/`.
Pulled in: new `config.py`; figs 1–5 + 24 per-PC plots; `graph/dag_edges.csv`, `pc_edges.csv`; tables `root_cause_by_event`, `t2_peaks`, `short_flares`, `pcs_per_unit`. Old `outputs/data/*` removed.

## Setup
| item | value |
|---|---|
| window | 2025-07-01 to 2025-11-01, 1-min rows (177,120) |
| flare event | flare_net > 1.5 t/h for >= 30 min: 620 events; 1,723 shorter runs |
| overall T² limit | 36.51 (Eq. 13, p = 24 PCs, m = 12,157 NOC rows, α = 0.05) |
| per-PC T²_j limit | 3.84 (Eq. 13, p = 1, k = parents) |
| "detected" | T² above limit at least once in the 30 min before onset |

## Detection (overall T²)
| metric | value |
|---|---|
| evaluable events (pre-window > 0 rows) | 619 (E1 starts at window start) |
| detected | **532 (85.9%)** |
| missed alarms | **87 (14.1%)** |
| pre-flare rows above limit | 78.1% (NOC rows: 11.7%) |
| chance level: 30-min NOC window with >= 1 alarm | 16.5% |
| detected with full 30-min window / short window (<30 min, previous flare close) | 322/375 (85.9%) / 210/244 (86.1%) |

## Root cause (per-PC T²_j)
- Detected events with a top root-cause PC: 532/532; top PC above its limit at the first T² alarm: 522.
- Top PC by unit: APS_NVPS_LEU 178, DAK 144, DIU 77, SEU3 21, PDA 20, LOUP 18, others <= 17.
- Top PCs: DAK_PC1 144, APS_NVPS_LEU_PC1 103, APS_NVPS_LEU_PC2 75, DIU_PC1 71 (= 393 of 532).
- PCs above their limit at the first alarm: median 5, max 17.
- 66 of the 87 missed events still have a PC above its own limit (per-PC terms catch what overall T² misses).

## False alarms (overall T², outside flare events)
| alarm episode length | true (in pre-flare window) | near a short flare | false |
|---|---|---|---|
| >= 1 min | 586 | 129 | **147** (1,091 min) |
| >= 5 min | 523 | 95 | **54** (916 min) |
| >= 10 min | 409 | 73 | **29** (754 min) |
- Row level: 1,422 of 12,157 NOC rows above limit = 11.7% (design target 5%).

## Findings
- Detection rate (85.9%) is far above chance (16.5%); the limit separates pre-flare from NOC.
- 478 of 532 detections: alarm was already on the minute before the 30-min window. T² stays high across short gaps, so "minutes before onset" is not a true lead time.
- Root cause is concentrated: DAK_PC1 and APS_NVPS_LEU PCs lead 322 of 532 events. DAK has 3 tags, rank 1 — check it is not tracking flare-linked flow. UNVERIFIED.
- NOC false-alarm rate 11.7% is 2.3x the 5% design; the NOC set likely holds non-normal periods. UNVERIFIED.
- Missed events look like detected ones (median 69 vs 65 min, peak 2.55 vs 3.01 t/h); no simple size cut-off explains them.
