"""Stage 2 - unit-wise PCA (scikit-learn): one PCA per unit on that unit's process tags, top PCS_PER_UNIT components
kept per unit; flare tags do not enter PCA.

Each process tag is standardized over the PCA fit rows (zero mean, unit variance); the fit rows are the full cleaned
matrix. Scores are named <unit>_PC1, <unit>_PC2, ... and standardized to unit variance for Stage 3. PCS_PER_UNIT is set in
.env (default 2: 13 units x 2 = 26 PCs); a unit with fewer tags than PCS_PER_UNIT keeps as many PCs as it has tags.

Statistical checks on the scores (reported only, never used to tune the graph):
- PCs of one unit are uncorrelated by construction, so only PC-PC edges between units can carry information;
  the cross-unit score correlations bound how many PC-PC edges are possible at all.
- Fisher-Z assumes linear-Gaussian data: skewness and excess kurtosis of every score.
- Fisher-Z assumes independent rows: lag autocorrelation at the Stage 3 row step and the AR(1) effective sample size.

Writes outputs/data/{pc_scores.parquet, pca_loadings.csv, pca_variance.csv, s2_summary.json} and
outputs/tables/{pca_top_tags.csv, pca_unit_share.csv, pc_score_correlation.csv, pc_assumption_checks.csv};
process tags get status 'used_in_pca' in tag_registry.csv.
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from sklearn.decomposition import PCA

from common import (DATA, PC_ALPHA, PC_ROW_STEP, PCS_PER_UNIT, SEED, TAB, TOP_LOAD_N, UNIT_FOLDERS, assumption,
                    clear_notes, done, gate, log, read_registry, save_json, stage, write_registry)

S = "s2_pca"
t0 = time.time()
stage("Stage 2 - unit-wise PCA on the process tags")
clear_notes(S)
X = pd.read_parquet(DATA / "X_process.parquet")
reg = read_registry()
unit_of = dict(zip(reg["tag"], reg["unit"]))
tag_unit = pd.Series([unit_of[t] for t in X.columns], index=X.columns)
units = [u for u in UNIT_FOLDERS if u in set(tag_unit)] + sorted(set(tag_unit) - set(UNIT_FOLDERS))
missing_units = [u for u in UNIT_FOLDERS if u not in units]
log(f"units with process tags: {len(units)} ({len(missing_units)} of the {len(UNIT_FOLDERS)} in .env have none); "
    f"tags per unit: {tag_unit.value_counts()[units].to_dict()}")
if missing_units:
    assumption(S, f"units named in .env with no process tag in the cleaned matrix get no PCs: {missing_units}; "
                  f"the planned {PCS_PER_UNIT * len(UNIT_FOLDERS)} PCs need all {len(UNIT_FOLDERS)} units")
gate(len(units) >= 2, f"{len(units)} units with process tags (a cross-unit graph needs at least 2)")

A = X.to_numpy(dtype=np.float64)
mu, sd = A.mean(axis=0), A.std(axis=0)
gate(bool((sd > 0).all()), "every process tag varies over the fit rows")
A = (A - mu) / sd
assumption(S, f"PCA fit rows = the full cleaned matrix ({len(A)} rows); each process tag standardized over these rows "
              f"(zero mean, unit variance); one PCA per unit on its own tags, top {PCS_PER_UNIT} components kept")

# ---- one PCA per unit; loadings are stored over all process tags (zero outside the unit)
pcs, pc_unit, cols_scores, load_rows, var_rows = [], [], [], [], []
for u in units:
    cols = np.flatnonzero((tag_unit == u).to_numpy())
    k = min(PCS_PER_UNIT, len(cols))
    if k < PCS_PER_UNIT:
        assumption(S, f"{u} has {len(cols)} process tag(s): {k} PC(s) kept instead of {PCS_PER_UNIT}")
    pca = PCA(n_components=k, svd_solver="full", random_state=SEED)
    sc = pca.fit_transform(A[:, cols])
    ssd = sc.std(axis=0)
    gate(pca.n_components_ == k and bool(np.isfinite(sc).all()) and bool((ssd > 0).all()),
         f"{u}: {k} PCs from {len(cols)} tags, scores finite and not constant")
    cum = np.cumsum(pca.explained_variance_ratio_)
    for i in range(k):
        name = f"{u}_PC{i + 1}"
        pcs.append(name)
        pc_unit.append(u)
        cols_scores.append(sc[:, i] / ssd[i])
        row = np.zeros(X.shape[1])
        row[cols] = pca.components_[i]
        load_rows.append(row)
        var_rows.append({"pc": name, "unit": u, "n_unit_tags": len(cols), "explained_variance": pca.explained_variance_[i],
                         "explained_variance_ratio": pca.explained_variance_ratio_[i], "cumulative": cum[i],
                         "share_of_all_tags": pca.explained_variance_ratio_[i] * len(cols) / X.shape[1]})
    log(f"  {u}: {len(cols)} tags, {k} PCs keep {100 * cum[-1]:.1f}% of the unit's variance "
        f"({', '.join(f'{100 * v:.1f}%' for v in pca.explained_variance_ratio_)})")
del A
scores = np.column_stack(cols_scores)
var = pd.DataFrame(var_rows)
n_pcs = len(pcs)
kept_all = float(var["share_of_all_tags"].sum())
log(f"{n_pcs} PCs from {len(units)} units keep {100 * kept_all:.2f}% of the total process-tag variance")
assumption(S, "PC scores standardized to unit variance (each divided by its std over the fit rows) before Stage 3")
pd.DataFrame(scores.astype("float32"), index=X.index, columns=pcs).to_parquet(DATA / "pc_scores.parquet")

load = pd.DataFrame(np.vstack(load_rows), index=pd.Index(pcs, name="pc"), columns=X.columns)
load.to_csv(DATA / "pca_loadings.csv")
var.to_csv(DATA / "pca_variance.csv", index=False)
top_rows = []
for pc, u in zip(pcs, pc_unit):
    s = load.loc[pc, tag_unit.index[tag_unit == u]]
    for rank, t in enumerate(s.abs().sort_values(ascending=False).index[:TOP_LOAD_N], start=1):
        top_rows.append({"pc": pc, "rank": rank, "tag": t, "unit": u, "loading": float(s[t]),
                         "squared_loading": float(s[t] ** 2)})
pd.DataFrame(top_rows).to_csv(TAB / "pca_top_tags.csv", index=False)
share = (load ** 2).T.groupby(tag_unit).sum().T[units]          # each PC: 1.0 in its own unit, 0 elsewhere
share.to_csv(TAB / "pca_unit_share.csv")
gate(bool(np.allclose(share.sum(axis=1), 1.0)) and all(share.loc[p].idxmax() == u for p, u in zip(pcs, pc_unit)),
     "every PC loads only on its own unit (squared-loading share 1)")

# ---- assumption checks on the scores
C = np.corrcoef(scores, rowvar=False)
pu = np.array(pc_unit)
same = pu[:, None] == pu[None, :]
iu = np.triu_indices(n_pcs, 1)
within = np.abs(C[iu][same[iu]])
cross = np.abs(C[iu][~same[iu]])
gate(within.size == 0 or float(within.max()) < 1e-6, "PCs of the same unit are uncorrelated over the fit rows "
     f"(max |r| {float(within.max()) if within.size else 0:.1e})")
pd.DataFrame(C, index=pcs, columns=pcs).to_csv(TAB / "pc_score_correlation.csv")
n_sub = len(scores[::PC_ROW_STEP])
r_crit = float(np.tanh(norm.ppf(1 - PC_ALPHA / 2) / np.sqrt(n_sub - 3)))
n_dep = int((cross > r_crit).sum())
log(f"cross-unit |r| between PC scores: median {np.median(cross):.3f}, max {cross.max():.3f}; {n_dep} of {cross.size} "
    f"pairs above the depth-0 Fisher-Z limit |r| > {r_crit:.4f} (alpha {PC_ALPHA}, n = {n_sub})")
assumption(S, f"{n_dep} cross-unit PC pairs are marginally dependent at alpha {PC_ALPHA} (n = {n_sub}); this is the most "
              "PC-PC edges PC can keep, since within-unit PC pairs are uncorrelated by construction")

lag = np.array([np.corrcoef(scores[:-PC_ROW_STEP, j], scores[PC_ROW_STEP:, j])[0, 1] for j in range(n_pcs)])
ess = n_sub * (1 - lag) / (1 + lag)
Cx = C.copy()
Cx[same] = 0.0
partner = np.abs(Cx).argmax(axis=1)
chk = pd.DataFrame({"pc": pcs, "unit": pc_unit, "n_unit_tags": var["n_unit_tags"],
                    "pct_variance_of_unit": 100 * var["explained_variance_ratio"],
                    "pct_variance_of_all_tags": 100 * var["share_of_all_tags"],
                    "skewness": skew(scores, axis=0), "excess_kurtosis": kurtosis(scores, axis=0),
                    f"autocorr_lag{PC_ROW_STEP}": lag, "ess_ar1_of_pc_rows": ess,
                    "max_abs_r_other_unit": np.abs(Cx[np.arange(n_pcs), partner]),
                    "max_r_partner": [pcs[j] for j in partner]})
chk.to_csv(TAB / "pc_assumption_checks.csv", index=False)
log(f"skewness |median| {chk['skewness'].abs().median():.2f} / max {chk['skewness'].abs().max():.2f}; excess kurtosis "
    f"median {chk['excess_kurtosis'].median():.2f} / max {chk['excess_kurtosis'].max():.2f} (Gaussian: 0 / 0)")
log(f"lag-{PC_ROW_STEP} autocorrelation median {np.median(lag):.3f} / max {lag.max():.3f}; AR(1) effective sample size of "
    f"the {n_sub} PC rows: median {np.median(ess):.0f} / min {ess.min():.0f}")
assumption(S, f"Fisher-Z in Stage 3 treats the {n_sub} subsampled rows as independent; with lag-{PC_ROW_STEP} "
              f"autocorrelation median {np.median(lag):.2f} the effective sample size is about {np.median(ess):.0f}, "
              "so p-values are too small and the graph can keep spurious edges")
assumption(S, f"causal sufficiency: the {100 * (1 - kept_all):.1f}% of process-tag variance outside the kept PCs is "
              "unobserved; a dependence carried only by dropped components is missed or appears as a spurious edge")

reg["status"] = ["used_in_pca" if t in load.columns else st for t, st in zip(reg["tag"], reg["status"])]
write_registry(reg)
seconds = time.time() - t0
save_json({"n_process_tags": X.shape[1], "rows": len(X), "n_pcs": n_pcs, "pcs_per_unit": PCS_PER_UNIT,
           "units": units, "missing_units": missing_units, "pcs": pcs,
           "evr": var["share_of_all_tags"].tolist(), "evr_within_unit": var["explained_variance_ratio"].tolist(),
           "cum_variance": kept_all,
           "unit_variance_kept": var.groupby("unit")["explained_variance_ratio"].sum()[units].to_dict(),
           "dominant_unit": dict(zip(pcs, pc_unit)), "dominant_share": share.max(axis=1).to_dict(),
           "cross_unit_abs_r_median": float(np.median(cross)), "cross_unit_abs_r_max": float(cross.max()),
           "cross_unit_pairs": int(cross.size), "cross_unit_pairs_dependent": n_dep, "r_crit_depth0": r_crit,
           "lag_autocorr_median": float(np.median(lag)), "ess_median": float(np.median(ess)),
           "skew_abs_median": float(chk["skewness"].abs().median()),
           "excess_kurtosis_median": float(chk["excess_kurtosis"].median()), "seconds": seconds}, "s2_summary.json")
done(S, seconds)
