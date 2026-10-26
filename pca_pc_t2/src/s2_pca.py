"""Stage 2 - PCA (scikit-learn) with exactly 26 components on the process tags; flare tags do not enter PCA.

X_process from Stage 1 is standardized over the PCA fit rows (zero mean, unit variance per tag); the fit rows are the
full cleaned matrix. Scores are renamed PC1..PC26 and standardized to unit variance for Stage 3.

Writes outputs/data/{pc_scores.parquet, pca_loadings.csv, pca_variance.csv, s2_summary.json} and
outputs/tables/{pca_top_tags.csv, pca_unit_share.csv}; process tags get status 'used_in_pca' in tag_registry.csv.
"""
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import (DATA, N_PCS, SEED, TAB, TOP_LOAD_N, UNIT_FOLDERS, assumption, clear_notes, done, gate, log,
                    read_registry, save_json, stage, write_registry)

S = "s2_pca"
t0 = time.time()
stage("Stage 2 - PCA on the process tags")
clear_notes(S)
X = pd.read_parquet(DATA / "X_process.parquet")
reg = read_registry()
unit_of = dict(zip(reg["tag"], reg["unit"]))
gate(X.shape[1] >= N_PCS, f"{X.shape[1]} process tags >= {N_PCS} components")

A = X.to_numpy(dtype=np.float64)
mu, sd = A.mean(axis=0), A.std(axis=0)
gate(bool((sd > 0).all()), "every process tag varies over the fit rows")
A = (A - mu) / sd
assumption(S, f"PCA fit rows = the full cleaned matrix ({len(A)} rows); each process tag standardized over these rows "
              "(zero mean, unit variance)")
pca = PCA(n_components=N_PCS, svd_solver="full", random_state=SEED)
scores = pca.fit_transform(A)
del A
pcs = [f"PC{i + 1}" for i in range(N_PCS)]
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
gate(pca.n_components_ == N_PCS and scores.shape[1] == N_PCS, f"PCA has exactly {N_PCS} components")
gate(bool(np.isfinite(scores).all()), "PC scores have no NaN / inf")
log(f"variance retained by {N_PCS} PCs: {100 * cum[-1]:.2f}% (PC1 {100 * evr[0]:.2f}%)")

# ---- scores, standardized to unit variance for Stage 3
scores = scores / scores.std(axis=0)
assumption(S, "PC scores standardized to unit variance (each divided by its std over the fit rows) before Stage 3")
pd.DataFrame(scores.astype("float32"), index=X.index, columns=pcs).to_parquet(DATA / "pc_scores.parquet")

# ---- loadings (26 x tags), variance, top tags and unit shares per PC
load = pd.DataFrame(pca.components_, index=pd.Index(pcs, name="pc"), columns=X.columns)
load.to_csv(DATA / "pca_loadings.csv")
pd.DataFrame({"pc": pcs, "explained_variance": pca.explained_variance_, "explained_variance_ratio": evr,
              "cumulative": cum}).to_csv(DATA / "pca_variance.csv", index=False)
top_rows = []
for pc in pcs:
    s = load.loc[pc]
    for rank, t in enumerate(s.abs().sort_values(ascending=False).index[:TOP_LOAD_N], start=1):
        top_rows.append({"pc": pc, "rank": rank, "tag": t, "unit": unit_of[t], "loading": float(s[t]),
                         "squared_loading": float(s[t] ** 2)})
top = pd.DataFrame(top_rows)
top.to_csv(TAB / "pca_top_tags.csv", index=False)
share = (load ** 2).T.groupby(pd.Series(X.columns, index=X.columns).map(unit_of)).sum().T   # rows sum to 1
units = [u for u in UNIT_FOLDERS if u in share.columns] + sorted(set(share.columns) - set(UNIT_FOLDERS))
share = share[units]
share.to_csv(TAB / "pca_unit_share.csv")
gate(bool(np.allclose(share.sum(axis=1), 1.0)), "squared-loading unit shares sum to 1 for every PC")
dominant = share.idxmax(axis=1)
for pc in pcs[:6]:
    log(f"  {pc}: {100 * evr[pcs.index(pc)]:.2f}% variance; dominant unit {dominant[pc]} "
        f"({100 * share.loc[pc].max():.0f}%); top tag {top[top['pc'] == pc]['tag'].iloc[0]}")

reg["status"] = ["used_in_pca" if t in load.columns else st for t, st in zip(reg["tag"], reg["status"])]
write_registry(reg)
seconds = time.time() - t0
save_json({"n_process_tags": X.shape[1], "rows": len(X), "n_pcs": N_PCS, "evr": evr.tolist(), "cumulative": cum.tolist(),
           "cum_variance": float(cum[-1]), "dominant_unit": dominant.to_dict(),
           "dominant_share": share.max(axis=1).to_dict(), "seconds": seconds}, "s2_summary.json")
done(S, seconds)
