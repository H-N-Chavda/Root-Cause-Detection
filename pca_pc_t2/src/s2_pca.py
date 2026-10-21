"""Stage 2 - PCA with k = 26 on the z-scored TRAIN matrix, plus the full eigen-spectrum for the plots.

Writes outputs/data/{pca.npz, T_train.parquet, T_test.parquet, s2_summary.json} and PCA tables.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import (DATA, K, PCS, SEED, TAB, gate, loading_mass, log, present_units, read_tag_map,
                    save_json, stage)

stage("Stage 2 - PCA")
X_train = pd.read_parquet(DATA / "X_train.parquet")
X_test = pd.read_parquet(DATA / "X_test.parquet")
cols = list(X_train.columns)
tm = read_tag_map()
unit_of = dict(zip(tm["column"], tm["unit"]))
units = present_units(cols, unit_of)
gate(X_train.shape[1] >= K and list(X_test.columns) == cols, f"{len(cols)} tags >= k = {K}; TRAIN/TEST columns match")

Xtr = X_train.to_numpy()
pca = PCA(n_components=K, svd_solver="full", random_state=SEED).fit(Xtr)
lam = pca.explained_variance_
P = pca.components_.T                                    # tags x 26
T_train = pca.transform(Xtr)
T_test = pca.transform(X_test.to_numpy())

# Full spectrum from the TRAIN covariance (same n-1 normalisation as sklearn; avoids a second big SVD).
eig_full = np.clip(np.linalg.eigvalsh(np.cov(Xtr, rowvar=False))[::-1], 0, None)
gate(np.allclose(eig_full[:K], lam, rtol=1e-6, atol=1e-9), "first 26 covariance eigenvalues equal the PCA eigenvalues")
evr = eig_full / eig_full.sum()
cum = np.cumsum(evr)
n_for = {f"{int(t * 100)}%": int(min(np.searchsorted(cum, t) + 1, len(cum))) for t in (0.80, 0.90, 0.95, 0.99)}
log(f"cumulative variance at {K} PCs: {cum[K - 1]:.4f}; PCs needed: {n_for}")

# Verification: reconstruction from 26 PCs and orthonormal loadings.
X_hat = pca.inverse_transform(T_train)
r2 = 1 - ((Xtr - X_hat) ** 2).sum() / ((Xtr - Xtr.mean(axis=0)) ** 2).sum()
del X_hat
orth_err = float(np.abs(P.T @ P - np.eye(K)).max())
log(f"reconstruction R^2 (TRAIN, 26 PCs) = {r2:.4f}; max |P'P - I| = {orth_err:.2e}")
gate(np.allclose(P.T @ P, np.eye(K)), "np.allclose(P.T @ P, I)")

lm = loading_mass(P, cols, unit_of, units)
dominant = pd.DataFrame({"PC": PCS, "dominant_unit": lm.idxmax(axis=1).values, "share": lm.max(axis=1).values})
for r in dominant.head(6).itertuples():
    log(f"  {r.PC}: dominant unit {r.dominant_unit} ({r.share:.0%} of squared loadings)")

np.savez(DATA / "pca.npz", P=P, lam=lam, mean=pca.mean_, eig_full=eig_full, columns=np.array(cols))
pd.DataFrame(T_train, index=X_train.index, columns=PCS).to_parquet(DATA / "T_train.parquet")
pd.DataFrame(T_test, index=X_test.index, columns=PCS).to_parquet(DATA / "T_test.parquet")
pd.DataFrame({"PC": np.arange(1, len(eig_full) + 1), "eigenvalue": eig_full, "explained_ratio": evr,
              "cumulative": cum}).to_csv(TAB / "pca_variance.csv", index=False)
pd.DataFrame({"threshold": list(n_for), "n_pcs": list(n_for.values())}).to_csv(TAB / "pca_thresholds.csv", index=False)
pd.DataFrame(P, index=pd.Index(cols, name="column"), columns=PCS).to_csv(TAB / "loadings.csv")
lm.rename_axis("PC").to_csv(TAB / "loading_mass.csv")
dominant.to_csv(TAB / "pca_dominant_units.csv", index=False)

save_json({"k": K, "n_tags": len(cols), "cum_var_k": float(cum[K - 1]), "n_pcs_for": n_for,
           "reconstruction_r2": float(r2), "max_orth_err": orth_err,
           "dominant_units": dominant.head(10).to_dict("records")}, "s2_summary.json")
gate(True, "PCA saved: eigenvalues, variance ratios, loadings P, scores T_train / T_test")
