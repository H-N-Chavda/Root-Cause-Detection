"""Stage 4 - causation-based T² decomposition (Li, Jin & Shi 2008, J. Quality Technology 40(1), 46-58).

The Stage 3 DAG is the Bayesian network. Tags are standardized with reference-period statistics and Σ is their
correlation matrix on the reference period (the paper assumes Σ known; here it is estimated on that period).
- Overall T² = Zᵀ Σ⁻¹ Z (Eq. 7), control limit chi²_p(1 - ALPHA0) (Eq. 6: T² ~ chi²_p).
- Tag j with parents PA(j): β = Σ_PP⁻¹ Σ_Pj (path coefficients by multiple regression on Σ, p. 49),
  μ_j·PA(j) = Σ_k β_k Z_pa_k (Eq. 10), σ²_j·PA(j) = 1 - Σ_k β_k ρ(Z_j, pa_k) (Eqs. 5 and 11),
  T_j·PA(j) = (Z_j - μ_j·PA(j)) / σ_j·PA(j) (Eq. 9). A tag with no parents gets T_j = Z_j.
- Causation-based decomposition T² = Σ_j T²_j·PA(j) (Definition 2, Eq. 13). It is exact only for the true graph
  (Proposition 1), so it is NOT asserted; the relative discrepancy |T² - Σ_j T²_j·PA(j)| / T² is reported per
  timestamp as a diagnostic of graph quality.
- |T_j·PA(j)| > z_{α'/2} with Bonferroni α' = ALPHA0 / p (the paper's simulation setup) is significant: a mean
  shift that originates at tag j instead of being propagated from its parents ("Monitoring and Diagnosis", Fig. 5).

Writes outputs/data/{T_terms.parquet, t2.parquet, coefficients.csv, s4_summary.json}, outputs/tables/
{root_cause_tags.csv, root_cause_by_event.csv, reference_check.csv} and sets status 'used' in tag_registry.csv.
"""
import time

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import chi2, kurtosis, norm, skew

from common import (ALPHA0, DATA, EVENT_DIAG_WINDOW, MATCH_WINDOW, SIGMA2_MIN, TAB, TOP_N, assumption, clear_notes,
                    deviation, done, gate, load_json, log, read_flare_events, read_registry, save_json, stage,
                    write_registry)

S = "s4_t2"
CHUNK = 50000
t0 = time.time()
stage("Stage 4 - causation-based T² decomposition")
clear_notes(S)
Z = pd.read_parquet(DATA / "Z.parquet")
meta = pd.read_parquet(DATA / "meta.parquet")
parents = load_json("parents.json")
reg = read_registry()
events = read_flare_events()
unit_of = dict(zip(reg["tag"], reg["unit"]))
flare_of = dict(zip(reg["tag"], reg["is_flare"]))
gate(meta.index.equals(Z.index), "Z and meta share the same timestamps")
gate(set(parents) <= set(Z.columns), "every DAG node is a column of the combined matrix")
ref = meta["reference"].to_numpy(dtype=bool)
flare_rows = meta["flare"].to_numpy(dtype=bool)

# ---- tags with zero variance in the reference period cannot enter Σ (Stage 1 lets only flare tags through)
nodes = list(parents)
sd0 = Z.loc[ref, nodes].to_numpy(dtype=np.float64).std(axis=0, ddof=1)
excluded = [t for t, v in zip(nodes, sd0) if not v > 1e-12]
tags = [t for t in nodes if t not in excluded]
if excluded:
    assumption(S, f"left out of T² (zero variance in the reference period): {excluded}")
    bad = [t for t in excluded if not flare_of[t]]
    if bad:
        deviation(S, f"non-flare tags with zero reference variance left out of T²; their children lose them as parents: {bad}")
p = len(tags)
pos = {t: i for i, t in enumerate(tags)}
par = {t: [q for q in parents[t] if q in pos] for t in tags}

# ---- standardize on the reference period and estimate Σ there
Zr = Z.loc[ref, tags].to_numpy(dtype=np.float64)
m, s = Zr.mean(axis=0), Zr.std(axis=0, ddof=1)
Xr = (Zr - m) / s
Sigma = Xr.T @ Xr / (len(Xr) - 1)        # = correlation matrix of the reference period
Sigma = (Sigma + Sigma.T) / 2
np.fill_diagonal(Sigma, 1.0)
del Zr
assumption(S, f"Σ = correlation matrix of the {len(Xr)} reference rows; tags re-standardized with reference mean "
              "and std so that Σ is exactly their covariance")
df = p
try:
    L = np.linalg.cholesky(Sigma)
    W = None
except np.linalg.LinAlgError:
    vals, vecs = np.linalg.eigh(Sigma)
    keepv = vals > 1e-10 * vals.max()
    df = int(keepv.sum())
    W = vecs[:, keepv] / np.sqrt(vals[keepv])
    deviation(S, f"Σ is singular: T² computed with a pseudo-inverse on {df} of {p} eigen-directions, limit chi²_{df}")
t2_lim = float(chi2.ppf(1 - ALPHA0, df))
alpha_prime = ALPHA0 / p
z_lim = float(norm.ppf(1 - alpha_prime / 2))
log(f"p = {p} tags; T² limit chi²_{df}(1 - {ALPHA0}) = {t2_lim:.2f}; per-tag limit z(α'/2) = {z_lim:.3f} "
    f"with α' = {ALPHA0}/{p} = {alpha_prime:.2e}")

# ---- regression of each tag on its parents using Σ (Eqs. 9-11)
B = np.zeros((p, p))                      # column j: coefficients of tag j on its parents
sigma2 = np.ones(p)
coef_rows, floored = [], []
for j, t in enumerate(tags):
    P = [pos[q] for q in par[t]]
    if P:
        beta = np.linalg.lstsq(Sigma[np.ix_(P, P)], Sigma[P, j], rcond=None)[0]
        B[P, j] = beta
        sigma2[j] = 1.0 - beta @ Sigma[P, j]
        coef_rows += [{"tag": t, "parent": tags[q], "beta": float(b)} for q, b in zip(P, beta)]
    if sigma2[j] < SIGMA2_MIN:
        floored.append(t)
        sigma2[j] = SIGMA2_MIN
sig = np.sqrt(sigma2)
pd.DataFrame(coef_rows, columns=["tag", "parent", "beta"]).to_csv(DATA / "coefficients.csv", index=False)
if floored:
    deviation(S, f"σ²_j·PA(j) below {SIGMA2_MIN:g} (parents explain the tag almost exactly), floored: {floored}")

# ---- T², T_j·PA(j), Proposition 1 discrepancy and significance, chunk by chunk
n = len(Z)
cols = Z.columns.get_indexer(tags)
Zv = Z.to_numpy()
T = np.empty((n, p), dtype=np.float32)
T2 = np.empty(n)
sumT2 = np.empty(n)
n_sig_row = np.empty(n, dtype=np.int32)
for a in range(0, n, CHUNK):
    X = (Zv[a:a + CHUNK][:, cols].astype(np.float64) - m) / s
    if W is None:
        T2[a:a + CHUNK] = (solve_triangular(L, X.T, lower=True) ** 2).sum(axis=0)     # Eq. 7
    else:
        T2[a:a + CHUNK] = ((X @ W) ** 2).sum(axis=1)
    Tc = (X - X @ B) / sig                                                            # Eqs. 9-10
    sumT2[a:a + CHUNK] = (Tc ** 2).sum(axis=1)                                        # Eq. 13
    n_sig_row[a:a + CHUNK] = (np.abs(Tc) > z_lim).sum(axis=1)
    T[a:a + CHUNK] = Tc
del Zv, X, Tc
ooc = T2 > t2_lim
rel = np.abs(T2 - sumT2) / np.maximum(T2, 1e-12)
log(f"T² out of control: {100 * ooc.mean():.2f}% of all rows, {100 * ooc[ref].mean():.2f}% of reference rows "
    f"(expected about {100 * ALPHA0:g}%)")

# ---- reference period: T_j·PA(j) should be about N(0, 1)
Tr = T[ref].astype(np.float64)
chk = pd.DataFrame({"tag": tags, "unit": [unit_of[t] for t in tags], "n_parents": [len(par[t]) for t in tags],
                    "mean": Tr.mean(axis=0), "std": Tr.std(axis=0, ddof=1), "skew": skew(Tr, axis=0),
                    "excess_kurtosis": kurtosis(Tr, axis=0),
                    "pct_beyond_limit": 100 * (np.abs(Tr) > z_lim).mean(axis=0), "floored": [t in floored for t in tags]})
chk.to_csv(TAB / "reference_check.csv", index=False)
del Tr
ok = chk[~chk["floored"]]
max_mean, max_sd_dev = float(ok["mean"].abs().max()), float((ok["std"] - 1).abs().max())
gate(max_mean < 0.05 and max_sd_dev < 0.05,
     f"reference T_j·PA(j) about N(0,1): max |mean| = {max_mean:.2e}, max |std - 1| = {max_sd_dev:.2e} (tags not floored)")
sample = chk.iloc[np.unique(np.linspace(0, p - 1, min(10, p)).astype(int))]

# ---- per-tag root-cause counts
n_sig,n_sig_ooc, n_sig_flare, max_abs = (np.zeros(p) for _ in range(4))
for a in range(0, n, CHUNK):
    g = np.abs(T[a:a + CHUNK]) > z_lim
    n_sig += g.sum(axis=0)
    n_sig_ooc += g[ooc[a:a + CHUNK]].sum(axis=0)
    n_sig_flare += g[flare_rows[a:a + CHUNK]].sum(axis=0)
    max_abs = np.maximum(max_abs, np.abs(T[a:a + CHUNK]).max(axis=0))
rc = pd.DataFrame({"tag": tags, "unit": [unit_of[t] for t in tags], "is_flare": [flare_of[t] for t in tags],
                   "n_parents": [len(par[t]) for t in tags], "sigma2": sigma2,
                   "pct_significant": 100 * n_sig / n, "pct_significant_when_T2_ooc": 100 * n_sig_ooc / max(ooc.sum(), 1),
                   "pct_significant_in_flare_rows": 100 * n_sig_flare / max(flare_rows.sum(), 1),
                   "pct_significant_reference": chk["pct_beyond_limit"].to_numpy(), "max_abs_T": max_abs})
rc = rc.sort_values("pct_significant_when_T2_ooc", ascending=False)
rc.to_csv(TAB / "root_cause_tags.csv", index=False)

# ---- per flare event: decompose the first out-of-control T² (paper Fig. 5) and the EVENT_DIAG_WINDOW after it
idx = Z.index
ev_rows, no_ooc = [], []
for ev in events.itertuples(index=False):
    w = (idx >= ev.start - MATCH_WINDOW) & (idx <= ev.end) & ooc
    if not w.any():
        no_ooc.append(int(ev.event_id))
        continue
    i0 = int(np.argmax(w))
    t_first = idx[i0]
    win = (idx >= t_first) & (idx <= t_first + EVENT_DIAG_WINDOW)
    Tw = T[win].astype(np.float64)
    frac, mean_abs, mean_T = (np.abs(Tw) > z_lim).mean(axis=0), np.abs(Tw).mean(axis=0), Tw.mean(axis=0)
    for rank, j in enumerate(np.lexsort((-mean_abs, -frac))[:TOP_N], start=1):
        ev_rows.append({"event_id": int(ev.event_id), "event_start": ev.start, "event_end": ev.end,
                        "first_ooc": t_first, "window_rows": int(win.sum()), "rank": rank, "tag": tags[j],
                        "unit": unit_of[tags[j]], "frac_significant": frac[j], "mean_abs_T": mean_abs[j],
                        "mean_T": mean_T[j], "T_at_first_ooc": float(T[i0, j]),
                        "significant_at_first_ooc": bool(abs(T[i0, j]) > z_lim)})
ev_cols = ["event_id", "event_start", "event_end", "first_ooc", "window_rows", "rank", "tag", "unit",
           "frac_significant", "mean_abs_T", "mean_T", "T_at_first_ooc", "significant_at_first_ooc"]
pd.DataFrame(ev_rows, columns=ev_cols).to_csv(TAB / "root_cause_by_event.csv", index=False)
assumption(S, f"per flare event: first out-of-control T² from {MATCH_WINDOW} before onset to the event end; tags ranked by "
              f"the share of significant |T_j·PA(j)| (then mean |T|) over the {EVENT_DIAG_WINDOW} after it")
if no_ooc:
    log(f"flare events without an out-of-control T² in their window: {no_ooc}")

# ---- save
pd.DataFrame(T, index=idx, columns=tags).to_parquet(DATA / "T_terms.parquet")
pd.DataFrame({"T2": T2, "sum_T2_terms": sumT2, "rel_discrepancy": rel, "ooc": ooc, "n_sig_tags": n_sig_row},
             index=idx).to_parquet(DATA / "t2.parquet")
reg["status"] = ["used" if t in pos else st for t, st in zip(reg["tag"], reg["status"])]
write_registry(reg)
gate(int((reg["status"] == "used").sum()) == p, f"tag_registry.csv marks exactly the {p} T² tags as 'used'")


def q(x, v):
    return float(np.quantile(x, v)) if len(x) else float("nan")


disc = {name: {"median": q(rel[mask], 0.5), "p90": q(rel[mask], 0.9), "p99": q(rel[mask], 0.99)}
        for name, mask in (("all", np.ones(n, bool)), ("reference", ref), ("T2_ooc", ooc), ("flare", flare_rows))}
log(f"Proposition 1 relative discrepancy (median / p90): all {disc['all']['median']:.3f} / {disc['all']['p90']:.3f}, "
    f"reference {disc['reference']['median']:.3f} / {disc['reference']['p90']:.3f}")
seconds = time.time() - t0
save_json({"p": p, "df": df, "rows": n, "n_reference_rows": int(ref.sum()), "alpha0": ALPHA0, "alpha_prime": alpha_prime,
           "z_lim": z_lim, "t2_lim": t2_lim, "excluded": excluded, "floored": floored,
           "n_root_tags": int(sum(1 for t in tags if not par[t])),
           "pct_ooc_all": 100 * ooc.mean(), "pct_ooc_reference": 100 * ooc[ref].mean(),
           "pct_ooc_flare": 100 * ooc[flare_rows].mean() if flare_rows.any() else None,
           "pct_ooc_nonflare": 100 * ooc[~flare_rows].mean() if (~flare_rows).any() else None,
           "discrepancy": disc, "ref_max_abs_mean": max_mean, "ref_max_abs_std_dev": max_sd_dev,
           "ref_median_excess_kurtosis": float(chk["excess_kurtosis"].median()),
           "ref_pct_beyond_limit_mean": float(chk["pct_beyond_limit"].mean()), "ref_expected_pct": 100 * alpha_prime,
           "ref_sample": sample[["tag", "unit", "n_parents", "mean", "std", "excess_kurtosis", "pct_beyond_limit"]]
           .to_dict("records"),
           "events_without_ooc": no_ooc, "top_tags": rc.head(10)[["tag", "unit", "pct_significant_when_T2_ooc"]]
           .to_dict("records"), "seconds": seconds}, "s4_summary.json")
done(S, seconds)
