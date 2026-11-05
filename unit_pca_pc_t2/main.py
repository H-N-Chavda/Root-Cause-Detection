"""Run the pipeline: python unit_pca_pc_t2/main.py (paths in unit_pca_pc_t2/.env, parameters in config.py).

Then draw the figures with plots.py (rescore.py re-scores saved outputs without re-running PC) and write SUMMARY.md with summary.py; both only read the outputs saved here.
"""
import numpy as np
import pandas as pd

import config as C
import evaluation as E
import helpers as H


def main() -> None:
    """Run every step in order and save all outputs under OUT_DIR.

    Steps: load the 13 units and the flare tags -> flare_net and flare events (>= 30 min) -> clean -> label flare,
    pre-flare and NOC rows -> standardize with NOC -> unit-wise PCA -> PC algorithm on the PCs only -> DAG among PCs
    + every PC -> flare_net -> causal T² decomposition on the PCs (no flare data) -> detection in the pre-flare windows
    -> rescore (moving-window T², TP / FN / FP blocks, F1) and RCA_report.xlsx.

    Raises:
        SystemExit: If any check (gate) fails; the log says which one.
    """
    H.check_packages()
    P = C.load_paths()
    d = H.make_output_dirs(P.out)
    H.start_log(d["logs"] / "run.log")

    # 1. load, flare_net, flare events, cleaning, row labels, standardization
    reg, frames, intervals = H.load_all_tags(P)
    X, interval = H.join_on_time_grid(frames, intervals, reg)
    parts = X.reindex(columns=[P.flare_1_tag, P.flare_2_tag, P.flare_recovery_tag]).set_axis(
        ["flare_1", "flare_2", "recovery"], axis=1)
    X, reg, net, status = H.build_flare_net(X, reg, P)
    events, short, thr = H.detect_flare_events(net, interval)
    H.gate(len(events) > 0, f"at least one flare event of >= {C.FLARE_MIN_DURATION_MIN} min in the window")
    events.to_csv(d["tables"] / "flare_events.csv", index=False)
    short.to_csv(d["tables"] / "short_flares.csv", index=False)
    H.crosscheck_flare_events_ta(events)
    X, status = H.clean_matrix(X, reg, status)
    flare, short_flare, pre_eid, noc = H.label_flare_rows(X.index, events, short, interval, X.shape[1])
    Z, mu, sd, status, _ = H.standardize_with_noc(X, reg, noc, status)
    del X
    H.check_flare_source_agreement(Z[C.FLARE_NET], float(mu[C.FLARE_NET]), float(sd[C.FLARE_NET]), net, thr,
                                   flare | short_flare, d["tables"] / "flare_source_mismatch.csv")

    # 2. unit-wise PCA on the process tags (flare_net is not part of any unit)
    scores, per_unit, loadings = H.fit_unit_pca(Z, reg, P.unit_folders, noc)
    per_unit.to_csv(d["tables"] / "pcs_per_unit.csv", index=False)
    loadings.rename_axis("pc").to_csv(d["tables"] / "pca_loadings.csv")
    reg["status"] = [("used_in_pca" if status.get(t) == "in_matrix" and t != C.FLARE_NET else status.get(t, ""))
                     for t in reg["tag"]]
    reg.to_csv(P.out / "tag_registry.csv", index=False)

    # 3. PC algorithm on the PCs only, DAG among PCs, then every PC -> flare_net
    names, edges, drops = H.run_pc_algorithm(scores)
    edges.to_csv(d["graph"] / "pc_edges.csv", index=False)
    drops.to_csv(d["graph"] / "pc_dropped_nodes.csv", index=False)
    parents, dag_edges, dag_stats = H.cpdag_to_dag(edges, names)
    H.connect_pcs_to_flare(dag_edges, names).to_csv(d["graph"] / "dag_edges.csv", index=False)
    pd.DataFrame([dag_stats]).to_csv(d["graph"] / "dag_stats.csv", index=False)

    # 4. causation-based T² on the PCs only (no flare data), using the DAG's parent sets
    nodes, pa = names, parents
    Xs, S = H.noc_correlation(scores[nodes].to_numpy(dtype=np.float64), noc)
    B, sigma2, coef, floored = H.fit_causal_regressions(S, nodes, pa)
    coef.to_csv(d["t2"] / "path_coefficients.csv", index=False)
    T = H.decompose_t2_causal(Xs, B, sigma2)
    t2_terms, identity_err = H.check_sum_identity(T, Xs, B, sigma2)
    H.check_noc_terms(T, noc, nodes, floored)
    t2, rank_s = H.compute_t2(Xs, S)
    p, m = len(nodes), int(noc.sum())
    t2_lim = C.T2_LIMIT_SCALE * H.t2_threshold(m, p, 0, C.ALPHA0)
    term_lim = np.array([C.T2_LIMIT_SCALE * H.t2_threshold(m, 1, len(pa[t]), C.ALPHA_TERM) for t in nodes])
    pd.DataFrame({"node": nodes, "n_parents": [len(pa[t]) for t in nodes], "sigma2": sigma2,
                  "t2_limit": term_lim}).to_csv(d["t2"] / "term_limits.csv", index=False)
    noc_above, noc_term_rate = 100 * (t2[noc] > t2_lim).mean(), 100 * (T[noc] ** 2 > term_lim).mean(axis=0)
    gap = np.abs(t2 - t2_terms) / np.maximum(t2, 1e-12)
    H.log(f"p = {p} PCs, m = {m} NOC rows; T² limit (report Eq. 13, corrected) = {t2_lim:.2f} at alpha {C.ALPHA0}; "
          f"NOC rows above it: {noc_above:.2f}% (expected about {100 * C.ALPHA0:g}%)")
    H.log(f"per-PC limits on T²_j (Eq. 13 with p = 1, k = number of parents, alpha {C.ALPHA_TERM}): "
          f"{term_lim.min():.3f} to {term_lim.max():.3f}; NOC rows above them: mean {noc_term_rate.mean():.2f}% per PC "
          f"(expected about {100 * C.ALPHA_TERM:g}%)")
    H.log(f"overall T² = Zᵀ inv(R) Z (rank of R {rank_s} of {p}); relative gap to the sum of the causal "
          f"terms: median {np.median(gap[noc]):.3f} on NOC, "
          f"{np.median(gap[pre_eid > 0]) if (pre_eid > 0).any() else float('nan'):.3f} in pre-flare windows")
    pd.DataFrame({"T2": t2, "T2_sum_terms": t2_terms, "above_limit": t2 > t2_lim, "noc": noc, "flare_event": flare,
                  "short_flare": short_flare, "preflare_event_id": pre_eid},
                 index=Z.index).to_parquet(d["t2"] / "t2_all_rows.parquet")

    # 5. pre-flare windows: was each flare event found, and which PCs were above their limits
    w = pre_eid > 0
    terms = pd.DataFrame(T[w], index=Z.index[w], columns=nodes)
    terms.insert(0, "event_id", pre_eid[w])
    terms.insert(1, "T2", t2[w])
    terms.to_parquet(d["t2"] / "preflare_terms.parquet")
    pd.DataFrame(T.astype(np.float32), index=Z.index, columns=nodes).to_parquet(d["t2"] / "terms_all_rows.parquet")
    pd.concat([net, parts], axis=1).to_parquet(d["tables"] / "flare_net_raw.parquet")
    scores.astype(np.float32).to_parquet(d["graph"] / "pc_scores.parquet")
    first = H.evaluate_preflare_windows(Z.index, pre_eid, events, T, t2, t2_lim, term_lim, nodes)
    first.to_csv(d["t2"] / "preflare_first_ooc.csv", index=False)
    rule = (f"median + {C.FLARE_MAD_K:g} x 1.4826 MAD" if C.FLARE_USE_MAD_THRESHOLD else "manual FLARE_THRESHOLD")
    pd.DataFrame([{"p": p, "n_noc_rows": m, "t2_limit": t2_lim, "alpha0": C.ALPHA0, "alpha_term": C.ALPHA_TERM,
                   "flare_threshold": thr, "flare_threshold_rule": rule, "interval": str(interval),
                   "n_events": len(events), "n_events_found": int(first["detected"].sum()),
                   "pct_noc_above_t2_limit": noc_above, "pct_noc_above_term_limit_mean": float(noc_term_rate.mean()),
                   "pct_preflare_above_t2_limit": 100 * float((t2[w] > t2_lim).mean()) if w.any() else float("nan"),
                   "sum_identity_max_rel_error": identity_err,
                   "gap_t2_vs_sum_terms_median_noc": float(np.median(gap[noc]))}]).to_csv(d["t2"] / "limits.csv",
                                                                                           index=False)
    E.rescore(P.out)  # moving-window baseline, alarms and metrics (also runnable alone: rescore.py)
    E.write_excel(P.out)
    H.log("pipeline finished")


if __name__ == "__main__":
    main()
