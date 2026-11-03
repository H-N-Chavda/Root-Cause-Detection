"""Run the pipeline: python unit_pca_pc_t2/main.py (paths in unit_pca_pc_t2/.env, parameters in config.py)."""
import numpy as np
import pandas as pd
from scipy.stats import norm

import config as C
import helpers as H


def main():
    """Load and clean, unit-wise PCA, append flare_net, PC algorithm, DAG, causation-based T² on the pre-flare windows."""
    H.check_packages()
    P = C.load_paths()
    d = H.make_output_dirs(P.out)
    H.start_log(d["logs"] / "run.log")

    # 1. load the 13 units and the flare tags, build flare_net, find flare events, clean
    reg, frames, intervals = H.load_all_tags(P)
    X, interval = H.join_and_resample(frames, intervals, reg)
    X, reg, net, status = H.build_flare_net(X, reg, P)
    events, thr = H.detect_flare_events(net, interval)
    H.gate(len(events) > 0, "at least one flare event in the window")
    events.to_csv(d["tables"] / "flare_events.csv", index=False)
    H.crosscheck_flare_events_ta(events)
    X, status = H.clean_matrix(X, reg, status)
    flare, pre_eid, noc = H.label_flare_rows(X.index, events, interval, X.shape[1])
    Z, mu, sd, status, pc_only = H.standardize_with_noc(X, reg, noc, status)
    del X
    H.check_flare_source_agreement(Z[C.FLARE_NET], float(mu[C.FLARE_NET]), float(sd[C.FLARE_NET]), net, thr, flare,
                                   d["tables"] / "flare_source_mismatch.csv")

    # 2. unit-wise PCA on the process tags (flare_net is not part of any unit)
    scores, per_unit, loadings = H.fit_unit_pca(Z, reg, P.unit_folders, noc)
    per_unit.to_csv(d["tables"] / "pcs_per_unit.csv", index=False)
    loadings.rename_axis("pc").to_csv(d["tables"] / "pca_loadings.csv")
    reg["status"] = [("used_in_pca" if status.get(t) == "in_matrix" and t != C.FLARE_NET else status.get(t, ""))
                     for t in reg["tag"]]
    reg.to_csv(P.out / "tag_registry.csv", index=False)

    # 3. stack the PCs, append flare_net, PC algorithm, CPDAG -> DAG
    N = H.append_flare_tags(scores, Z[[C.FLARE_NET]], noc, pc_only)
    names, edges, drops = H.run_pc_algorithm(N, [C.FLARE_NET])
    edges.to_csv(d["graph"] / "pc_edges.csv", index=False)
    drops.to_csv(d["graph"] / "pc_dropped_nodes.csv", index=False)
    parents, dag_edges = H.cpdag_to_dag(edges, names, {C.FLARE_NET})
    dag_edges.to_csv(d["graph"] / "dag_edges.csv", index=False)

    # 4. causation-based T² on the PCs only, using the DAG's parent sets
    nodes, pa = H.pc_subgraph_parents(parents, {C.FLARE_NET})
    Xs, S = H.noc_correlation(scores[nodes].to_numpy(dtype=np.float64), noc)
    B, sigma2, coef, floored = H.fit_causal_regressions(S, nodes, pa)
    coef.to_csv(d["t2"] / "path_coefficients.csv", index=False)
    pd.DataFrame({"node": nodes, "n_parents": [len(pa[t]) for t in nodes], "sigma2": sigma2}).to_csv(
        d["t2"] / "conditional_variance.csv", index=False)
    T = H.decompose_t2_causal(Xs, B, sigma2)
    t2 = H.check_sum_identity(T, Xs, B, sigma2)
    H.check_noc_terms(T, noc, nodes, floored)
    t2_sample, df_sample = H.compute_t2(Xs, S)
    p = len(nodes)
    t2_lim = H.t2_control_limit(p)
    z_lim = float(norm.ppf(1 - C.ALPHA0 / p / 2))
    gap = np.abs(t2_sample - t2) / np.maximum(t2_sample, 1e-12)
    H.log(f"p = {p} PCs; T² limit chi²_{p}(1 - {C.ALPHA0}) = {t2_lim:.2f}; per-term limit z(alpha'/2) = {z_lim:.3f}, "
          f"alpha' = {C.ALPHA0}/{p}; NOC rows above the T² limit: {100 * (t2[noc] > t2_lim).mean():.2f}% "
          f"(expected about {100 * C.ALPHA0:g}% under the model)")
    H.log(f"Proposition 1 gap between sample-Σ T² (df {df_sample}) and the BN T²: median {np.median(gap[noc]):.3f} on NOC, "
          f"{np.median(gap[pre_eid > 0]) if (pre_eid > 0).any() else float('nan'):.3f} in pre-flare windows")
    pd.DataFrame({"T2": t2, "T2_sample_sigma": t2_sample, "above_limit": t2 > t2_lim, "noc": noc, "flare_event": flare,
                  "preflare_event_id": pre_eid}, index=Z.index).to_parquet(d["t2"] / "t2_all_rows.parquet")

    # 5. pre-flare windows: T² and its decomposition against the NOC model
    w = pre_eid > 0
    terms = pd.DataFrame(T[w], index=Z.index[w], columns=nodes)
    terms.insert(0, "event_id", pre_eid[w])
    terms.insert(1, "T2", t2[w])
    terms.to_parquet(d["t2"] / "preflare_terms.parquet")
    pd.DataFrame(T.astype(np.float32), index=Z.index, columns=nodes).to_parquet(d["t2"] / "terms_all_rows.parquet")
    net.to_frame().to_parquet(d["tables"] / "flare_net_raw.parquet")
    N.astype(np.float32).to_parquet(d["graph"] / "node_matrix.parquet")
    first = H.evaluate_preflare_windows(Z.index, pre_eid, events, T, t2, t2_lim, z_lim, nodes)
    first.to_csv(d["t2"] / "preflare_first_ooc.csv", index=False)
    pd.DataFrame([{"p": p, "t2_limit": t2_lim, "z_limit": z_lim, "alpha0": C.ALPHA0, "flare_threshold": thr,
                   "interval": str(interval), "n_events": len(events), "n_noc_rows": int(noc.sum())}]).to_csv(
        d["t2"] / "limits.csv", index=False)
    H.log("pipeline finished")


if __name__ == "__main__":
    main()
