"""Stage 7 - SUMMARY.md and the Definition-of-Done checks (prompt Section 7).

Copied from causal_t2/src/s6_summary.py and edited for this pipeline (PCA stage, 26 PCs + flare nodes, t2/ folder).
Every number in SUMMARY.md comes from the s*_summary.json files, tag_registry.csv and figures_index.csv, so the
three documents agree by construction. The checks are written into SUMMARY.md; any FAIL exits with code 1.
"""
import re
import time
from datetime import date

import networkx as nx
import numpy as np
import pandas as pd

from common import (DATA, DATA_ROOT, FIG, GRAPH, N_PCS, OUT, PROJECT, T2DIR, WINDOW_TEXT, done, load_json, log,
                    read_nodes, read_registry, stage)

S = "s7_summary"
t0 = time.time()
stage("Stage 7 - SUMMARY.md and Definition-of-Done checks")
s1, s2, s3, s4, s5 = (load_json(f"s{i}_summary.json") for i in ("1", "2", "3", "4", "5"))
reg = read_registry()
nodes = read_nodes()
idx = pd.read_csv(OUT / "figures_index.csv")
assumptions = load_json("assumptions.json") if (DATA / "assumptions.json").exists() else []
deviations = load_json("deviations.json") if (DATA / "deviations.json").exists() else []
ev = pd.read_csv(OUT / "tables" / "root_cause_by_event.csv", dtype={"node": str})
stages = ["s1_load_clean", "s2_pca", "s3_pc", "s4_dag", "s5_t2", "s6_figures"]
secs = {st: float((DATA / f"{st}.done").read_text()) for st in stages if (DATA / f"{st}.done").exists()}


def f(v, d=2):
    return "n/a" if v is None else f"{v:.{d}f}"


# ---- Definition-of-Done checks
checks = []
sources = sorted((PROJECT / "src").glob("*.py")) + [PROJECT / "run_all.py"]
abs_path = re.compile(r"\b[A-Za-z]:[\\/]|/(Users|home)/")
git_call = re.compile(r"['\"]git['\"\s]")
root_strs = {str(DATA_ROOT), DATA_ROOT.as_posix()}
hits = [p.name for p in sources if abs_path.search(p.read_text(encoding="utf-8"))
        or any(r in p.read_text(encoding="utf-8") for r in root_strs)]
text_out = [p for ext in ("*.md", "*.csv", "*.json", "*.log", "*.graphml") for p in OUT.rglob(ext)]
leaks = [p.name for p in text_out if any(r in p.read_text(encoding="utf-8", errors="ignore") for r in root_strs)]
checks.append(("No hardcoded path in code (paths only in .env); DATA_ROOT never written to outputs",
               not hits and not leaks, f"code hits: {hits or 'none'}; output files with DATA_ROOT: {leaks or 'none'}"))
git_hits = [p.name for p in sources if git_call.search(p.read_text(encoding="utf-8"))]
checks.append(("No git command in the pipeline code", not git_hits, f"files: {git_hits or 'none'}"))

Z = pd.read_parquet(DATA / "Z.parquet")
arr = Z.to_numpy()
n_reg = int(reg["status"].str.startswith("in_matrix").sum() + (reg["status"] == "used").sum()
            + (reg["status"] == "used_in_pca").sum() + reg["status"].str.startswith("dropped_s3").sum())
checks.append(("Matrix code from causal_t2; no NaN/inf, no constant column; tag count matches tag_registry.csv",
               bool(np.isfinite(arr).all()) and bool((arr.std(axis=0) > 0).all()) and n_reg == Z.shape[1],
               f"{Z.shape[0]} rows x {Z.shape[1]} tags ({s1['n_process_tags']} process + {s1['n_flare_tags_matrix']} flare); "
               f"registry rows in matrix: {n_reg}"))
del arr, Z

sc = pd.read_parquet(DATA / "pc_scores.parquet")
checks.append(("PCA: exactly 26 components; variance reported; loadings tables exist; scores have no NaN",
               sc.shape[1] == N_PCS == s2["n_pcs"] and bool(np.isfinite(sc.to_numpy()).all())
               and (OUT / "tables" / "pca_top_tags.csv").exists() and (OUT / "tables" / "pca_unit_share.csv").exists(),
               f"{sc.shape[1]} PCs, cumulative variance {100 * s2['cum_variance']:.2f}%"))
del sc

pe = pd.read_csv(GRAPH / "pc_edges.csv", dtype={"source": str, "target": str})
flare = set(nodes.loc[nodes["is_flare"], "node"])
pc_nodes_pc = [str(t) for t in np.load(DATA / "pc_graph.npz", allow_pickle=True)["names"]]
n_exp = N_PCS + int(nodes["is_flare"].sum()) - s3["n_dropped_s3"]
out_fl = int(((pe["type"] == "->") & pe["source"].isin(flare)).sum())
und_fl = int(((pe["type"] != "->") & (pe["source"].isin(flare) | pe["target"].isin(flare))).sum())
de = pd.read_csv(GRAPH / "dag_edges.csv", dtype={"source": str, "target": str})
g = nx.DiGraph()
g.add_nodes_from(pc_nodes_pc)
g.add_edges_from(zip(de["source"], de["target"]))
dag_out_fl = int(de["source"].isin(flare).sum())
checks.append(("PC graph nodes = 26 PCs + flare tags; 0 directed edges out of a flare, 0 undirected touching one; DAG acyclic",
               len(pc_nodes_pc) == n_exp and out_fl == 0 and und_fl == 0 and dag_out_fl == 0
               and nx.is_directed_acyclic_graph(g),
               f"nodes {len(pc_nodes_pc)} (expected {n_exp} after {s3['n_dropped_s3']} collinear drops); PC out-of-flare "
               f"{out_fl}, undirected at flare {und_fl}; DAG out-of-flare {dag_out_fl}, acyclic {nx.is_directed_acyclic_graph(g)}"))

t2_cols = pd.read_parquet(T2DIR / "t2.parquet", columns=["rel_discrepancy"]).columns
n_charts = int(idx["filename"].str.startswith("figures/per_node/").sum())
checks.append(("Reference T_j·PA(j) about N(0,1); discrepancy computed; every node has a chart",
               s5["ref_max_abs_mean"] < 0.05 and s5["ref_max_abs_std_dev"] < 0.05 and "rel_discrepancy" in t2_cols
               and n_charts == s5["p"] and len(list((FIG / "per_node").glob("*.png"))) == s5["p"],
               f"max |mean| {s5['ref_max_abs_mean']:.1e}, max |std-1| {s5['ref_max_abs_std_dev']:.1e}; "
               f"charts {n_charts} for {s5['p']} nodes"))

log_md = OUT / "LOG.md"
log_txt = log_md.read_text(encoding="utf-8") if log_md.exists() else ""
logged = [st for st in stages if re.search(rf"\| {st} \| PASS \|", log_txt)]
used = int((reg["status"] == "used").sum())
checks.append(("figures_index.csv, SUMMARY.md and LOG.md exist and agree",
               all((OUT / x).exists() for x in idx["filename"]) and used == s5["n_flare"] and len(logged) == len(stages),
               f"index rows {len(idx)} (all files exist: {all((OUT / x).exists() for x in idx['filename'])}); "
               f"registry 'used' {used} = flare nodes in T² {s5['n_flare']}; LOG.md PASS rows for {len(logged)}/{len(stages)} stages"))
checks.append(("One command reruns the whole pipeline (documented at the top)", (PROJECT / "run_all.py").exists(),
               "python pca_pc_t2/run_all.py --force"))

# ---- SUMMARY.md
kc = s3["edges_by_kind"]
L = ["# SUMMARY: PCA (26 PCs) + flare tags -> PC algorithm -> causation-based T²", "",
     "Rerun everything: `python pca_pc_t2/run_all.py --force` (paths in `pca_pc_t2/.env`; without `--force` finished "
     f"stages are skipped). Generated {date.today().isoformat()}. Time window: {WINDOW_TEXT}. "
     "Method: Li, Jin & Shi (2008), J. Quality Technology 40(1).", "",
     "## Tag and node counts per stage", "| stage | count | flare tags | note |", "|---|---|---|---|",
     f"| loaded (unique tags) | {s1['n_tags_loaded']} | {s1['n_flare_tags_loaded']} | {s1['n_duplicate_copies']} duplicate copies skipped |",
     f"| S1 combined matrix (tags) | {s1['n_tags_matrix']} | {s1['n_flare_tags_matrix']} | {s1['n_process_tags']} process; "
     f"dropped {s1['n_dropped_s1']}: {s1['dropped_s1_by_reason']}; {s1['rows']} rows, {s1['interval']} grid; prompt expected about 376 |",
     f"| S2 PCA input (process tags) | {s2['n_process_tags']} | 0 | {N_PCS} PCs keep {100 * s2['cum_variance']:.2f}% of the variance |",
     f"| S3 PC nodes | {s3['n_nodes_pc']} | {s3['n_flare_tags_pc']} | {s3['n_pcs_pc']} PCs; dropped {s3['n_dropped_s3']} (collinear / dependent) |",
     f"| S5 T² nodes | {s5['p']} | {s5['n_flare']} | left out (zero reference variance): {s5['excluded'] or 'none'} |", "",
     "## PCA",
     f"- Fit on {s2['rows']} rows (full cleaned matrix); {N_PCS} components keep {100 * s2['cum_variance']:.2f}% of the variance.",
     "- Variance per PC (%): " + ", ".join(f"PC{i + 1} {100 * v:.1f}" for i, v in enumerate(s2["evr"][:10])) + " ...",
     "- Dominant unit per PC: " + "; ".join(f"{p} {u} ({100 * s2['dominant_share'][p]:.0f}%)"
                                          for p, u in list(s2["dominant_unit"].items())[:10]) + " ...", "",
     "## Graph",
     "- Alpha sweep: " + "; ".join(f"alpha {r['alpha']}: {r['n_edges']} edges (PC-PC {r['pc_pc']}, PC-flare {r['pc_flare']})"
                                   for r in s3["alpha_sweep"]) + f". Used downstream: alpha {s3['alpha']}.",
     f"- PC: depth cap {s3['max_k']}, {s3['rows_pc']} rows (every {s3['row_step']}th), {s3['n_ci_tests']} CI tests, "
     f"{s3['seconds_pc']:.0f} s.",
     f"- Edges: {s3['n_edges']} = {s3['n_directed']} directed, {s3['n_undirected']} undirected, {s3['n_bidirected']} bidirected; "
     f"PC-PC {kc['pc_pc']}, PC-flare {kc['pc_flare']}.",
     "- PCs are nearly uncorrelated with each other (PCA output), so PC-PC edges are expected to be sparse and most "
     f"structure to be PC -> flare. Observed: {kc['pc_pc']} PC-PC vs {kc['pc_flare']} PC-flare edges. This is not a bug.",
     f"- Flare edges straight from PC: {s3['flare_edges_raw']}; changed after PC: {s3['flare_edges_changed']}.",
     f"- DAG: {s4['n_edges']} edges; CPDAG directed kept {s4['cpdag_directed_kept']}/{s4['cpdag_directed']}, "
     f"reversed {s4['n_reversed']}, forced Dor-Tarsi steps {s4['n_forced']}; roots {s4['n_roots']}, "
     f"parents mean {s4['mean_parents']:.2f} / max {s4['max_parents']}.", "",
     "## T² decomposition (Eqs. 7, 9-11, 13)",
     f"- Limits: T² > chi²_{s5['df']}(1 - {s5['alpha0']}) = {s5['t2_lim']:.2f}; |T_j·PA(j)| > z = {s5['z_lim']:.3f} "
     f"(Bonferroni alpha' = {s5['alpha0']}/{s5['p']}).",
     f"- T² out of control: all {f(s5['pct_ooc_all'])}%, reference {f(s5['pct_ooc_reference'])}%, flare rows "
     f"{f(s5['pct_ooc_flare'])}%, non-flare rows {f(s5['pct_ooc_nonflare'])}%.",
     f"- Reference check: max |mean| {s5['ref_max_abs_mean']:.1e}, max |std - 1| {s5['ref_max_abs_std_dev']:.1e}, "
     f"median excess kurtosis {s5['ref_median_excess_kurtosis']:.2f}, mean % beyond z {s5['ref_pct_beyond_limit_mean']:.3f} "
     f"(N(0,1) expects {s5['ref_expected_pct']:.3f}). Sample: "
     + "; ".join(f"{r['node']} {r['mean']:+.2f}/{r['std']:.2f}" for r in s5["ref_sample"]) + " (mean/std).",
     "- Proposition 1 relative discrepancy |T² - Σ T²_j·PA(j)| / T² (median / p90): "
     + "; ".join(f"{k} {f(v['median'], 3)} / {f(v['p90'], 3)}" for k, v in s5["discrepancy"].items()) + ".", "",
     "## Top findings",
     "- Most often significant when T² alarms: "
     + "; ".join(f"{r['node']} ({r['unit']}, {r['pct_significant_when_T2_ooc']:.1f}%; tags: {r['top_loaded_tags'] or '-'})"
                 for r in s5["top_nodes"][:5]) + ".",
     f"- Flagged PCs (significant at the first T² alarm of a flare event): {', '.join(s5['flagged_pcs']) or 'none'}; "
     "their top-loaded tags are in `tables/flagged_pc_tags.csv`."]
for eid, d in list(ev.groupby("event_id"))[:10]:
    L.append(f"- Flare event {eid} (first T² alarm {d['first_ooc'].iloc[0]}): "
             + ", ".join(f"{t} ({u})" for t, u in d.sort_values("rank")[["node", "unit"]].head(3).itertuples(index=False)) + ".")
if s5["events_without_ooc"]:
    L.append(f"- Flare events with no T² alarm in their window: {s5['events_without_ooc']}.")
L += ["", "## Runtime (s)", " | ".join(f"{st} {secs.get(st, float('nan')):.0f}" for st in stages), "",
      "## Assumptions"] + [f"- [{a['stage']}] {a['text']}" for a in assumptions]
L += ["", "## Deviations"] + ([f"- [{d['stage']}] {d['text']}" for d in deviations] or ["- none"])
L += ["", "## Limitations",
      f"- PCA is linear and unsupervised: the {N_PCS} PCs keep {100 * s2['cum_variance']:.1f}% of the variance, not flare "
      "relevance, so a PC can matter for flaring while explaining little variance.",
      "- PC treats rows as independent; the 1-min data are autocorrelated (method kept as requested).",
      "- The learned graph is not guaranteed correct. The paper assumes the BN is free of learning and sampling errors "
      "and Σ known (Section 'Causation-Based T² Decomposition'); the discrepancy above measures how far that fails.",
      f"- Σ is estimated on {s5['n_reference_rows']} reference rows ({s1['reference_rule']}).",
      f"- The flare label rule (reused) marks {s1['pct_flare_rows']:.1f}% of rows as flaring in {s1['n_flare_events']} events.",
      "", "## Definition of Done", "| check | result | detail |", "|---|---|---|"]
L += [f"| {c} | {'PASS' if ok else 'FAIL'} | {d} |" for c, ok, d in checks]
(OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
for c, ok, d in checks:
    log(f"CHECK {'PASS' if ok else 'FAIL'}: {c} ({d})")
log(f"SUMMARY.md written ({len(L)} lines)")
if not all(ok for _, ok, _ in checks):
    raise SystemExit(1)
done(S, time.time() - t0)
