"""Stage 6 - SUMMARY.md and the Definition-of-Done checks (prompt Section 6).

Every number in SUMMARY.md comes from the s*_summary.json files, tag_registry.csv and figures_index.csv, so the
three documents agree by construction. The checks are written into SUMMARY.md; any FAIL exits with code 1.
"""
import re
import time
from datetime import date

import networkx as nx
import numpy as np
import pandas as pd

from common import (DATA, DATA_ROOT, FIG, GRAPH, OUT, PROJECT, done, load_json, log, read_registry, stage)

S = "s6_summary"
t0 = time.time()
stage("Stage 6 - SUMMARY.md and Definition-of-Done checks")
s1, s2, s3, s4 = (load_json(f"s{i}_summary.json") for i in ("1", "2", "3", "4"))
reg = read_registry()
idx = pd.read_csv(OUT / "figures_index.csv")
assumptions = load_json("assumptions.json") if (DATA / "assumptions.json").exists() else []
deviations = load_json("deviations.json") if (DATA / "deviations.json").exists() else []
ev = pd.read_csv(OUT / "tables" / "root_cause_by_event.csv", dtype={"tag": str})
stages = ["s1_load_clean", "s2_pc", "s3_dag", "s4_t2", "s5_figures"]
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
n_reg = int((reg["status"].str.startswith("in_matrix") | (reg["status"] == "used")
             | reg["status"].str.startswith("dropped_s2")).sum())
checks.append(("Combined matrix: no NaN/inf, no constant column; tag count matches tag_registry.csv",
               bool(np.isfinite(arr).all()) and bool((arr.std(axis=0) > 0).all()) and n_reg == Z.shape[1],
               f"{Z.shape[0]} rows x {Z.shape[1]} tags; registry rows in matrix: {n_reg}"))
del arr, Z

pe = pd.read_csv(GRAPH / "pc_edges.csv", dtype={"source": str, "target": str})
flare = set(reg.loc[reg["is_flare"], "tag"])
out_fl = int(((pe["type"] == "->") & pe["source"].isin(flare)).sum())
und_fl = int(((pe["type"] != "->") & (pe["source"].isin(flare) | pe["target"].isin(flare))).sum())
de = pd.read_csv(GRAPH / "dag_edges.csv", dtype={"source": str, "target": str})
g = nx.DiGraph(list(zip(de["source"], de["target"])))
dag_out_fl = int(de["source"].isin(flare).sum())
checks.append(("PC graph: 0 directed edges out of a flare tag, 0 undirected edges touching one; DAG acyclic",
               out_fl == 0 and und_fl == 0 and dag_out_fl == 0 and nx.is_directed_acyclic_graph(g),
               f"PC out-of-flare {out_fl}, undirected at flare {und_fl}; DAG out-of-flare {dag_out_fl}, "
               f"acyclic {nx.is_directed_acyclic_graph(g)}"))

t2_cols = pd.read_parquet(DATA / "t2.parquet", columns=["rel_discrepancy"]).columns
n_charts = int(idx["filename"].str.startswith("figures/per_tag/").sum())
checks.append(("Reference T_j·PA(j) about N(0,1); discrepancy computed; every tag has a chart",
               s4["ref_max_abs_mean"] < 0.05 and s4["ref_max_abs_std_dev"] < 0.05 and "rel_discrepancy" in t2_cols
               and n_charts == s4["p"] and len(list((FIG / "per_tag").glob("*.png"))) == s4["p"],
               f"max |mean| {s4['ref_max_abs_mean']:.1e}, max |std-1| {s4['ref_max_abs_std_dev']:.1e}; "
               f"charts {n_charts} for {s4['p']} tags"))

log_md = OUT / "LOG.md"
log_txt = log_md.read_text(encoding="utf-8") if log_md.exists() else ""
logged = [st for st in stages if re.search(rf"\| {st} \| PASS \|", log_txt)]
used = int((reg["status"] == "used").sum())
checks.append(("figures_index.csv, SUMMARY.md and LOG.md exist and agree",
               all((OUT / x).exists() for x in idx["filename"]) and used == s4["p"] and len(logged) == len(stages),
               f"index rows {len(idx)} (all files exist: {all((OUT / x).exists() for x in idx['filename'])}); "
               f"registry 'used' {used} = T² tags {s4['p']}; LOG.md PASS rows for {len(logged)}/{len(stages)} stages"))
checks.append(("One command reruns the whole pipeline (documented at the top)", (PROJECT / "run_all.py").exists(),
               "python causal_t2/run_all.py --force"))

# ---- SUMMARY.md
L = ["# SUMMARY: causal PC + causation-based T² on plant tag data", "",
     "Rerun everything: `python causal_t2/run_all.py --force` (paths in `causal_t2/.env`; without `--force` finished "
     f"stages are skipped). Generated {date.today().isoformat()}. Method: Li, Jin & Shi (2008), J. Quality Technology 40(1).", "",
     "## Tag counts per stage", "| stage | tags | flare tags | note |", "|---|---|---|---|",
     f"| loaded (unique) | {s1['n_tags_loaded']} | {s1['n_flare_tags_loaded']} | {s1['n_duplicate_copies']} duplicate copies skipped |",
     f"| S1 combined matrix | {s1['n_tags_matrix']} | {s1['n_flare_tags_matrix']} | dropped {s1['n_dropped_s1']}: "
     f"{s1['dropped_s1_by_reason']}; {s1['rows']} rows, {s1['interval']} grid |",
     f"| S2 PC input | {s2['n_tags_pc']} | {s2['n_flare_tags_pc']} | dropped {s2['n_dropped_s2']} (collinear / dependent) |",
     f"| S4 T² | {s4['p']} | {sum(1 for t in reg.loc[reg['status'] == 'used', 'tag'] if t in flare)} "
     f"| left out (zero reference variance): {s4['excluded'] or 'none'} |", "",
     "## Graph",
     f"- PC: alpha {s2['alpha']}, depth cap {s2['max_k']}, {s2['rows_pc']} rows (every {s2['row_step']}th), "
     f"{s2['n_ci_tests']} CI tests, {s2['seconds_pc']:.0f} s.",
     f"- Edges: {s2['n_edges']} = {s2['n_directed']} directed, {s2['n_undirected']} undirected, {s2['n_bidirected']} bidirected.",
     f"- Flare edges straight from PC: {s2['flare_edges_raw']}; changed after PC: {s2['flare_edges_changed']}.",
     f"- DAG: {s3['n_edges']} edges; CPDAG directed kept {s3['cpdag_directed_kept']}/{s3['cpdag_directed']}, "
     f"reversed {s3['n_reversed']}, forced Dor-Tarsi steps {s3['n_forced']}; roots {s3['n_roots']}, "
     f"parents mean {s3['mean_parents']:.2f} / max {s3['max_parents']}.", "",
     "## T² decomposition (Eqs. 7, 9-11, 13)",
     f"- Limits: T² > chi²_{s4['df']}(1 - {s4['alpha0']}) = {s4['t2_lim']:.2f}; |T_j·PA(j)| > z = {s4['z_lim']:.3f} "
     f"(Bonferroni alpha' = {s4['alpha0']}/{s4['p']}).",
     f"- T² out of control: all {f(s4['pct_ooc_all'])}%, reference {f(s4['pct_ooc_reference'])}%, flare rows "
     f"{f(s4['pct_ooc_flare'])}%, non-flare rows {f(s4['pct_ooc_nonflare'])}%.",
     f"- Reference check: max |mean| {s4['ref_max_abs_mean']:.1e}, max |std - 1| {s4['ref_max_abs_std_dev']:.1e}, "
     f"median excess kurtosis {s4['ref_median_excess_kurtosis']:.2f}, mean % beyond z {s4['ref_pct_beyond_limit_mean']:.3f} "
     f"(N(0,1) expects {s4['ref_expected_pct']:.3f}). Sample: "
     + "; ".join(f"{r['tag']} {r['mean']:+.2f}/{r['std']:.2f}" for r in s4["ref_sample"]) + " (mean/std).",
     "- Proposition 1 relative discrepancy |T² - Σ T²_j·PA(j)| / T² (median / p90): "
     + "; ".join(f"{k} {f(v['median'], 3)} / {f(v['p90'], 3)}" for k, v in s4["discrepancy"].items()) + ".", "",
     "## Top findings",
     "- Most often significant when T² alarms: "
     + "; ".join(f"{r['tag']} ({r['unit']}, {r['pct_significant_when_T2_ooc']:.1f}%)" for r in s4["top_tags"][:5]) + "."]
for eid, d in list(ev.groupby("event_id"))[:10]:
    L.append(f"- Flare event {eid} (first T² alarm {d['first_ooc'].iloc[0]}): "
             + ", ".join(f"{t} ({u})" for t, u in d.sort_values("rank")[["tag", "unit"]].head(3).itertuples(index=False)) + ".")
if s4["events_without_ooc"]:
    L.append(f"- Flare events with no T² alarm in their window: {s4['events_without_ooc']}.")
L += ["", "## Runtime (s)", " | ".join(f"{st} {secs.get(st, float('nan')):.0f}" for st in stages), "",
      "## Assumptions"] + [f"- [{a['stage']}] {a['text']}" for a in assumptions]
L += ["", "## Deviations"] + ([f"- [{d['stage']}] {d['text']}" for d in deviations] or ["- none"])
L += ["", "## Limitations",
      "- PC treats rows as independent; the 1-min data are autocorrelated (method kept as requested).",
      "- The learned graph is not guaranteed correct. The paper assumes the BN is free of learning and sampling errors "
      "and Σ known (Section 'Causation-Based T² Decomposition'); the discrepancy above measures how far that fails.",
      f"- Σ is estimated on {s4['n_reference_rows']} reference rows ({s1['reference_rule']}).",
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
