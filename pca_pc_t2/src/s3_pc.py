"""Stage 3 - PC algorithm (causal-learn, Fisher-Z, stable) on the node matrix [26 PC scores + flare tags].

Copied from causal_t2/src/s2_pc.py and edited in three places only: (1) the input is the node matrix (standardized
PC scores from Stage 2 + the flare tags as standardized in Stage 1) instead of every tag; (2) PC runs once per alpha
in PC_ALPHAS and the edge counts are reported, PC_ALPHA is used downstream; (3) edges are also counted by kind
(PC-PC, PC -> flare). Unchanged from the copy:

Flare constraint: a flare tag can have no outgoing edge. Enforced inside PC with causal-learn BackgroundKnowledge
tiers: flare tags in tier 1, all other nodes in tier 0. An edge from a higher to a lower tier is forbidden
(flare -> X), and edges within tier 1 are forbidden too (flare -> flare would be an outgoing flare edge).
The output is then checked; a flare edge left undirected or bidirected is oriented X -> flare.

Before PC, collinear pairs (|r| >= COLLINEAR_R) and any remaining near-linear dependence are removed, because the
Fisher-Z test inverts correlation submatrices. The depth cap max_k is the largest value <= PC_MAX_K whose
CI-test upper bound (computed from the marginal-correlation skeleton) fits PC_TEST_BUDGET.

Writes outputs/graph/{pc_edges.csv, pc_edges_alpha*.csv, pc_cpdag_adjacency.csv, pc_graph.graphml},
outputs/data/{node_matrix.parquet, nodes.csv, pc_graph.npz, s3_summary.json}, outputs/tables/s3_dropped_nodes.csv
and updates tag_registry.csv (flare tags dropped here).
"""
import inspect
import time
from math import comb

import networkx as nx
import numpy as np
import pandas as pd
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from scipy.stats import norm

from common import (COLLINEAR_R, DATA, EDGE_BIDIRECTED, EDGE_DIRECTED, EDGE_UNDIRECTED, EIG_MIN, FLARE_FOLDER, GRAPH,
                    N_PCS, NODES, PC_ALPHA, PC_ALPHAS, PC_MAX_K, PC_ROW_STEP, PC_TEST_BUDGET, TAB, assumption,
                    clear_notes, done, edge_list, gate, log, read_registry, save_json, stage, write_registry)

S = "s3_pc"


def orient(G, a, b):
    """Set the edge between a and b to a -> b (causal-learn encoding)."""
    G[a, b], G[b, a] = -1, 1


def flare_edge_counts(e, flare_of):
    src_f, tgt_f = e["source"].map(flare_of), e["target"].map(flare_of)
    return {"directed_out_of_flare": int(((e["type"] == EDGE_DIRECTED) & src_f).sum()),
            "undirected_or_bidirected_touching_flare": int((e["type"].isin([EDGE_UNDIRECTED, EDGE_BIDIRECTED])
                                                            & (src_f | tgt_f)).sum()),
            "flare_flare": int((src_f & tgt_f).sum())}


def kind_counts(e, flare_of):
    """New: edges between two PCs and edges between a PC and a flare tag."""
    src_f, tgt_f = e["source"].map(flare_of), e["target"].map(flare_of)
    return {"pc_pc": int((~src_f & ~tgt_f).sum()), "pc_flare": int((src_f ^ tgt_f).sum()),
            "flare_flare": int((src_f & tgt_f).sum())}


t0 = time.time()
stage("Stage 3 - PC algorithm on 26 PCs + flare tags")
clear_notes(S)
gate("max_k" in inspect.signature(pc).parameters,
     "installed causal-learn supports the depth cap max_k (if not: pip install -U causal-learn)")

# ---- node matrix: standardized PC scores + flare tags (new in this pipeline)
reg = read_registry()
flare_tags = list(reg.loc[reg["is_flare"] & reg["status"].str.startswith("in_matrix"), "tag"])
P = pd.read_parquet(DATA / "pc_scores.parquet")
F = pd.read_parquet(DATA / "X_flare.parquet", columns=flare_tags)
gate(P.index.equals(F.index), "PC scores and flare tags share the same timestamps")
N = pd.concat([P, F], axis=1)
del P, F
gate(N.shape[1] == N_PCS + len(flare_tags), f"node matrix: {N_PCS} PCs + {len(flare_tags)} flare tags = {N.shape[1]} nodes")
N.to_parquet(DATA / "node_matrix.parquet")
share = pd.read_csv(TAB / "pca_unit_share.csv", index_col=0)
nodes = pd.DataFrame({"node": list(N.columns), "is_flare": [False] * N_PCS + [True] * len(flare_tags),
                      "kind": ["PC"] * N_PCS + ["flare"] * len(flare_tags),
                      "unit": [f"PC ({share.loc[p].idxmax()})" for p in N.columns[:N_PCS]] + [FLARE_FOLDER] * len(flare_tags)})
assumption(S, "flare tags enter PC as standardized in Stage 1 (reference mean / std); PC scores as standardized in Stage 2")
flare_of = dict(zip(nodes["node"], nodes["is_flare"]))
unit_of = dict(zip(nodes["node"], nodes["unit"]))
names = list(N.columns)
D = N.iloc[::PC_ROW_STEP].to_numpy(dtype=np.float64)
del N
assumption(S, f"PC runs on every {PC_ROW_STEP}th row of the node matrix ({len(D)} rows): consecutive samples are "
              "strongly autocorrelated, and Fisher-Z uses the data only through the correlation matrix and n")
p = len(names)
fl = np.array([flare_of[t] for t in names])
keep = np.ones(p, dtype=bool)
drops = []

# ---- constant in the subsample, collinear pairs, near-linear dependence
sd = D.std(axis=0)
for i in np.flatnonzero(sd == 0):
    keep[i] = False
    drops.append({"tag": names[i], "reason": "constant in the PC row subsample", "partner": "", "value": 0.0})
sd[sd == 0] = 1.0
D = (D - D.mean(axis=0)) / sd
R = np.corrcoef(D, rowvar=False)
R[~keep, :] = 0.0
R[:, ~keep] = 0.0
np.fill_diagonal(R, 1.0)
iu = np.triu_indices(p, 1)
r = np.abs(R[iu])
hits = np.flatnonzero(r >= COLLINEAR_R)
for h in hits[np.argsort(-r[hits])]:
    i, j = iu[0][h], iu[1][h]
    if not (keep[i] and keep[j]):
        continue
    victim, partner = (i, j) if (fl[j] and not fl[i]) else (j, i)      # never drop a flare tag if avoidable
    keep[victim] = False
    drops.append({"tag": names[victim], "reason": f"collinear (|r| >= {COLLINEAR_R})", "partner": names[partner],
                  "value": float(r[h])})
    log(f"  drop {names[victim]}: |r| = {r[h]:.5f} with {names[partner]}")
while True:
    k = np.flatnonzero(keep)
    ev, V = np.linalg.eigh(R[np.ix_(k, k)])
    log(f"correlation matrix of {len(k)} nodes: smallest eigenvalue {ev[0]:.3e}, condition number {ev[-1] / max(ev[0], 1e-300):.3e}")
    if ev[0] >= EIG_MIN:
        break
    load = np.abs(V[:, 0])
    load[fl[k]] = -1.0                       # flare tags are kept
    v = k[int(np.argmax(load))]
    keep[v] = False
    drops.append({"tag": names[v], "reason": f"near-linear dependence (smallest eigenvalue {ev[0]:.2e} < {EIG_MIN:g}), "
                  "largest loading on that eigenvector", "partner": "", "value": float(ev[0])})
    log(f"  drop {names[v]}: largest loading on the eigenvector of eigenvalue {ev[0]:.2e}")
pd.DataFrame(drops, columns=["tag", "reason", "partner", "value"]).to_csv(TAB / "s3_dropped_nodes.csv", index=False)
k = np.flatnonzero(keep)
names_k = [names[i] for i in k]
fl_k = fl[k]
Dk = np.ascontiguousarray(D[:, k])
Rk = R[np.ix_(k, k)]
del D
gate(fl_k.any(), f"flare tags still present for PC: {int(fl_k.sum())}")
log(f"PC input: {Dk.shape[0]} rows x {Dk.shape[1]} nodes ({int((~fl_k).sum())} PCs, {int(fl_k.sum())} flare tags; "
    f"{len(drops)} dropped in this stage)")
nodes["in_pc"] = nodes["node"].isin(names_k)
nodes.to_csv(NODES, index=False)

# ---- depth cap from an upper bound on the number of CI tests (largest alpha of the sweep = loosest bound)
n = len(Dk)
zc = norm.ppf(1 - max(PC_ALPHAS) / 2)
zs = np.arctanh(np.clip(np.abs(Rk), 0, 1 - 1e-12)) * np.sqrt(n - 3)
A0 = zs > zc
np.fill_diagonal(A0, False)
A0[np.ix_(fl_k, fl_k)] = False                  # flare-flare edges are removed by the background knowledge
deg = A0.sum(axis=1)
bound = [int(sum(int(d) * comb(max(int(d) - 1, 0), m) for d in deg)) for m in range(PC_MAX_K + 1)]
cum = np.cumsum(bound)
fits = [m for m in range(1, PC_MAX_K + 1) if cum[m] <= PC_TEST_BUDGET]
max_k = max(fits) if fits else 1
log(f"marginal skeleton (alpha {max(PC_ALPHAS)}): {int(A0.sum() // 2)} edges, max degree {int(deg.max())}; "
    f"CI-test upper bound by depth {dict(enumerate(bound))}")
assumption(S, f"PC depth cap max_k = {max_k}: largest depth <= {PC_MAX_K} whose CI-test upper bound "
              f"({int(cum[max_k]):.3g}) fits the budget {PC_TEST_BUDGET:.3g}"
              + ("" if fits else "; even depth 1 exceeds the budget (the bound is loose), depth 1 kept"))

# ---- PC with the flare constraint as background knowledge, once per alpha (sweep is new in this pipeline)
node_names = [f"X{i + 1}" for i in range(len(names_k))]       # causal-learn's default node names
bk = BackgroundKnowledge()
for i, nm in enumerate(node_names):
    bk.add_node_to_tier(GraphNode(nm), 1 if fl_k[i] else 0)
bk.forbid_within_tier(1)
assumption(S, f"PC: causal-learn pc, Fisher-Z, stable = True, uc_rule = 0, uc_priority = 2, alphas {list(PC_ALPHAS)}; "
              "flare tags in tier 1 (no edge into tier 0, none within tier 1, so flare-flare edges are forbidden too)")
sweep, G, secs_pc, n_tests = [], None, 0.0, -1
for alpha in PC_ALPHAS:
    t_pc = time.time()
    cg = pc(Dk, alpha=alpha, indep_test="fisherz", stable=True, uc_rule=0, uc_priority=2, background_knowledge=bk,
            verbose=False, show_progress=False, max_k=max_k)
    secs = time.time() - t_pc
    cache = getattr(cg.test, "pvalue_cache", None)
    tests = len(cache) if cache is not None else -1
    Ga = np.array(cg.G.graph, dtype=int)
    ea = edge_list(Ga, names_k)
    ea.to_csv(GRAPH / f"pc_edges_alpha{alpha:g}.csv", index=False)
    types_a = ea["type"].value_counts().to_dict()
    sweep.append({"alpha": alpha, "n_edges": len(ea), "n_directed": int(types_a.get(EDGE_DIRECTED, 0)),
                  "n_undirected": int(types_a.get(EDGE_UNDIRECTED, 0)), "n_bidirected": int(types_a.get(EDGE_BIDIRECTED, 0)),
                  **kind_counts(ea, flare_of), "n_ci_tests": tests, "seconds": round(secs, 1)})
    log(f"PC alpha {alpha}: {len(ea)} edges ({kind_counts(ea, flare_of)}), {tests} CI tests, {secs:.1f}s")
    if alpha == PC_ALPHA:
        G, secs_pc, n_tests = Ga, secs, tests
pd.DataFrame(sweep).to_csv(TAB / "pc_alpha_sweep.csv", index=False)
gate(G is not None, f"the downstream alpha {PC_ALPHA} is part of the sweep {list(PC_ALPHAS)}")
assumption(S, f"downstream alpha = {PC_ALPHA}: the stricter level of the sweep, as in causal_t2; with thousands of rows "
              "Fisher-Z rejects independence for very small correlations, so the stricter level keeps the graph sparser")
e_raw = edge_list(G, names_k)
raw = flare_edge_counts(e_raw, flare_of)
log(f"alpha {PC_ALPHA} used downstream: {len(e_raw)} edges; flare edges straight from PC: {raw}")

# ---- verify the flare constraint; orient leftovers X -> flare, remove any flare-flare edge
pos = {t: i for i, t in enumerate(names_k)}
fixed = 0
for s, t, ty in e_raw[["source", "target", "type"]].itertuples(index=False):
    a, b = pos[s], pos[t]
    if fl_k[a] and fl_k[b]:
        G[a, b] = G[b, a] = 0
        fixed += 1
        log(f"  removed flare-flare edge {s} {ty} {t}")
    elif fl_k[a]:                           # flare -> X, flare -- X or flare <-> X: becomes X -> flare
        orient(G, b, a)
        fixed += 1
    elif fl_k[b] and ty != EDGE_DIRECTED:   # X -- flare or X <-> flare: becomes X -> flare
        orient(G, a, b)
        fixed += 1
e = edge_list(G, names_k)
after = flare_edge_counts(e, flare_of)
kinds = kind_counts(e, flare_of)
log(f"flare edges changed after PC: {fixed}; flare edges now: {after}; edges by kind: {kinds}")
gate(after["directed_out_of_flare"] == 0, "0 directed edges out of any flare tag")
gate(after["undirected_or_bidirected_touching_flare"] == 0, "0 undirected / bidirected edges touching a flare tag")
gate(after["flare_flare"] == 0, "0 flare-flare edges")

# ---- save adjacency, edge list, GraphML
e = e.assign(source_unit=e["source"].map(unit_of), target_unit=e["target"].map(unit_of),
             source_is_flare=e["source"].map(flare_of), target_is_flare=e["target"].map(flare_of))
e.to_csv(GRAPH / "pc_edges.csv", index=False)
A = (G != 0) & ~((G == 1) & (G.T == -1))       # A[i, j] = 1: i -> j, or undirected / bidirected i - j
pd.DataFrame(A.astype(int), index=names_k, columns=names_k).to_csv(GRAPH / "pc_cpdag_adjacency.csv")
g = nx.DiGraph()
for t in names_k:
    g.add_node(t, unit=unit_of[t], is_flare=bool(flare_of[t]))
kind = {EDGE_DIRECTED: "directed", EDGE_UNDIRECTED: "undirected", EDGE_BIDIRECTED: "bidirected"}
for s, t, ty in e[["source", "target", "type"]].itertuples(index=False):
    g.add_edge(s, t, type=kind.get(ty, ty))
nx.write_graphml(g, GRAPH / "pc_graph.graphml")
np.savez(DATA / "pc_graph.npz", G=G, names=np.array(names_k, dtype=object))

dropped = {d["tag"]: d["reason"] for d in drops}
reg["status"] = [f"dropped_s3: {dropped[t]}" if t in dropped else s for t, s in zip(reg["tag"], reg["status"])]
write_registry(reg)
types = e["type"].value_counts().to_dict()
seconds = time.time() - t0
save_json({"rows_pc": n, "row_step": PC_ROW_STEP, "n_nodes_pc": len(names_k), "n_pcs_pc": int((~fl_k).sum()),
           "n_flare_tags_pc": int(fl_k.sum()), "n_dropped_s3": len(drops), "dropped_s3": drops,
           "alpha": PC_ALPHA, "alpha_sweep": sweep, "max_k": max_k, "ci_test_bound_by_depth": bound,
           "test_budget": PC_TEST_BUDGET, "marginal_edges": int(A0.sum() // 2), "n_ci_tests": n_tests,
           "n_edges": len(e), "n_directed": int(types.get(EDGE_DIRECTED, 0)),
           "n_undirected": int(types.get(EDGE_UNDIRECTED, 0)), "n_bidirected": int(types.get(EDGE_BIDIRECTED, 0)),
           "edges_by_kind": kinds, "flare_edges_raw": raw, "flare_edges_changed": int(fixed), "flare_edges_after": after,
           "seconds_pc": secs_pc, "seconds": seconds}, "s3_summary.json")
done(S, seconds)
