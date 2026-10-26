"""Stage 2 - PC algorithm (causal-learn, Fisher-Z, stable) on the standardized combined matrix.

Flare constraint: a flare tag can have no outgoing edge. Enforced inside PC with causal-learn BackgroundKnowledge
tiers: flare tags in tier 1, all other tags in tier 0. An edge from a higher to a lower tier is forbidden
(flare -> X), and edges within tier 1 are forbidden too (flare -> flare would be an outgoing flare edge).
The output is then checked; a flare edge left undirected or bidirected is oriented X -> flare.

Before PC, collinear pairs (|r| >= COLLINEAR_R) and any remaining near-linear dependence are removed, because the
Fisher-Z test inverts correlation submatrices. The depth cap max_k is the largest value <= PC_MAX_K whose
CI-test upper bound (computed from the marginal-correlation skeleton) fits PC_TEST_BUDGET.

Writes outputs/graph/{pc_edges.csv, pc_cpdag_adjacency.csv, pc_graph.graphml}, outputs/data/{pc_graph.npz,
s2_summary.json}, outputs/tables/s2_dropped_tags.csv and updates tag_registry.csv.
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
from causallearn.graph.GraphClass import CausalGraph
from scipy.stats import norm

from common import (COLLINEAR_R, DATA, EDGE_BIDIRECTED, EDGE_DIRECTED, EDGE_UNDIRECTED, EIG_MIN, GRAPH, PC_ALPHA,
                    PC_MAX_K, PC_ROW_STEP, PC_TEST_BUDGET, TAB, assumption, clear_notes, done, edge_list, gate, log,
                    read_registry, save_json, stage, write_registry)

S = "s2_pc"


def orient(G, a, b):
    """Set the edge between a and b to a -> b (causal-learn encoding)."""
    G[a, b], G[b, a] = -1, 1


def flare_edge_counts(e, flare_of):
    src_f, tgt_f = e["source"].map(flare_of), e["target"].map(flare_of)
    return {"directed_out_of_flare": int(((e["type"] == EDGE_DIRECTED) & src_f).sum()),
            "undirected_or_bidirected_touching_flare": int((e["type"].isin([EDGE_UNDIRECTED, EDGE_BIDIRECTED])
                                                            & (src_f | tgt_f)).sum()),
            "flare_flare": int((src_f & tgt_f).sum())}


t0 = time.time()
stage("Stage 2 - PC algorithm")
clear_notes(S)
gate("max_k" in inspect.signature(pc).parameters,
     "installed causal-learn supports the depth cap max_k (if not: pip install -U causal-learn)")

Z = pd.read_parquet(DATA / "Z.parquet")
reg = read_registry()
flare_of = dict(zip(reg["tag"], reg["is_flare"]))
unit_of = dict(zip(reg["tag"], reg["unit"]))
names = list(Z.columns)
D = Z.iloc[::PC_ROW_STEP].to_numpy(dtype=np.float64)
del Z
assumption(S, f"PC runs on every {PC_ROW_STEP}th row of the combined matrix ({len(D)} rows): consecutive samples are "
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
    log(f"correlation matrix of {len(k)} tags: smallest eigenvalue {ev[0]:.3e}, condition number {ev[-1] / max(ev[0], 1e-300):.3e}")
    if ev[0] >= EIG_MIN:
        break
    load = np.abs(V[:, 0])
    load[fl[k]] = -1.0                       # flare tags are kept
    v = k[int(np.argmax(load))]
    keep[v] = False
    drops.append({"tag": names[v], "reason": f"near-linear dependence (smallest eigenvalue {ev[0]:.2e} < {EIG_MIN:g}), "
                  "largest loading on that eigenvector", "partner": "", "value": float(ev[0])})
    log(f"  drop {names[v]}: largest loading on the eigenvector of eigenvalue {ev[0]:.2e}")
pd.DataFrame(drops, columns=["tag", "reason", "partner", "value"]).to_csv(TAB / "s2_dropped_tags.csv", index=False)
k = np.flatnonzero(keep)
names_k = [names[i] for i in k]
fl_k = fl[k]
Dk = np.ascontiguousarray(D[:, k])
Rk = R[np.ix_(k, k)]
del D
gate(fl_k.any(), f"flare tags still present for PC: {int(fl_k.sum())}")
log(f"PC input: {Dk.shape[0]} rows x {Dk.shape[1]} tags ({len(drops)} dropped in this stage)")

# ---- depth cap from an upper bound on the number of CI tests
n = len(Dk)
zc = norm.ppf(1 - PC_ALPHA / 2)
zs = np.arctanh(np.clip(np.abs(Rk), 0, 1 - 1e-12)) * np.sqrt(n - 3)
A0 = zs > zc
np.fill_diagonal(A0, False)
A0[np.ix_(fl_k, fl_k)] = False                  # flare-flare edges are removed by the background knowledge
deg = A0.sum(axis=1)
bound = [int(sum(int(d) * comb(max(int(d) - 1, 0), m) for d in deg)) for m in range(PC_MAX_K + 1)]
cum = np.cumsum(bound)
fits = [m for m in range(1, PC_MAX_K + 1) if cum[m] <= PC_TEST_BUDGET]
max_k = max(fits) if fits else 1
log(f"marginal skeleton: {int(A0.sum() // 2)} edges, max degree {int(deg.max())}; CI-test upper bound by depth "
    f"{dict(enumerate(bound))}")
assumption(S, f"PC depth cap max_k = {max_k}: largest depth <= {PC_MAX_K} whose CI-test upper bound "
              f"({int(cum[max_k]):.3g}) fits the budget {PC_TEST_BUDGET:.3g}"
              + ("" if fits else "; even depth 1 exceeds the budget (the bound is loose), depth 1 kept"))

# ---- PC with the flare constraint as background knowledge
node_names = [f"X{i + 1}" for i in range(len(names_k))]       # causal-learn's default node names
bk = BackgroundKnowledge()
for i, nm in enumerate(node_names):
    bk.add_node_to_tier(GraphNode(nm), 1 if fl_k[i] else 0)
bk.forbid_within_tier(1)
assumption(S, f"PC: causal-learn pc, Fisher-Z, alpha = {PC_ALPHA}, stable = True, uc_rule = 0, uc_priority = 2; "
              "flare tags in tier 1 (no edge into tier 0, none within tier 1)")
t_pc = time.time()

def _find_adj(self):
    g = self.G.graph
    t, a = np.where(g == -1), np.where(g == 1)
    return list(zip(t[1], t[0])) + list(zip(a[1], a[0]))

CausalGraph.find_adj = _find_adj

cg = pc(Dk, alpha=PC_ALPHA, indep_test="fisherz", stable=True, uc_rule=0, uc_priority=2, background_knowledge=bk,
        verbose=False, show_progress=False, max_k=max_k)
secs_pc = time.time() - t_pc
cache = getattr(cg.test, "pvalue_cache", None)
n_tests = len(cache) if cache is not None else -1
G = np.array(cg.G.graph, dtype=int)
e_raw = edge_list(G, names_k)
raw = flare_edge_counts(e_raw, flare_of)
log(f"PC done in {secs_pc:.1f}s, {n_tests} distinct CI tests, {len(e_raw)} edges; flare edges straight from PC: {raw}")

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
log(f"flare edges changed after PC: {fixed}; flare edges now: {after}")
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
reg["status"] = [f"dropped_s2: {dropped[t]}" if t in dropped else s for t, s in zip(reg["tag"], reg["status"])]
write_registry(reg)
types = e["type"].value_counts().to_dict()
seconds = time.time() - t0
save_json({"rows_pc": n, "row_step": PC_ROW_STEP, "n_tags_pc": len(names_k), "n_flare_tags_pc": int(fl_k.sum()),
           "n_dropped_s2": len(drops), "alpha": PC_ALPHA, "max_k": max_k, "ci_test_bound_by_depth": bound,
           "test_budget": PC_TEST_BUDGET, "marginal_edges": int(A0.sum() // 2), "n_ci_tests": n_tests,
           "n_edges": len(e), "n_directed": int(types.get(EDGE_DIRECTED, 0)),
           "n_undirected": int(types.get(EDGE_UNDIRECTED, 0)), "n_bidirected": int(types.get(EDGE_BIDIRECTED, 0)),
           "flare_edges_raw": raw, "flare_edges_changed": int(fixed), "flare_edges_after": after,
           "seconds_pc": secs_pc, "seconds": seconds}, "s2_summary.json")
done(S, seconds)
