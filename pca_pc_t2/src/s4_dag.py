"""Stage 4 - CPDAG -> DAG (consistent extension, Dor & Tarsi 1992), keeping the flare constraint.

Copied from causal_t2/src/s3_dag.py; edited only to read the node table (26 PCs + flare tags) instead of the
tag registry, and the stage name.

Dor-Tarsi: repeatedly pick a node x with no outgoing directed edge whose undirected neighbours are each adjacent
to every other node adjacent to x; orient all undirected edges at x into x and remove x. Flare tags have no
outgoing edge (Stage 3), so they are removed first and every edge at a flare becomes X -> flare.
PC output need not have a consistent extension. Then, logged and counted:
  - no node passes the adjacency condition -> the sink with the fewest violations is taken ("forced");
  - no sink at all (directed cycle) -> the node with the fewest outgoing edges is taken, those edges reversed.
Bidirected (conflict) edges are treated as undirected.

Writes outputs/graph/{dag_edges.csv, dag_adjacency.csv, dag.graphml}, outputs/data/{parents.json, s4_summary.json}.
"""
import time

import networkx as nx
import numpy as np
import pandas as pd

from common import (DATA, EDGE_BIDIRECTED, EDGE_DIRECTED, GRAPH, assumption, clear_notes, deviation, done, gate, log, read_nodes,
                    save_json, stage)

S = "s4_dag"
t0 = time.time()
stage("Stage 4 - CPDAG to DAG")
clear_notes(S)
nodes = read_nodes()
flare_of = dict(zip(nodes["node"], nodes["is_flare"]))
unit_of = dict(zip(nodes["node"], nodes["unit"]))
e = pd.read_csv(GRAPH / "pc_edges.csv", dtype={"source": str, "target": str})
names = [str(t) for t in np.load(DATA / "pc_graph.npz", allow_pickle=True)["names"]]

out = {v: set() for v in names}       # remaining graph: directed out / in, undirected neighbours
inn = {v: set() for v in names}
und = {v: set() for v in names}
for s, t, ty in e[["source", "target", "type"]].itertuples(index=False):
    if ty == EDGE_DIRECTED:
        out[s].add(t)
        inn[t].add(s)
    else:
        und[s].add(t)
        und[t].add(s)
cpdag_type = {}
for s, t, ty in e[["source", "target", "type"]].itertuples(index=False):
    cpdag_type[(s, t)] = cpdag_type[(t, s)] = ty
cpdag_dir = {(s, t) for s, t, ty in e[["source", "target", "type"]].itertuples(index=False) if ty == EDGE_DIRECTED}

dag, how = set(), {}
remaining = set(names)
n_forced, reversed_edges = 0, []


def adj(x):
    return out[x] | inn[x] | und[x]


def remove(x, label):
    """Orient every remaining edge at x into x, then delete x from the remaining graph."""
    for y in list(und[x]):
        dag.add((y, x))
        how[(y, x)] = label
        und[y].discard(x)
    for y in list(inn[x]):
        dag.add((y, x))
        how[(y, x)] = "directed in CPDAG, kept"
        out[y].discard(x)
    for y in list(out[x]):                    # only in the directed-cycle fallback
        dag.add((y, x))
        how[(y, x)] = "reversed (directed cycle in CPDAG)"
        reversed_edges.append((x, y))
        inn[y].discard(x)
    out[x], inn[x], und[x] = set(), set(), set()
    remaining.discard(x)


def violations(x):
    """Number of undirected neighbours y of x that miss some other neighbour of x (0 = Dor-Tarsi condition holds)."""
    ax = adj(x)
    return sum(1 for y in und[x] if not (ax - {y}) <= (adj(y) | {y}))


flares = [v for v in names if flare_of[v]]
gate(all(not out[f] for f in flares), "no flare tag has an outgoing edge in the CPDAG")
for f in flares:
    remove(f, "flare: oriented X -> flare")
log(f"{len(flares)} flare tags removed first as sinks")

order_pos = {v: i for i, v in enumerate(names)}
while remaining:
    sinks = sorted((v for v in remaining if not out[v]), key=order_pos.get)
    if sinks:
        best = None
        for v in sinks:
            if violations(v) == 0:
                best = v
                break
        if best is None:
            best = min(sinks, key=lambda v: (violations(v), order_pos[v]))
            n_forced += 1
            remove(best, "undirected, oriented (forced: no consistent extension)")
        else:
            remove(best, "undirected, oriented (Dor-Tarsi)")
    else:
        best = min(remaining, key=lambda v: (len(out[v]), order_pos[v]))
        remove(best, "undirected, oriented")

for st in dag:
    if cpdag_type[st] == EDGE_BIDIRECTED:
        how[st] = how[st].replace("undirected", "bidirected")
if n_forced:
    deviation(S, f"the CPDAG has no consistent extension: {n_forced} Dor-Tarsi steps took the sink with the fewest "
                 "adjacency violations")
if reversed_edges:
    deviation(S, f"{len(reversed_edges)} CPDAG directed edges reversed to break directed cycles: "
                 + "; ".join(f"{a}->{b}" for a, b in reversed_edges[:20]) + (" ..." if len(reversed_edges) > 20 else ""))
assumption(S, "DAG = Dor-Tarsi consistent extension of the PC CPDAG; flare tags removed first (all edges into flares); "
              "bidirected edges treated as undirected; ties broken by the matrix column order")

# ---- checks
g = nx.DiGraph()
g.add_nodes_from(names)
g.add_edges_from(dag)
skel_cpdag = {frozenset(k) for k in cpdag_type}
skel_dag = {frozenset(k) for k in dag}
gate(nx.is_directed_acyclic_graph(g), "DAG is acyclic")
gate(all(g.out_degree(f) == 0 for f in flares), "0 outgoing edges from any flare tag in the DAG")
gate(skel_cpdag == skel_dag, f"DAG keeps exactly the CPDAG skeleton ({len(skel_dag)} edges)")
kept_dir = len(cpdag_dir & dag)
log(f"DAG: {len(dag)} edges; CPDAG directed kept {kept_dir} of {len(cpdag_dir)}, reversed {len(reversed_edges)}, "
    f"forced steps {n_forced}")

# ---- save
rows = sorted(dag, key=lambda st: (order_pos[st[1]], order_pos[st[0]]))
d = pd.DataFrame(rows, columns=["source", "target"])
d["how"] = [how[st] for st in rows]
d["source_unit"], d["target_unit"] = d["source"].map(unit_of), d["target"].map(unit_of)
d.to_csv(GRAPH / "dag_edges.csv", index=False)
A = np.zeros((len(names), len(names)), dtype=int)
for s, t in dag:
    A[order_pos[s], order_pos[t]] = 1
pd.DataFrame(A, index=names, columns=names).to_csv(GRAPH / "dag_adjacency.csv")
for v in names:
    g.nodes[v]["unit"], g.nodes[v]["is_flare"] = unit_of[v], bool(flare_of[v])
nx.write_graphml(g, GRAPH / "dag.graphml")
parents = {v: sorted(g.predecessors(v), key=order_pos.get) for v in names}
save_json(parents, "parents.json")
n_par = pd.Series({v: len(p) for v, p in parents.items()})
seconds = time.time() - t0
save_json({"n_nodes": len(names), "n_edges": len(dag), "cpdag_directed": len(cpdag_dir), "cpdag_directed_kept": kept_dir,
           "n_reversed": len(reversed_edges), "n_forced": n_forced,
           "how_counts": d["how"].value_counts().to_dict(), "n_roots": int((n_par == 0).sum()),
           "max_parents": int(n_par.max()), "mean_parents": float(n_par.mean()), "seconds": seconds}, "s4_summary.json")
done(S, seconds)
