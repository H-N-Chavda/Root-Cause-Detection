"""Look for cycles in the PC output and in the final graph: python unit_pca_pc_t2/check_cycles.py.

Same check as the TA's Main.py (lines 182-198): adjacency matrix -> networkx DiGraph -> nx.simple_cycles.
Writes graph/adjacency_final.csv and graph/cycles.csv.
"""
import networkx as nx
import pandas as pd

import config as C
import helpers as H


def cycles_of(edges: list[tuple[str, str]], nodes: list[str]) -> tuple[pd.DataFrame, list[list[str]]]:
    """Adjacency matrix and all simple cycles of a directed edge list.

    Args:
        edges: (source, target) pairs.
        nodes: Node names (matrix order).

    Returns:
        A tuple (adjacency matrix, cycles).
    """
    A = pd.DataFrame(0, index=nodes, columns=nodes)
    for s, t in edges:
        A.at[s, t] = 1
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from((s, t) for s in nodes for t in nodes if A.at[s, t])
    return A, [list(c) for c in nx.simple_cycles(G)]


def main() -> None:
    """Report cycles in the PC output (directed edges only) and in the final graph used for T²."""
    P = C.load_paths()
    d = H.make_output_dirs(P.out)
    H.start_log(d["logs"] / "check_cycles.log")
    cp = pd.read_csv(d["graph"] / "pc_edges.csv")
    dag = pd.read_csv(d["graph"] / "dag_edges.csv")
    nodes = sorted(set(dag["source"]) | set(dag["target"]))
    rows = []
    for name, edges in (("PC output (directed edges)",
                         [(s, t) for s, t, ty in cp[["source", "target", "type"]].itertuples(index=False)
                          if ty == H.EDGE_DIRECTED]),
                        ("final graph", list(zip(dag["source"], dag["target"])))):
        A, cyc = cycles_of(edges, nodes)
        H.log(f"{name}: {len(edges)} edges, {len(cyc)} cycles" + ("" if cyc else " (acyclic)"))
        for c in cyc:
            H.log("  " + " -> ".join(c + [c[0]]))
            rows.append({"graph": name, "length": len(c), "cycle": " -> ".join(c + [c[0]])})
        if name == "final graph":
            A.to_csv(d["graph"] / "adjacency_final.csv")
    pd.DataFrame(rows, columns=["graph", "length", "cycle"]).to_csv(d["graph"] / "cycles.csv", index=False)


if __name__ == "__main__":
    main()
