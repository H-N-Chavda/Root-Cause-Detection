"""Compare the CPDAGs from standard_pc.py and parallel_pc.py, and explain any difference.

Reports: identical adjacencies? identical orientations? number of differing node pairs;
skeleton errors of each vs the true DAG for context; a side-by-side figure.
Controls (to attribute differences):
  A. Parallel PC with 1 vs 4 workers                -> does parallelism itself change anything?
  B. Both algorithms under 10 column permutations    -> order dependence of skeleton and CPDAG.
"""
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import parallel_pc  # noqa: E402
import standard_pc  # noqa: E402

OUT = Path(__file__).parent / "outputs"


def edge_type(G, i, j):
    """'->', '<-', '--' or None for pair (i, j) in a causal-learn endpoint matrix."""
    if G[i, j] == 0 and G[j, i] == 0:
        return None
    if G[i, j] == -1 and G[j, i] == 1:
        return "->"
    if G[i, j] == 1 and G[j, i] == -1:
        return "<-"
    return "--" if G[i, j] == G[j, i] == -1 else "<>"


def diff(G1, G2):
    p = len(G1)
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    adj = [(i, j) for i, j in pairs if (edge_type(G1, i, j) is None) != (edge_type(G2, i, j) is None)]
    ori = [(i, j) for i, j in pairs if edge_type(G1, i, j) and edge_type(G2, i, j)
           and edge_type(G1, i, j) != edge_type(G2, i, j)]
    return adj, ori


def unpermute(G, perm):
    """Map a graph learned on data[:, perm] back to original column labels."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return G[np.ix_(inv, inv)]


def draw(ax, G, title, highlight, pos):
    p = len(G)
    g = nx.DiGraph()
    g.add_nodes_from(range(p))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#dbe7f3", edgecolors="#2f5d8a", node_size=420)
    nx.draw_networkx_labels(g, pos, ax=ax, labels={i: f"X{i}" for i in range(p)}, font_size=8)
    for i in range(p):
        for j in range(i + 1, p):
            t = edge_type(G, i, j)
            if t is None:
                continue
            col = "#c0392b" if (i, j) in highlight else "#333333"
            u, v = (j, i) if t == "<-" else (i, j)
            nx.draw_networkx_edges(g, pos, ax=ax, edgelist=[(u, v)], edge_color=col, width=1.6,
                                   arrows=True, arrowsize=14, node_size=420,
                                   arrowstyle="-|>" if t in ("->", "<-") else "-")
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def main() -> None:
    data = np.loadtxt(OUT / "data.csv", delimiter=",", skiprows=1)
    true = np.load(OUT / "true_dag.npy")
    Gs, Gp = np.load(OUT / "standard_pc.npy"), np.load(OUT / "parallel_pc.npy")
    adj, ori = diff(Gs, Gp)
    n_edges = lambda G: int(np.triu((G != 0) | (G.T != 0)).sum())
    true_skel = (true + true.T) > 0
    skel_err = lambda G: int(np.triu(((G != 0) | (G.T != 0)) != true_skel, 1).sum())

    print(f"=== Standard PC (stable={standard_pc.STABLE}) vs Parallel PC "
          f"(sepset rule '{parallel_pc.SEPSET_RULE}'), Fisher-z, alpha=0.05 ===")
    print(f"edges: standard={n_edges(Gs)}  parallel={n_edges(Gp)}  true DAG={int(true.sum())}")
    print(f"adjacencies identical : {not adj}   ({len(adj)} differing pairs: "
          f"{[(f'X{i}', f'X{j}') for i, j in adj]})")
    print(f"orientations identical (on shared edges): {not ori}   ({len(ori)} differing: "
          + ", ".join(f"X{i}{edge_type(Gs, i, j)}X{j} vs X{i}{edge_type(Gp, i, j)}X{j}"
                      for i, j in ori) + ")")
    print(f"total differing node pairs: {len(adj) + len(ori)}")
    print(f"skeleton errors vs true DAG: standard={skel_err(Gs)}  parallel={skel_err(Gp)}")

    # Control A: parallelism itself
    G1 = parallel_pc.run(data, workers=1)
    print(f"\n[A] parallel PC, 1 vs 4 workers identical: {np.array_equal(G1, Gp)}")

    # Control B: order dependence (10 column permutations)
    rng = np.random.default_rng(0)
    skel = lambda G: ((G != 0) | (G.T != 0)).tobytes()
    res = {"standard": [], "parallel": []}
    for _ in range(10):
        perm = rng.permutation(data.shape[1])
        res["standard"].append(unpermute(standard_pc.run(data[:, perm]), perm))
        res["parallel"].append(unpermute(parallel_pc.run(data[:, perm]), perm))
    for k, Gl in res.items():
        print(f"[B] {k:8s} over 10 column orders: distinct skeletons={len({skel(G) for G in Gl})}"
              f"  distinct CPDAGs={len({G.tobytes() for G in Gl})}")

    # Figure: same layout, differing pairs in red
    pos = nx.circular_layout(range(data.shape[1]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    hl = set(adj) | set(ori)
    draw(axes[0], Gs, f"Standard PC (causal-learn, stable={standard_pc.STABLE}): {n_edges(Gs)} edges", hl, pos)
    draw(axes[1], Gp, f"Parallel PC (Alg. 2, 4 workers, sepset={parallel_pc.SEPSET_RULE}): {n_edges(Gp)} edges", hl, pos)
    fig.suptitle(f"Red = differs between the two CPDAGs ({len(hl)} node pairs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=150)
    print(f"\nfigure: {OUT / 'comparison.png'}")


if __name__ == "__main__":
    main()
