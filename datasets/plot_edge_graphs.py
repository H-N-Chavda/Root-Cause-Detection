"""Ground truth next to each recovered graph, raw vs copula, same node layout.

Edges come from the `directed_edges` field of each saved graph JSON, collapsed
over lags to unique (cause, effect) pairs. Self-loops are dropped. Colour:
black = true positive, red = false positive, dashed grey = missing (false
negative). Output: results/qtank/figures/edge_graphs_<dataset>.png

    python plot_edge_graphs.py [--raw run-parcorr-tau20-s3] [--copula run-copula-tau20-s3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "generated"
ALGOS = ["pc", "pcmci_plus", "var_lingam"]
LABELS = {"pc": "PC", "pcmci_plus": "PCMCI+", "var_lingam": "VAR-LiNGAM"}

# Fixed layout: inputs on top, upper tanks in the middle, lower tanks at the bottom.
POS = {
    "v1": (0.0, 2.0), "d": (1.5, 2.0), "v2": (3.0, 2.0),
    "h3": (0.5, 1.0), "h4": (2.5, 1.0),
    "h1": (0.5, 0.0), "h2": (2.5, 0.0),
}
TP, FP, FN = "#000000", "#C0392B", "#9AA0A6"


def unique_edges(path: Path) -> set[tuple[str, str]]:
    g = json.loads(path.read_text())
    return {(e["cause"], e["effect"]) for e in g["directed_edges"] if e["cause"] != e["effect"]}


def draw(ax, edges: set, truth: set, title: str) -> None:
    G = nx.DiGraph()
    G.add_nodes_from(POS)
    nx.draw_networkx_nodes(G, POS, ax=ax, node_size=900, node_color="white",
                           edgecolors="black", linewidths=1.2)
    nx.draw_networkx_labels(G, POS, ax=ax, font_size=11)
    groups = [
        (edges & truth, TP, "solid"),
        (edges - truth, FP, "solid"),
        (truth - edges, FN, "dashed"),
    ]
    for es, colour, style in groups:
        if es:
            nx.draw_networkx_edges(
                G, POS, edgelist=sorted(es), ax=ax, edge_color=colour, style=style,
                arrows=True, arrowstyle="-|>", arrowsize=16, width=1.6,
                node_size=900, connectionstyle="arc3,rad=0.12")
    tp, fp, fn = len(edges & truth), len(edges - truth), len(truth - edges)
    ax.set_title(f"{title}\nTP {tp}  FP {fp}  FN {fn}", fontsize=11)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="run-parcorr-tau20-s3")
    ap.add_argument("--copula", default="run-copula-tau20-s3")
    ap.add_argument("--datasets", default="P_minus,P_plus")
    args = ap.parse_args()

    truth = {tuple(e) for e in json.loads((DATA / "ground_truth_edges.json").read_text())["edges"]}
    out_dir = REPO / "results" / "qtank" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {"raw": REPO / "results/qtank" / args.raw, "copula": REPO / "results/qtank" / args.copula}

    for ds in args.datasets.split(","):
        fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
        for r, (tag, run) in enumerate(runs.items()):
            draw(axes[r, 0], truth, truth, "Ground truth")
            for c, algo in enumerate(ALGOS, start=1):
                edges = unique_edges(run / "graphs" / f"{ds}__{algo}.json")
                draw(axes[r, c], edges, truth, f"{LABELS[algo]}, {tag}")
        handles = [
            Line2D([], [], color=TP, lw=1.6, label="true positive"),
            Line2D([], [], color=FP, lw=1.6, label="false positive"),
            Line2D([], [], color=FN, lw=1.6, ls="--", label="missing"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11)
        fig.suptitle(f"{ds}: unique directed edges, lags collapsed  (top: {args.raw}, bottom: {args.copula})",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        path = out_dir / f"edge_graphs_{ds}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
