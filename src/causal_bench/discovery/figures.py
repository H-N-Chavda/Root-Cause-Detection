"""One graph plot per run, plus the ground truth.

Layout is computed once from the ground truth and reused for every estimate, so
the plots are visually comparable: a node sits in the same place in all of them
and the difference between two panels is the edges, not the arrangement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

# Non-interactive backend, chosen before pyplot is imported: this runs headless.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from ..graph.conversions import lagged_to_summary, summary_to_digraph  # noqa: E402
from ..graph.representation import CausalGraph  # noqa: E402
from ..utils.logging import get_logger  # noqa: E402
from .targets import ScoringTargets  # noqa: E402

log = get_logger(__name__)

DPI = 130


def _layout(graph: nx.DiGraph, seed: int = 0) -> dict[str, Any]:
    # Seeded so the same graph always draws the same way.
    return nx.spring_layout(graph, seed=seed, k=1.2)


def _draw(
    digraph: nx.DiGraph,
    layout: dict[str, Any],
    title: str,
    path: Path,
    undirected: list[tuple[str, str]] | None = None,
    truth: nx.DiGraph | None = None,
) -> str:
    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_nodes(
        digraph,
        layout,
        ax=ax,
        node_size=420,
        node_color="#dde5f0",
        edgecolors="#5a6b82",
        linewidths=0.8,
    )
    nx.draw_networkx_labels(digraph, layout, ax=ax, font_size=7)

    edges = list(digraph.edges())
    colours: str | list[str]
    if truth is None:
        colours = "#33475b"
    else:
        # Green where the estimate agrees with the truth, red where it does not.
        colours = ["#2e7d32" if truth.has_edge(u, v) else "#c62828" for u, v in edges]
    if edges:
        nx.draw_networkx_edges(
            digraph,
            layout,
            ax=ax,
            edgelist=edges,
            edge_color=colours,
            arrows=True,
            arrowsize=9,
            width=1.0,
            alpha=0.8,
            connectionstyle="arc3,rad=0.06",
        )
    if undirected:
        nx.draw_networkx_edges(
            digraph,
            layout,
            ax=ax,
            edgelist=undirected,
            arrows=False,
            edge_color="#f9a825",
            width=1.4,
            style="dashed",
            alpha=0.9,
        )
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path.name


def build_all(
    graphs: dict[str, CausalGraph], targets: ScoringTargets, out_dir: Path
) -> dict[str, str]:
    """Draws the ground truth and every run. A failing figure is logged and
    skipped rather than aborting a run whose numbers are already computed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = summary_to_digraph(targets.gt_projected)
    truth.add_nodes_from(targets.var_names)
    layout = _layout(truth)

    produced: dict[str, str] = {}
    try:
        produced["ground_truth"] = _draw(
            truth,
            layout,
            f"Ground truth (gt_projected): {truth.number_of_edges()} edges",
            out_dir / "ground_truth.png",
        )
    except Exception as exc:  # pragma: no cover
        log.warning("ground-truth figure failed: %s", exc)

    for label, graph in graphs.items():
        try:
            if not graph.meta.get("completed", True):
                continue
            summary = graph if graph.tau_max == 0 else lagged_to_summary(graph)
            digraph = summary_to_digraph(summary)
            digraph.add_nodes_from(targets.var_names)
            undirected = [(a, b) for a, b, _ in summary.undirected_edges()]
            produced[label] = _draw(
                digraph,
                layout,
                f"{label}: {summary.n_edges()} edges "
                f"(green agrees with gt_projected, red does not, "
                f"dashed amber undirected)",
                out_dir / f"{label}.png",
                undirected=undirected,
                truth=truth,
            )
        except Exception as exc:
            log.warning("figure for %s failed: %s", label, exc)
    log.info("figures: %d written", len(produced))
    return produced
