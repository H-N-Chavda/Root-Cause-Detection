"""The scoring targets built from a ground-truth adjacency matrix.

Three graphs, per decision 2, plus a CPDAG of each:

``gt_full``
    The raw N-node graph as the file gives it. Reference only -- no algorithm
    run on the observed columns can be scored against nodes it never saw.

``gt_induced``
    The induced subgraph on the observed variables. Every edge touching a
    missing node is dropped. The strict, conservative target.

``gt_projected``
    The latent projection onto the observed variables. A missing node that
    mediates ``A -> L -> B`` leaves ``A -> B``, because that dependence survives
    marginalisation and a method run on the observed data can and should find
    it. **This is the primary target**: it is the graph the observed data can
    actually support.

Latent nodes are *derived*, never listed in config: the set is whatever the
ground truth names and the data does not have, so this works on a dataset
nobody has looked at yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from ..graph.conversions import (
    count_v_structures,
    dag_to_cpdag,
    digraph_to_summary,
    induced_subgraph,
    latent_projection,
)
from ..graph.representation import CausalGraph
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ScoringTargets:
    """Every graph an estimate can be scored against, plus how they were built."""

    var_names: list[str]
    latent_nodes: list[str]
    gt_full: nx.DiGraph
    gt_induced: CausalGraph
    gt_projected: CausalGraph
    gt_induced_cpdag: CausalGraph
    gt_projected_cpdag: CausalGraph
    provenance: dict[str, Any]

    def by_name(self) -> dict[str, CausalGraph]:
        """The four scoreable targets, in report order."""
        return {
            "gt_projected": self.gt_projected,
            "gt_projected_cpdag": self.gt_projected_cpdag,
            "gt_induced": self.gt_induced,
            "gt_induced_cpdag": self.gt_induced_cpdag,
        }


#: Ground-truth node names are reconstructed as <prefix><k+1> from row k. The
#: file carries no names, so the data's own naming scheme is the only source.
def _node_names(prefix: str, n_nodes: int) -> list[str]:
    return [f"{prefix}{k + 1}" for k in range(n_nodes)]


def build_targets(
    matrix: np.ndarray,
    observed_names: Sequence[str],
    node_prefix: str = "X",
) -> ScoringTargets:
    """Builds the scoring targets from a raw adjacency matrix.

    `matrix[i, j] == 1` means node i -> node j, which is the ground-truth file
    convention and the *opposite* of lingam's coefficient matrices.
    """
    matrix = (np.asarray(matrix) != 0).astype(int)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"ground truth must be square, got {matrix.shape}")

    n_nodes = matrix.shape[0]
    full_names = _node_names(node_prefix, n_nodes)
    observed = [str(n) for n in observed_names]

    gt_full = nx.DiGraph()
    gt_full.add_nodes_from(full_names)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and matrix[i, j]:
                gt_full.add_edge(full_names[i], full_names[j])

    unknown = sorted(set(observed) - set(full_names))
    if unknown:
        raise ValueError(
            f"data columns {unknown} have no row in the ground truth; the node "
            "naming convention does not line up"
        )
    latents = [n for n in full_names if n not in set(observed)]

    if not nx.is_directed_acyclic_graph(gt_full):
        raise ValueError(
            "ground truth is cyclic; the CPDAG construction and the latent "
            "projection both assume acyclicity"
        )

    induced = induced_subgraph(gt_full, observed)
    projected = latent_projection(gt_full, latents)

    added = sorted(set(projected.edges()) - set(induced.edges()))
    dropped = sorted(set(gt_full.edges()) - set(induced.edges()))
    log.info(
        "targets: full %d edges, induced %d, projected %d (latents %s; projection adds %s)",
        gt_full.number_of_edges(),
        induced.number_of_edges(),
        projected.number_of_edges(),
        latents or "none",
        added or "nothing",
    )

    provenance = {
        "node_prefix": node_prefix,
        "n_ground_truth_nodes": n_nodes,
        "n_observed": len(observed),
        "latent_nodes": latents,
        "gt_full_edges": gt_full.number_of_edges(),
        "gt_full_acyclic": True,
        "gt_full_self_loops": int(nx.number_of_selfloops(gt_full)),
        "gt_induced_edges": induced.number_of_edges(),
        "gt_projected_edges": projected.number_of_edges(),
        "edges_dropped_by_induction": [list(e) for e in dropped],
        "edges_added_by_projection": [list(e) for e in added],
        "v_structures_full": count_v_structures(gt_full),
        "v_structures_projected": count_v_structures(projected),
        "primary_target": "gt_projected",
    }

    targets = ScoringTargets(
        var_names=observed,
        latent_nodes=latents,
        gt_full=gt_full,
        gt_induced=digraph_to_summary(induced, observed),
        gt_projected=digraph_to_summary(projected, observed),
        gt_induced_cpdag=dag_to_cpdag(induced, observed),
        gt_projected_cpdag=dag_to_cpdag(projected, observed),
        provenance=provenance,
    )

    # A CPDAG's undirected edges are what no observational method can orient, so
    # this count is the floor on any CPDAG-scored result and belongs in the
    # report next to it.
    for name, graph in (
        ("gt_induced_cpdag", targets.gt_induced_cpdag),
        ("gt_projected_cpdag", targets.gt_projected_cpdag),
    ):
        provenance[name] = {
            "directed": len(graph.directed_edges()),
            "undirected": len(graph.undirected_edges()),
        }
    provenance["gt_full_cpdag"] = _full_cpdag_counts(gt_full, full_names)
    return targets


def _full_cpdag_counts(gt_full: nx.DiGraph, full_names: list[str]) -> dict[str, int]:
    """CPDAG counts for the reference graph, reported but never scored against."""
    cpdag = dag_to_cpdag(gt_full, full_names)
    return {
        "directed": len(cpdag.directed_edges()),
        "undirected": len(cpdag.undirected_edges()),
        "v_structures": count_v_structures(gt_full),
    }
