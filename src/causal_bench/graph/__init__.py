"""Graph representation, conversions and serialisation.

One object crosses every boundary in this phase: `CausalGraph`, laid out in
tigramite's `(N, N, tau_max + 1)` convention. Each algorithm wrapper converts
its library's native output into it, and everything downstream -- scoring,
plotting, serialisation -- reads only this type.
"""

from .conversions import (
    adjacency_matrix_to_graph,
    count_v_structures,
    dag_to_cpdag,
    digraph_to_summary,
    induced_subgraph,
    lagged_to_summary,
    latent_projection,
    summary_to_digraph,
)
from .io import load_graph, load_ground_truth_matrix, save_graph
from .representation import (
    DIRECTED_MARKS,
    EDGE_MARKS,
    UNDIRECTED_MARKS,
    CausalGraph,
    RunMetadata,
)

__all__ = [
    "CausalGraph",
    "RunMetadata",
    "EDGE_MARKS",
    "DIRECTED_MARKS",
    "UNDIRECTED_MARKS",
    "lagged_to_summary",
    "summary_to_digraph",
    "digraph_to_summary",
    "adjacency_matrix_to_graph",
    "latent_projection",
    "induced_subgraph",
    "dag_to_cpdag",
    "count_v_structures",
    "save_graph",
    "load_graph",
    "load_ground_truth_matrix",
]
