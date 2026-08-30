"""Graph conversions, each against a hand-built expected output.

Every expected value here is written out by hand rather than computed by the
code under test. A transposed matrix or a reversed edge produces a graph that
still looks plausible and an SHD that still looks reasonable, so these are the
tests that have to be independent of the implementation.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from causal_bench.graph.conversions import (
    adjacency_matrix_to_graph,
    count_v_structures,
    dag_to_cpdag,
    digraph_to_summary,
    induced_subgraph,
    lagged_to_summary,
    latent_projection,
    summary_to_digraph,
)
from causal_bench.graph.representation import CausalGraph

# ---------------------------------------------------------------------------
# lagged -> summary
# ---------------------------------------------------------------------------


def test_lagged_to_summary_collapses_over_lags():
    """A -> B held only at lag 2 must appear in the summary graph."""
    graph = CausalGraph.empty(["A", "B", "C"], tau_max=2)
    graph.links[0, 1, 2] = "-->"  # A(t-2) -> B(t)

    summary = lagged_to_summary(graph)

    assert summary.tau_max == 0
    assert summary.directed_edges() == [("A", "B", 0)]
    assert summary.links[0, 1, 0] == "-->"
    assert summary.links[1, 0, 0] == "<--"
    assert summary.n_edges() == 1


def test_lagged_to_summary_drops_self_loops():
    """A variable's own past is autocorrelation, not a causal claim between two
    variables, and the ground truth has no self-loops to score it against."""
    graph = CausalGraph.empty(["A", "B"], tau_max=1)
    graph.links[0, 0, 1] = "-->"  # A(t-1) -> A(t)
    graph.links[0, 1, 1] = "-->"  # A(t-1) -> B(t)

    summary = lagged_to_summary(graph)

    assert summary.directed_edges() == [("A", "B", 0)]
    assert summary.n_edges() == 1


def test_lagged_to_summary_marks_a_two_way_pair_as_conflicting():
    """A -> B at one lag and B -> A at another cannot both be an orientation in
    an acyclic summary, so the edge is marked conflicting rather than one
    direction being silently picked."""
    graph = CausalGraph.empty(["A", "B"], tau_max=2)
    graph.links[0, 1, 1] = "-->"
    graph.links[1, 0, 2] = "-->"

    summary = lagged_to_summary(graph)

    assert summary.links[0, 1, 0] == "x-x"
    assert summary.links[1, 0, 0] == "x-x"
    assert summary.directed_edges() == []
    assert summary.undirected_edges() == [("A", "B", 0)]


def test_lagged_to_summary_keeps_an_undirected_contemporaneous_edge():
    graph = CausalGraph.empty(["A", "B"], tau_max=1)
    graph.links[0, 1, 0] = graph.links[1, 0, 0] = "o-o"

    summary = lagged_to_summary(graph)

    assert summary.undirected_edges() == [("A", "B", 0)]
    assert summary.directed_edges() == []


# ---------------------------------------------------------------------------
# adjacency matrix <-> graph
# ---------------------------------------------------------------------------


def test_adjacency_matrix_uses_row_cause_column_effect():
    """The ground-truth file convention: `matrix[i, j] == 1` means i -> j.

    This is the opposite of lingam's coefficient matrices, so it is asserted
    explicitly rather than assumed.
    """
    matrix = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])

    graph = adjacency_matrix_to_graph(matrix, ["A", "B", "C"])

    assert graph.directed_edges() == [("A", "B", 0)]
    assert graph.links[0, 1, 0] == "-->"
    assert graph.links[1, 0, 0] == "<--"


def test_digraph_round_trip():
    digraph = nx.DiGraph([("A", "B"), ("B", "C")])
    digraph.add_node("D")

    summary = digraph_to_summary(digraph, ["A", "B", "C", "D"])
    back = summary_to_digraph(summary)

    assert set(back.edges()) == {("A", "B"), ("B", "C")}
    assert set(back.nodes()) == {"A", "B", "C", "D"}


def test_summary_to_digraph_drops_undirected_edges():
    """A DiGraph cannot say "adjacent, direction unknown"; rendering an
    undirected edge as two arcs would invent two claims."""
    graph = CausalGraph.empty(["A", "B"], tau_max=0)
    graph.links[0, 1, 0] = graph.links[1, 0, 0] = "o-o"

    assert summary_to_digraph(graph).number_of_edges() == 0


def test_summary_to_digraph_rejects_a_lagged_graph():
    with pytest.raises(ValueError, match="summary graph"):
        summary_to_digraph(CausalGraph.empty(["A", "B"], tau_max=1))


# ---------------------------------------------------------------------------
# latent projection
# ---------------------------------------------------------------------------


def test_latent_projection_connects_a_mediators_parent_to_its_child():
    """A -> L -> B with L latent must leave A -> B: the dependence survives
    marginalisation, so the observed data can support that edge."""
    digraph = nx.DiGraph([("A", "L"), ("L", "B")])

    projected = latent_projection(digraph, ["L"])

    assert set(projected.edges()) == {("A", "B")}
    assert "L" not in projected.nodes


def test_latent_projection_drops_a_source_latent_with_one_child():
    digraph = nx.DiGraph([("L", "A"), ("A", "B")])

    projected = latent_projection(digraph, ["L"])

    assert set(projected.edges()) == {("A", "B")}


def test_latent_projection_rejects_a_pure_confounder():
    """A latent common cause induces a bidirected edge, which a DAG cannot
    express. Rejecting is correct; silently emitting a directed edge would
    invent an orientation."""
    digraph = nx.DiGraph([("L", "A"), ("L", "B")])

    with pytest.raises(ValueError, match="bidirected"):
        latent_projection(digraph, ["L"])


def test_latent_projection_is_transitive_over_a_chain_of_latents():
    digraph = nx.DiGraph([("A", "L1"), ("L1", "L2"), ("L2", "B")])

    projected = latent_projection(digraph, ["L1", "L2"])

    assert set(projected.edges()) == {("A", "B")}


def test_induced_subgraph_drops_every_edge_touching_a_latent():
    digraph = nx.DiGraph([("A", "L"), ("L", "B"), ("A", "B")])

    induced = induced_subgraph(digraph, ["A", "B"])

    assert set(induced.edges()) == {("A", "B")}


# ---------------------------------------------------------------------------
# the real Tennessee Eastman targets (decision 2)
# ---------------------------------------------------------------------------


def test_tennessee_eastman_projection_matches_decision_2(
    te_ground_truth_matrix, te_full_names, te_observed_names
):
    """The four named edges, and the two the projection recovers.

    X27 mediates X5 -> X27 -> X20 and X11 -> X27 -> X20, so marginalising it
    adds X5 -> X20 and X11 -> X20. X31 is a sink, so X19 -> X31 simply goes.
    """
    digraph = nx.DiGraph()
    digraph.add_nodes_from(te_full_names)
    for i in range(33):
        for j in range(33):
            if i != j and te_ground_truth_matrix[i, j]:
                digraph.add_edge(te_full_names[i], te_full_names[j])

    assert digraph.number_of_edges() == 32
    assert nx.is_directed_acyclic_graph(digraph)
    assert nx.number_of_selfloops(digraph) == 0

    induced = induced_subgraph(digraph, te_observed_names)
    projected = latent_projection(digraph, ["X27", "X31"])

    # The four edges decision 2 names, all dropped by induction.
    assert sorted(set(digraph.edges()) - set(induced.edges())) == [
        ("X11", "X27"),
        ("X19", "X31"),
        ("X27", "X20"),
        ("X5", "X27"),
    ]
    assert induced.number_of_edges() == 28

    # The two the projection recovers.
    assert sorted(set(projected.edges()) - set(induced.edges())) == [
        ("X11", "X20"),
        ("X5", "X20"),
    ]
    assert projected.number_of_edges() == 30
    assert projected.number_of_nodes() == 31


# ---------------------------------------------------------------------------
# DAG -> CPDAG
# ---------------------------------------------------------------------------


def test_cpdag_leaves_a_chain_undirected():
    """A -> B -> C has no unshielded collider, so every DAG in its equivalence
    class agrees on the skeleton and on nothing else."""
    digraph = nx.DiGraph([("A", "B"), ("B", "C")])

    cpdag = dag_to_cpdag(digraph, ["A", "B", "C"])

    assert cpdag.directed_edges() == []
    assert cpdag.undirected_edges() == [("A", "B", 0), ("B", "C", 0)]


def test_cpdag_orients_an_unshielded_collider():
    """A -> C <- B with A and B non-adjacent is a v-structure, and every DAG in
    the class shares it, so both edges stay directed."""
    digraph = nx.DiGraph([("A", "C"), ("B", "C")])

    cpdag = dag_to_cpdag(digraph, ["A", "B", "C"])

    assert sorted(cpdag.directed_edges()) == [("A", "C", 0), ("B", "C", 0)]
    assert cpdag.undirected_edges() == []


def test_cpdag_does_not_orient_a_shielded_triple():
    digraph = nx.DiGraph([("A", "C"), ("B", "C"), ("A", "B")])

    cpdag = dag_to_cpdag(digraph, ["A", "B", "C"])

    assert cpdag.directed_edges() == []
    assert len(cpdag.undirected_edges()) == 3


def test_cpdag_applies_meek_rule_1():
    """A -> C <- B orients the collider; Meek R1 then propagates C -> D, because
    orienting D -> C would create a second collider that is not in the DAG."""
    digraph = nx.DiGraph([("A", "C"), ("B", "C"), ("C", "D")])

    cpdag = dag_to_cpdag(digraph, ["A", "B", "C", "D"])

    assert ("C", "D", 0) in cpdag.directed_edges()
    assert cpdag.undirected_edges() == []


def test_cpdag_rejects_a_cyclic_input():
    with pytest.raises(ValueError, match="acyclic"):
        dag_to_cpdag(nx.DiGraph([("A", "B"), ("B", "A")]), ["A", "B"])


def test_tennessee_eastman_cpdag_matches_decision_4(te_ground_truth_matrix, te_full_names):
    """26 directed and 6 undirected, from 9 v-structures.

    Those 6 are undirectable from observational data by any method, which is
    what makes the CPDAG the right scoring target for a CPDAG-returning
    algorithm.
    """
    digraph = nx.DiGraph()
    digraph.add_nodes_from(te_full_names)
    for i in range(33):
        for j in range(33):
            if i != j and te_ground_truth_matrix[i, j]:
                digraph.add_edge(te_full_names[i], te_full_names[j])

    cpdag = dag_to_cpdag(digraph, te_full_names)

    assert len(cpdag.directed_edges()) == 26
    assert len(cpdag.undirected_edges()) == 6
    assert len(cpdag.directed_edges()) + len(cpdag.undirected_edges()) == 32
    assert count_v_structures(digraph) == 9
