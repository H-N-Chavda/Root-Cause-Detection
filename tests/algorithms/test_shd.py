"""SHD against small graph pairs whose distance is computed by hand.

Every expected number below is worked out in the docstring, so the test fails
if the implementation drifts rather than if it merely changes shape.
"""

from __future__ import annotations

import pytest

from causal_bench.graph.representation import CausalGraph
from causal_bench.scoring.shd import align_to, structural_hamming_distance


def _graph(names: list[str], directed=(), undirected=()) -> CausalGraph:
    graph = CausalGraph.empty(names, tau_max=0)
    index = {n: i for i, n in enumerate(names)}
    for cause, effect in directed:
        i, j = index[cause], index[effect]
        graph.links[i, j, 0] = "-->"
        graph.links[j, i, 0] = "<--"
    for a, b in undirected:
        i, j = index[a], index[b]
        graph.links[i, j, 0] = graph.links[j, i, 0] = "o-o"
    return graph


NAMES = ["A", "B", "C"]


def test_identical_graphs_score_zero():
    truth = _graph(NAMES, directed=[("A", "B"), ("B", "C")])
    assert structural_hamming_distance(truth, truth).shd == 0.0


def test_one_missing_edge_costs_one():
    """Truth A->B, B->C. Estimate A->B only. One false negative => 1."""
    truth = _graph(NAMES, directed=[("A", "B"), ("B", "C")])
    estimate = _graph(NAMES, directed=[("A", "B")])

    result = structural_hamming_distance(estimate, truth)

    assert result.shd == 1.0
    assert result.true_positives == 1
    assert result.false_negatives == 1
    assert result.false_positives == 0


def test_one_extra_edge_costs_one():
    """Truth A->B. Estimate A->B, A->C. One false positive => 1."""
    truth = _graph(NAMES, directed=[("A", "B")])
    estimate = _graph(NAMES, directed=[("A", "B"), ("A", "C")])

    result = structural_hamming_distance(estimate, truth)

    assert result.shd == 1.0
    assert result.false_positives == 1
    assert result.true_positives == 1


def test_a_reversed_edge_costs_one():
    """Truth A->B. Estimate B->A. Adjacency right, direction wrong => 1."""
    truth = _graph(NAMES, directed=[("A", "B")])
    estimate = _graph(NAMES, directed=[("B", "A")])

    result = structural_hamming_distance(estimate, truth)

    assert result.shd == 1.0
    assert result.reversed_edges == 1
    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_an_abstention_costs_the_undirected_weight():
    """Truth A->B directed. Estimate A-B undirected. The adjacency is right and
    the algorithm declined to orient, which is better than guessing wrong and
    worse than guessing right => the configured weight, 0.5 by default."""
    truth = _graph(NAMES, directed=[("A", "B")])
    estimate = _graph(NAMES, undirected=[("A", "B")])

    result = structural_hamming_distance(estimate, truth)

    assert result.shd == 0.5
    assert result.unoriented_edges == 1
    assert result.reversed_edges == 0


def test_the_undirected_weight_is_a_parameter():
    truth = _graph(NAMES, directed=[("A", "B")])
    estimate = _graph(NAMES, undirected=[("A", "B")])

    assert structural_hamming_distance(estimate, truth, 0.0).shd == 0.0
    assert structural_hamming_distance(estimate, truth, 1.0).shd == 1.0
    assert structural_hamming_distance(estimate, truth, 0.25).shd == 0.25


def test_over_committing_on_an_undirected_truth_edge_also_costs_the_weight():
    """Truth A-B undirected (a CPDAG abstention). Estimate A->B. The estimate
    claims more than the truth supports, symmetric to the abstention case."""
    truth = _graph(NAMES, undirected=[("A", "B")])
    estimate = _graph(NAMES, directed=[("A", "B")])

    result = structural_hamming_distance(estimate, truth)

    assert result.shd == 0.5
    assert result.unoriented_edges == 1


def test_a_worked_mixed_example():
    """Truth: A->B, B->C, and A-C absent.
    Estimate: A->B (correct, 0), C->B (reversed, 1), A-C undirected (extra, 1).
    Total = 2.0, with TP 1, reversed 1, FP 1.
    """
    truth = _graph(NAMES, directed=[("A", "B"), ("B", "C")])
    estimate = _graph(NAMES, directed=[("A", "B"), ("C", "B")], undirected=[("A", "C")])

    result = structural_hamming_distance(estimate, truth)

    assert result.shd == 2.0
    assert result.true_positives == 1
    assert result.reversed_edges == 1
    assert result.false_positives == 1
    assert result.false_negatives == 0


def test_an_empty_estimate_scores_the_edge_count():
    truth = _graph(NAMES, directed=[("A", "B"), ("B", "C")])
    empty = _graph(NAMES)

    result = structural_hamming_distance(empty, truth)

    assert result.shd == 2.0
    assert result.false_negatives == 2


def test_a_complete_estimate_scores_every_non_edge():
    """3 variables give 3 pairs. Truth has 1 edge, so a complete graph adds 2
    false positives and gets the one real edge right."""
    truth = _graph(NAMES, directed=[("A", "B")])
    complete = _graph(NAMES, directed=[("A", "B"), ("A", "C"), ("B", "C")])

    result = structural_hamming_distance(complete, truth)

    assert result.shd == 2.0
    assert result.false_positives == 2
    assert result.true_positives == 1


def test_both_directions_asserted_counts_as_unsettled():
    """An estimate claiming A->B and B->A has not claimed an orientation, so it
    is scored as an abstention rather than as a reversal."""
    truth = _graph(NAMES, directed=[("A", "B")])
    estimate = CausalGraph.empty(NAMES, tau_max=0)
    estimate.links[0, 1, 0] = "-->"
    estimate.links[1, 0, 0] = "-->"

    result = structural_hamming_distance(estimate, truth)

    assert result.shd == 0.5
    assert result.unoriented_edges == 1


def test_lagged_estimate_is_read_by_adjacency():
    """A lagged graph passed straight in has its adjacency read across lags,
    rather than lag 0 alone being compared and the rest silently ignored."""
    truth = _graph(NAMES, directed=[("A", "B")])
    estimate = CausalGraph.empty(NAMES, tau_max=2)
    estimate.links[0, 1, 2] = "-->"

    assert structural_hamming_distance(estimate, truth).shd == 0.0


def test_mismatched_variable_sets_are_refused():
    truth = _graph(["A", "B"], directed=[("A", "B")])
    estimate = _graph(["A", "B", "C"])

    with pytest.raises(ValueError, match="same variables"):
        structural_hamming_distance(estimate, truth)


def test_weight_outside_the_unit_interval_is_refused():
    truth = _graph(NAMES)
    with pytest.raises(ValueError, match="undirected_weight"):
        structural_hamming_distance(truth, truth, undirected_weight=2.0)


def test_align_to_reorders_columns():
    """Two graphs built from different sources rarely arrive in the same column
    order, and a mismatched order silently scores variable 1 against variable 3.
    """
    graph = _graph(["C", "A", "B"], directed=[("A", "B")])

    aligned = align_to(graph, ["A", "B", "C"])

    assert aligned.var_names == ["A", "B", "C"]
    assert aligned.directed_edges() == [("A", "B", 0)]
    assert aligned.links[0, 1, 0] == "-->"


def test_align_to_refuses_a_missing_variable():
    graph = _graph(["A", "B"], directed=[("A", "B")])

    with pytest.raises(ValueError, match="missing variables"):
        align_to(graph, ["A", "B", "C"])


def test_aligning_before_scoring_changes_the_answer():
    """The point of `align_to`: without it the same two graphs score wrongly."""
    truth = _graph(["A", "B", "C"], directed=[("A", "B")])
    shuffled = _graph(["C", "A", "B"], directed=[("A", "B")])

    assert (
        structural_hamming_distance(align_to(shuffled, truth.var_names), truth).shd == 0.0
    )
