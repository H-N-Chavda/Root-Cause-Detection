"""Each library's native output converted into the shared graph object.

The orientation convention is the thing under test. `tigramite` and `lingam` do
not use the same row and column meaning, and getting it backwards transposes
every graph while leaving every SHD looking plausible. So each converter is
checked against a hand-built matrix whose expected edges are written out, and
then against a real fit on a synthetic system with a known one-way edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_bench.algorithms.pc_manual import pc_result_to_graph
from causal_bench.algorithms.registry import available, build, get_algorithm_class
from causal_bench.algorithms.var_lingam import adjacency_matrices_to_graph
from causal_bench.graph.conversions import lagged_to_summary

# ---------------------------------------------------------------------------
# lingam: row = effect, column = cause
# ---------------------------------------------------------------------------


def test_lingam_converter_transposes():
    """`B[effect, cause]`, so B0[1, 0] = 0.8 is X0 -> X1, not X1 -> X0.

    Hand-built: two lags, one contemporaneous edge X0 -> X1 and one lagged edge
    X0(t-1) -> X2.
    """
    b0 = np.zeros((3, 3))
    b0[1, 0] = 0.8  # equation for X1 has a term in X0  =>  X0 -> X1
    b1 = np.zeros((3, 3))
    b1[2, 0] = 0.9  # equation for X2 has a term in X0(t-1)  =>  X0 -> X2 at lag 1

    graph = adjacency_matrices_to_graph(np.stack([b0, b1]), ["X0", "X1", "X2"])

    assert graph.tau_max == 1
    assert sorted(graph.directed_edges()) == [("X0", "X1", 0), ("X0", "X2", 1)]
    assert graph.links[0, 1, 0] == "-->"
    assert graph.links[1, 0, 0] == "<--"
    assert graph.links[0, 2, 1] == "-->"
    assert graph.links[2, 0, 1] == ""


def test_lingam_converter_is_not_symmetric():
    """The regression guard: a transposed converter would make these equal."""
    forward = np.zeros((2, 2))
    forward[1, 0] = 0.5
    backward = np.zeros((2, 2))
    backward[0, 1] = 0.5

    a = adjacency_matrices_to_graph(forward[None, :, :], ["A", "B"])
    b = adjacency_matrices_to_graph(backward[None, :, :], ["A", "B"])

    assert a.directed_edges() == [("A", "B", 0)]
    assert b.directed_edges() == [("B", "A", 0)]


def test_lingam_converter_ignores_the_lag_zero_diagonal():
    b0 = np.eye(2) * 0.7
    graph = adjacency_matrices_to_graph(b0[None, :, :], ["A", "B"])
    assert graph.directed_edges() == []


def test_lingam_converter_rejects_a_bad_shape():
    with pytest.raises(ValueError, match="coefficient matrices"):
        adjacency_matrices_to_graph(np.zeros((2, 3)), ["A", "B"])


def test_lingam_orientation_on_a_real_fit(synthetic_data, synthetic_names):
    """The empirical check the module docstring cites: fit a system whose
    direction is known and confirm the converted graph agrees."""
    lingam = pytest.importorskip("lingam")

    model = lingam.VARLiNGAM(lags=1, random_state=0).fit(synthetic_data)
    graph = adjacency_matrices_to_graph(model.adjacency_matrices_, synthetic_names)
    summary = lagged_to_summary(graph)
    edges = {(c, e) for c, e, _ in summary.directed_edges()}

    # The generating contemporaneous edges, in the right direction.
    assert ("X1", "X2") in edges
    assert ("X2", "X1") not in edges
    assert ("X3", "X4") in edges
    assert ("X4", "X3") not in edges


# ---------------------------------------------------------------------------
# tigramite: graph[i, j, tau] is the link from (i, t - tau) to (j, t)
# ---------------------------------------------------------------------------


def test_tigramite_convention_on_a_real_fit(synthetic_data, synthetic_names):
    """tigramite already returns the shared convention, so the "converter" is a
    pass-through. That is exactly the claim worth testing."""
    pytest.importorskip("tigramite")

    graph = build("pcmci_plus", seed=0, tau_max=2, pc_alpha=0.01).run(
        synthetic_data, synthetic_names
    )
    assert graph.meta["completed"], graph.meta.get("error")

    summary = lagged_to_summary(graph)
    edges = {(c, e) for c, e, _ in summary.directed_edges()}
    assert ("X1", "X2") in edges
    assert ("X2", "X1") not in edges
    assert ("X4", "X5") in edges
    assert ("X5", "X4") not in edges


# ---------------------------------------------------------------------------
# pc_manual
# ---------------------------------------------------------------------------


def test_pc_manual_converter():
    """`PCResult.directed_edges` holds (parent, child) index pairs, i.e. cause
    first, so the conversion needs no transpose."""
    from causal_bench.algorithms.pc_manual import PCResult

    result = PCResult(
        adjacency={0: {1}, 1: {0, 2}, 2: {1}},
        directed_edges={(0, 1)},
        sepset={},
        undirected_edges={(1, 2)},
        conflicting_edges=set(),
    )

    graph = pc_result_to_graph(result, ["A", "B", "C"])

    assert graph.directed_edges() == [("A", "B", 0)]
    assert graph.undirected_edges() == [("B", "C", 0)]
    assert graph.links[0, 1, 0] == "-->"
    assert graph.links[1, 0, 0] == "<--"


def test_pc_manual_converter_marks_conflicts():
    from causal_bench.algorithms.pc_manual import PCResult

    result = PCResult(
        adjacency={0: {1}, 1: {0}},
        directed_edges=set(),
        sepset={},
        conflicting_edges={(0, 1)},
    )

    graph = pc_result_to_graph(result, ["A", "B"])

    assert graph.links[0, 1, 0] == "x-x"
    assert graph.undirected_edges() == [("A", "B", 0)]


# ---------------------------------------------------------------------------
# the registry and the common interface
# ---------------------------------------------------------------------------


def test_registry_holds_every_algorithm():
    assert set(available()) == {"pc", "pc_manual", "pcmci_plus", "var_lingam", "lste"}


def test_build_drops_parameters_an_algorithm_does_not_accept():
    """Config blocks carry per-algorithm keys; a shared block must not be a
    TypeError for the algorithms that do not use them."""
    algorithm = build("pc_manual", seed=1, pc_alpha=0.02, sig_samples=99)
    assert algorithm.params["pc_alpha"] == 0.02
    assert "sig_samples" not in algorithm.params


def test_unknown_algorithm_is_refused():
    with pytest.raises(KeyError, match="unknown algorithm"):
        get_algorithm_class("nope")


def test_a_failing_algorithm_records_the_failure_instead_of_raising(
    synthetic_data, synthetic_names
):
    """A missing result is a finding; the run must not crash on it."""
    algorithm = build("pc_manual", seed=0)
    graph = algorithm.run(
        synthetic_data, synthetic_names, prior_knowledge={"unsupported": True}
    )

    assert graph.meta["completed"] is False
    assert "prior knowledge" in graph.meta["error"]
    assert graph.n_edges() == 0
    assert graph.meta["assumptions_met"] is None


def test_metadata_records_provenance(synthetic_data, synthetic_names):
    graph = build("pc", seed=7, pc_alpha=0.05).run(synthetic_data, synthetic_names)

    meta = graph.meta
    assert meta["algorithm"] == "pc"
    assert meta["library"] == "tigramite"
    assert meta["library_version"] != "unknown"
    assert meta["seed"] == 7
    assert meta["parameters"]["pc_alpha"] == 0.05
    assert meta["runtime_seconds"] >= 0
    assert meta["collider_rule"] == "majority"


def test_runs_are_reproducible(synthetic_data, synthetic_names):
    first = build("pcmci_plus", seed=0, tau_max=1).run(synthetic_data, synthetic_names)
    second = build("pcmci_plus", seed=0, tau_max=1).run(synthetic_data, synthetic_names)
    assert np.array_equal(first.links, second.links)
