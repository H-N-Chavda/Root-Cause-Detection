"""Prior-knowledge converters, each against a hand-built expected output.

One YAML source, two encodings. The libraries spell "no edge" and "direction
unknown" differently, so a converter that looks right can still forbid what it
meant to force.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_bench.prior.knowledge import (
    LINGAM_NO_PATH,
    LINGAM_PATH,
    LINGAM_UNKNOWN,
    PriorEdge,
    PriorKnowledge,
)

NAMES = ["A", "B", "C"]


@pytest.fixture
def knowledge() -> PriorKnowledge:
    return PriorKnowledge(
        edges=[
            PriorEdge(cause="A", effect="B", kind="forced"),
            PriorEdge(cause="B", effect="A", kind="forbidden"),
            PriorEdge(cause="A", effect="C", kind="forbidden"),
        ]
    )


# ---------------------------------------------------------------------------
# tigramite
# ---------------------------------------------------------------------------


def test_tigramite_encoding(knowledge):
    """`link_assumptions[j][(i, -tau)]` describes (i, t-tau) -> (j, t).

    A contemporaneous link occupies two entries and tigramite requires them to
    agree, so a forced A -> B is "-->" at [B][(A,0)] AND "<--" at [A][(B,0)].
    Forbidding one direction constrains the orientation rather than deleting the
    adjacency. Everything unmentioned keeps the permissive default.
    """
    pytest.importorskip("tigramite")

    assumptions = knowledge.to_tigramite(NAMES, tau_max=1)

    # A -> B forced, written on both sides.
    assert assumptions[1][(0, 0)] == "-->"
    assert assumptions[0][(1, 0)] == "<--"
    # A -> C forbidden: if they are adjacent the edge runs C -> A.
    assert assumptions[2][(0, 0)] == "<--"
    assert assumptions[0][(2, 0)] == "-->"
    # C -> B unmentioned: permissive default.
    assert assumptions[1][(2, 0)] == "o?o"
    assert assumptions[1][(2, -1)] == "o?>"


def test_forbidding_both_directions_removes_the_adjacency():
    """One direction forbidden constrains the orientation; both directions
    forbidden is the only way to say "these two are not adjacent"."""
    pytest.importorskip("tigramite")

    knowledge = PriorKnowledge(
        edges=[
            PriorEdge(cause="A", effect="B", kind="forbidden"),
            PriorEdge(cause="B", effect="A", kind="forbidden"),
        ]
    )
    assumptions = knowledge.to_tigramite(NAMES, tau_max=1)

    assert (0, 0) not in assumptions[1]
    assert (1, 0) not in assumptions[0]
    # The lagged links between them are untouched.
    assert assumptions[1][(0, -1)] == "o?>"


def test_tigramite_keeps_lagged_defaults_for_a_forbidden_contemporaneous_edge(
    knowledge,
):
    """Forbidding A -> C at lag 0 must not forbid it at lag 1: they are
    different links, and a converter that deletes both would silence a lagged
    edge nobody ruled out."""
    pytest.importorskip("tigramite")

    assumptions = knowledge.to_tigramite(NAMES, tau_max=2)

    assert assumptions[2][(0, 0)] == "<--"
    assert assumptions[2][(0, -1)] == "o?>"
    assert assumptions[2][(0, -2)] == "o?>"


def test_tigramite_drops_a_lag_outside_the_searched_range():
    pytest.importorskip("tigramite")

    knowledge = PriorKnowledge(
        edges=[PriorEdge(cause="A", effect="B", kind="forced", lag=5)]
    )
    assumptions = knowledge.to_tigramite(NAMES, tau_max=2)

    # Nothing forced anywhere; the lag-5 constraint has nowhere to go.
    assert all(
        mark in ("o?o", "o?>") for links in assumptions.values() for mark in links.values()
    )


# ---------------------------------------------------------------------------
# lingam
# ---------------------------------------------------------------------------


def test_lingam_encoding(knowledge):
    """`prior_knowledge[effect, cause]` with 1 = path, 0 = no path, -1 unknown.

    Same row = effect, column = cause transpose as the coefficient matrices.
    """
    matrix = knowledge.to_lingam(NAMES)

    assert matrix[1, 0] == LINGAM_PATH  # A -> B forced
    assert matrix[0, 1] == LINGAM_NO_PATH  # B -> A forbidden
    assert matrix[2, 0] == LINGAM_NO_PATH  # A -> C forbidden
    assert matrix[1, 2] == LINGAM_UNKNOWN  # C -> B unmentioned
    # The diagonal is stated: a variable is never its own contemporaneous cause.
    assert list(np.diag(matrix)) == [LINGAM_NO_PATH] * 3


def test_lingam_is_the_transpose_of_the_natural_reading(knowledge):
    """Regression guard: a converter without the transpose would put the forced
    entry at [cause, effect] and reverse every constraint."""
    matrix = knowledge.to_lingam(NAMES)

    assert matrix[1, 0] == LINGAM_PATH
    assert matrix[0, 1] != LINGAM_PATH


def test_lingam_drops_lagged_constraints_with_a_warning(caplog):
    """`DirectLiNGAM.prior_knowledge` constrains the contemporaneous B0 only, so
    a lagged constraint has nowhere to go and must not be silently mapped onto
    the contemporaneous matrix."""
    knowledge = PriorKnowledge(
        edges=[PriorEdge(cause="A", effect="B", kind="forced", lag=1)]
    )

    with caplog.at_level("WARNING"):
        matrix = knowledge.to_lingam(NAMES)

    assert matrix[1, 0] == LINGAM_UNKNOWN
    assert "contemporaneous B0 only" in caplog.text


# ---------------------------------------------------------------------------
# validation and loading
# ---------------------------------------------------------------------------


def test_validation_reports_variables_absent_from_the_data(knowledge):
    knowledge.edges.append(PriorEdge(cause="A", effect="Z", kind="forced"))

    report = knowledge.validate_against(NAMES)

    assert report["n_edges_declared"] == 4
    assert report["n_edges_applicable"] == 3
    assert report["variables_not_in_data"] == ["Z"]


def test_a_self_loop_is_refused():
    with pytest.raises(ValueError, match="self-loop"):
        PriorEdge(cause="A", effect="A", kind="forced")


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="forced or forbidden"):
        PriorEdge(cause="A", effect="B", kind="maybe")


def test_the_repository_controller_loop_file_loads_and_converts():
    """The real file, against the real variable set. X27 has no column in the
    data, so its edge must be dropped rather than crashing the conversion."""
    pytest.importorskip("tigramite")

    from causal_bench import paths

    path = paths.REPO_ROOT / "configs/prior_knowledge/te_controller_loops.yaml"
    knowledge = PriorKnowledge.from_yaml(path)
    observed = [f"X{k + 1}" for k in range(33) if k + 1 not in (27, 31)]

    report = knowledge.validate_against(observed)
    assert report["n_edges_declared"] == 5
    assert report["n_edges_applicable"] == 4
    assert report["variables_not_in_data"] == ["X27"]

    assumptions = knowledge.to_tigramite(observed, tau_max=1)
    i10, i28 = observed.index("X10"), observed.index("X28")
    # Forced X10 -> X28, written consistently on both sides.
    assert assumptions[i28][(i10, 0)] == "-->"
    assert assumptions[i10][(i28, 0)] == "<--"

    matrix = knowledge.to_lingam(observed)
    assert matrix[i28, i10] == LINGAM_PATH
    assert matrix[i10, i28] == LINGAM_NO_PATH


def test_prior_knowledge_actually_reaches_tigramite(synthetic_data, synthetic_names):
    """End to end: a forbidden edge must be absent from the fitted graph.

    X1 -> X2 is a real edge in the generating system, so forbidding it is a
    change the algorithm cannot ignore if the constraint is wired through.
    """
    pytest.importorskip("tigramite")

    from causal_bench.algorithms.registry import build

    knowledge = PriorKnowledge(edges=[PriorEdge(cause="X1", effect="X2", kind="forbidden")])
    assumptions = knowledge.to_tigramite(synthetic_names, tau_max=1)

    without = build("pcmci_plus", seed=0, tau_max=1).run(synthetic_data, synthetic_names)
    with_pk = build("pcmci_plus", seed=0, tau_max=1).run(
        synthetic_data, synthetic_names, prior_knowledge=assumptions
    )

    assert with_pk.meta["completed"], with_pk.meta.get("error")
    assert ("X1", "X2", 0) in without.directed_edges()
    assert ("X1", "X2", 0) not in with_pk.directed_edges()


def test_prior_knowledge_actually_reaches_lingam(synthetic_data, synthetic_names):
    """Same check on the other encoding: forbidding a real contemporaneous edge
    must remove it from B0."""
    pytest.importorskip("lingam")

    from causal_bench.algorithms.registry import build

    knowledge = PriorKnowledge(edges=[PriorEdge(cause="X1", effect="X2", kind="forbidden")])
    matrix = knowledge.to_lingam(synthetic_names)

    without = build("var_lingam", seed=0, tau_max=1, run_bootstrap=False).run(
        synthetic_data, synthetic_names
    )
    with_pk = build("var_lingam", seed=0, tau_max=1, run_bootstrap=False).run(
        synthetic_data, synthetic_names, prior_knowledge=matrix
    )

    assert with_pk.meta["completed"], with_pk.meta.get("error")
    assert ("X1", "X2", 0) in without.directed_edges()
    assert ("X1", "X2", 0) not in with_pk.directed_edges()
