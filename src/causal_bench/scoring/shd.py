"""Structural Hamming Distance, and nothing else.

SHD is an edge-level distance: for each unordered pair of variables, compare
what the estimate claims against what the truth says, and charge for a
mismatch. The only judgement call is what to charge when one side is undirected
and the other is directed, so that charge is a parameter rather than a buried
constant.

An undirected edge from a CPDAG is an *abstention*, not an error. A method that
correctly finds the adjacency and correctly declines to orient it has done
strictly better than one that guesses and gets it wrong, and strictly worse than
one that orients it correctly. Charging it 0.5 places it between the two; the
weight is reported next to every score so the choice stays visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Literal

from ..graph.representation import (
    DIRECTED_MARKS,
    UNDIRECTED_MARKS,
    CausalGraph,
)

#: What one unordered pair can look like in either graph.
EdgeState = Literal["absent", "forward", "backward", "undirected"]

#: Default charge for a directed-versus-undirected mismatch.
DEFAULT_UNDIRECTED_WEIGHT = 0.5


@dataclass(frozen=True)
class SHDResult:
    """The distance plus the counts it decomposes into.

    The counts matter as much as the total: two algorithms can reach the same
    SHD by opposite routes, one missing every edge and one inventing them, and
    only the breakdown separates those.
    """

    shd: float
    true_positives: int
    false_positives: int
    false_negatives: int
    reversed_edges: int
    unoriented_edges: int
    undirected_weight: float
    n_predicted_edges: int
    n_true_edges: int
    n_variables: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _edge_state(graph: CausalGraph, i: int, j: int) -> EdgeState:
    """How `graph` describes the unordered pair (i, j), collapsed over lags.

    A summary graph is expected (tau_max = 0), but the loop over lags is kept so
    a caller that passes a lagged graph gets its adjacency read rather than a
    silent wrong answer on lag 0 alone.
    """
    forward = backward = adjacent = False
    for tau in range(graph.tau_max + 1):
        ij, ji = str(graph.links[i, j, tau]), str(graph.links[j, i, tau])
        if ij in DIRECTED_MARKS:
            forward = adjacent = True
        if ji in DIRECTED_MARKS:
            backward = adjacent = True
        if ij in UNDIRECTED_MARKS or ji in UNDIRECTED_MARKS:
            adjacent = True

    if not adjacent:
        return "absent"
    if forward and backward:
        # Both directions asserted: no single orientation is claimed, so this is
        # an unsettled adjacency, the same class as an explicit `x-x`.
        return "undirected"
    if forward:
        return "forward"
    if backward:
        return "backward"
    return "undirected"


def structural_hamming_distance(
    predicted: CausalGraph,
    truth: CausalGraph,
    undirected_weight: float = DEFAULT_UNDIRECTED_WEIGHT,
) -> SHDResult:
    """SHD between two graphs over the same variables, in the same order.

    Charges, per unordered pair:

    ==========================  ======================  ==================
    truth                       estimate                charge
    ==========================  ======================  ==================
    absent                      absent                  0
    absent                      any edge                1  (false positive)
    any edge                    absent                  1  (false negative)
    directed i->j               directed i->j           0  (true positive)
    directed i->j               directed j->i           1  (reversed)
    directed                    undirected              `undirected_weight`
    undirected                  undirected              0  (true positive)
    undirected                  directed                `undirected_weight`
    ==========================  ======================  ==================
    """
    if list(predicted.var_names) != list(truth.var_names):
        raise ValueError(
            "SHD needs both graphs over the same variables in the same order. "
            f"Estimate has {len(predicted.var_names)}, truth has "
            f"{len(truth.var_names)}. Use `align_to` first."
        )

    n = predicted.n_vars
    if not 0.0 <= undirected_weight <= 1.0:
        raise ValueError(f"undirected_weight must be in [0, 1], got {undirected_weight}")

    total = 0.0
    tp = fp = fn = reversed_edges = unoriented = 0

    for i, j in combinations(range(n), 2):
        estimate = _edge_state(predicted, i, j)
        actual = _edge_state(truth, i, j)

        if estimate == "absent" and actual == "absent":
            continue
        if actual == "absent":
            fp += 1
            total += 1.0
        elif estimate == "absent":
            fn += 1
            total += 1.0
        elif estimate == actual:
            tp += 1
        elif estimate == "undirected" or actual == "undirected":
            # One side abstains and the other commits: the adjacency is right,
            # the orientation claim is not comparable.
            unoriented += 1
            total += undirected_weight
        else:
            # Both directed, opposite ways.
            reversed_edges += 1
            total += 1.0

    return SHDResult(
        shd=round(total, 6),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        reversed_edges=reversed_edges,
        unoriented_edges=unoriented,
        undirected_weight=undirected_weight,
        n_predicted_edges=predicted.n_edges(),
        n_true_edges=truth.n_edges(),
        n_variables=n,
    )


def align_to(graph: CausalGraph, var_names: list[str]) -> CausalGraph:
    """Reorders and restricts `graph` onto `var_names`.

    Every scoring comparison runs through this, because two graphs built from
    different sources rarely arrive with their columns in the same order, and a
    mismatched order silently scores variable 3 against variable 7.
    """
    missing = set(var_names) - set(graph.var_names)
    if missing:
        raise ValueError(f"graph is missing variables {sorted(missing)}")
    position = {name: i for i, name in enumerate(graph.var_names)}
    keep = [position[name] for name in var_names]

    aligned = CausalGraph.empty(
        list(var_names), tau_max=graph.tau_max, meta=dict(graph.meta)
    )
    aligned.links = graph.links[keep][:, keep, :].copy()
    if graph.p_matrix is not None:
        aligned.p_matrix = graph.p_matrix[keep][:, keep, :].copy()
    if graph.val_matrix is not None:
        aligned.val_matrix = graph.val_matrix[keep][:, keep, :].copy()
    return aligned
