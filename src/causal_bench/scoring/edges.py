"""Edge-level scoring of a discovered graph against a known edge list.

**Lags are collapsed.** The ground truth is a set of variable pairs with no lag
attached -- the ODE says `h3` drives `h1`, not at which lag -- so an estimate is
credited once for `h3 -> h1` however many lags it appears at. Scoring per lag
would punish an algorithm for correctly seeing that a slow effect persists.

**Self-loops are excluded**, matching the ground truth, which omits them because
every tank drains itself and the edge is true by construction for all seven
variables. Counting them would inflate every algorithm's score identically.

**Parsimony is reported, not folded in.** `lag_multiplicity` (raw lagged edges
per unique edge) and SHD both penalise over-reporting; an "adjusted F1" that
mixed them into one number would need an arbitrary weight, so the parts are kept
separate and the caller decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..graph.representation import CausalGraph


@dataclass(frozen=True)
class EdgeScore:
    """Detection scores for one graph against one ground-truth edge set."""

    n_true: int
    n_predicted: int
    true_positives: tuple[tuple[str, str], ...]
    reversed_edges: tuple[tuple[str, str], ...]
    false_positives: tuple[tuple[str, str], ...]
    false_negatives: tuple[tuple[str, str], ...]
    undirected_hits: tuple[tuple[str, str], ...]
    self_loops: tuple[str, ...]
    n_raw_lagged_edges: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return len(self.true_positives) / self.n_predicted if self.n_predicted else 0.0

    @property
    def recall(self) -> float:
        return len(self.true_positives) / self.n_true if self.n_true else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def lag_multiplicity(self) -> float:
        """Raw lagged edges per unique cross-variable edge.

        1.0 means every edge was reported at exactly one lag. Higher means the
        same relationship was repeated across lags -- not wrong, but it is why a
        raw edge count overstates how much an algorithm claims.
        """
        return self.n_raw_lagged_edges / self.n_predicted if self.n_predicted else 0.0

    @property
    def shd(self) -> int:
        """Skeleton-level SHD: false positives + false negatives + reversals.

        This is the count-penalising counterpart to F1 -- every extra edge costs
        1, so an algorithm cannot buy recall with indiscriminate edges.
        """
        return (
            len(self.false_positives)
            + len(self.false_negatives)
            + len(self.reversed_edges)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_true": self.n_true,
            "n_predicted": self.n_predicted,
            "n_raw_lagged_edges": self.n_raw_lagged_edges,
            "n_self_loops": len(self.self_loops),
            "true_positives": [list(e) for e in self.true_positives],
            "reversed_edges": [list(e) for e in self.reversed_edges],
            "false_positives": [list(e) for e in self.false_positives],
            "false_negatives": [list(e) for e in self.false_negatives],
            "undirected_hits": [list(e) for e in self.undirected_hits],
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "shd": self.shd,
            "lag_multiplicity": round(self.lag_multiplicity, 2),
            **self.meta,
        }


def score_edges(
    graph: CausalGraph,
    ground_truth: list[tuple[str, str]] | list[list[str]],
) -> EdgeScore:
    """Scores `graph` against a directed ground-truth edge list.

    An undirected estimate of a true pair counts as a *detection* but not as a
    true positive: the pair was found, the direction was not resolved. It is
    therefore kept out of `false_negatives` and out of `true_positives` both,
    and reported in its own field so neither precision nor recall silently
    absorbs an orientation the algorithm never committed to.
    """
    truth = {(c, e) for c, e in ground_truth}
    truth_pairs = {frozenset(p) for p in truth}

    directed = graph.directed_edges()
    undirected = graph.undirected_edges()

    self_loops = tuple(sorted({c for c, e, _ in directed if c == e}))
    predicted = {(c, e) for c, e, _ in directed if c != e}
    und_pairs = {frozenset((a, b)) for a, b, _ in undirected if a != b}

    true_positives = predicted & truth
    reversed_edges = {
        (c, e) for (c, e) in predicted if (e, c) in truth and (c, e) not in truth
    }
    undirected_hits = und_pairs & truth_pairs

    # A claimed adjacency whose unordered pair is absent from the truth is a
    # false positive whether or not the algorithm committed to a direction.
    # Counting only the directed ones would let an algorithm assert a spurious
    # link for free by leaving it unoriented.
    directed_fp = predicted - truth - reversed_edges
    undirected_fp = und_pairs - truth_pairs
    false_positives = directed_fp | {tuple(sorted(p)) for p in undirected_fp}

    found = (
        {frozenset(p) for p in true_positives}
        | {frozenset(p) for p in reversed_edges}
        | undirected_hits
    )
    false_negatives = {
        (c, e) for (c, e) in truth if frozenset((c, e)) not in found
    }

    def _srt(pairs: set) -> tuple:
        return tuple(
            sorted(
                tuple(sorted(p)) if isinstance(p, frozenset) else p for p in pairs
            )
        )

    claimed = {frozenset(p) for p in predicted} | und_pairs

    return EdgeScore(
        n_true=len(truth),
        n_predicted=len(claimed),
        true_positives=_srt(true_positives),
        reversed_edges=_srt(reversed_edges),
        false_positives=_srt(false_positives),
        false_negatives=_srt(false_negatives),
        undirected_hits=_srt(undirected_hits),
        self_loops=self_loops,
        n_raw_lagged_edges=len(directed) + len(undirected),
        meta={"tau_max": graph.tau_max},
    )
