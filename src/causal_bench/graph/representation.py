"""The one graph object every algorithm returns.

The layout follows tigramite's convention, because of the four libraries in play
it is the only one that already expresses lags and edge marks together, so
adopting it means three converters instead of four.

**Convention, verified against tigramite 5.2.10.1 on a synthetic system with a
known one-way edge:**

    links[i, j, tau]  describes the link from (variable i, time t - tau)
                      to (variable j, time t)

so ``links[i, j, tau] == "-->"`` means *i causes j at lag tau*. At tau = 0 the
array is stored redundantly: an oriented contemporaneous edge appears as ``-->``
at ``[i, j, 0]`` and ``<--`` at ``[j, i, 0]``. For tau > 0 only the forward
entry is filled, because time does not run backwards.

Edge marks:

===========  ==========================================================
``""``       no edge
``"-->"``    i causes j
``"<--"``    j causes i (the mirror entry of a ``-->``)
``"o-o"``    adjacent, direction not determined (a CPDAG abstention)
``"x-x"``    conflicting orientation, the algorithm could not decide
``"<->"``    both endpoints have arrowheads, i.e. a latent common cause
===========  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Every mark the array is allowed to hold.
EDGE_MARKS = ("", "-->", "<--", "o-o", "x-x", "<->")

#: Marks that assert a definite direction from i to j.
DIRECTED_MARKS = ("-->",)

#: Marks that assert adjacency without a settled direction. An undirected edge
#: from a CPDAG is an abstention, not an error, and the scorer treats it so.
UNDIRECTED_MARKS = ("o-o", "x-x", "<->")

#: numpy string width: the widest mark is three characters.
_MARK_DTYPE = "<U3"


@dataclass
class CausalGraph:
    """A (possibly lagged) causal graph plus the provenance of the run.

    `p_matrix` and `val_matrix` are optional because only the constraint-based
    methods produce them; a score-based method such as VAR-LiNGAM has
    coefficients instead, which go in `val_matrix`.
    """

    links: np.ndarray
    var_names: list[str]
    p_matrix: np.ndarray | None = None
    val_matrix: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.links = np.asarray(self.links, dtype=_MARK_DTYPE)
        if self.links.ndim != 3:
            raise ValueError(f"links must be 3-dimensional, got {self.links.shape}")
        n_vars = len(self.var_names)
        if self.links.shape[:2] != (n_vars, n_vars):
            raise ValueError(
                f"links has shape {self.links.shape} but there are {n_vars} variable names"
            )
        unknown = set(np.unique(self.links)) - set(EDGE_MARKS)
        if unknown:
            raise ValueError(f"unknown edge marks: {sorted(unknown)}")

    # -- shape ------------------------------------------------------------

    @property
    def n_vars(self) -> int:
        return len(self.var_names)

    @property
    def tau_max(self) -> int:
        return int(self.links.shape[2] - 1)

    @classmethod
    def empty(cls, var_names: list[str], tau_max: int = 0, **kwargs: Any) -> CausalGraph:
        n = len(var_names)
        return cls(
            links=np.full((n, n, tau_max + 1), "", dtype=_MARK_DTYPE),
            var_names=list(var_names),
            **kwargs,
        )

    # -- reading ----------------------------------------------------------

    def directed_edges(self) -> list[tuple[str, str, int]]:
        """Every `(cause, effect, lag)` the graph asserts a direction for."""
        return [
            (self.var_names[i], self.var_names[j], int(tau))
            for i, j, tau in zip(
                *np.where(np.isin(self.links, DIRECTED_MARKS)), strict=True
            )
        ]

    def undirected_edges(self) -> list[tuple[str, str, int]]:
        """Adjacencies with no settled direction, each listed once (i < j)."""
        found = {
            (min(i, j), max(i, j), int(tau))
            for i, j, tau in zip(
                *np.where(np.isin(self.links, UNDIRECTED_MARKS)), strict=True
            )
        }
        return [(self.var_names[i], self.var_names[j], tau) for i, j, tau in sorted(found)]

    def n_edges(self) -> int:
        """Adjacency count: an undirected edge counts once, not twice.

        Directed lag-0 edges are stored twice (`-->` and its `<--` mirror), so
        counting raw non-empty cells would double every contemporaneous edge.
        """
        return len(self.directed_edges()) + len(self.undirected_edges())

    def adjacency_pairs(self) -> set[tuple[str, str]]:
        """Unordered `{a, b}` pairs adjacent at any lag, as sorted tuples.

        Self-loops (a variable's own past) are excluded: the ground truth has
        no self-loops and a lagged self-dependence is autocorrelation, not a
        causal claim between two variables.
        """
        pairs = set()
        for a, b, _ in self.directed_edges() + self.undirected_edges():
            if a != b:
                pairs.add((min(a, b), max(a, b)))
        return pairs

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "var_names": list(self.var_names),
            "tau_max": self.tau_max,
            "links": self.links.tolist(),
            "p_matrix": None if self.p_matrix is None else self.p_matrix.tolist(),
            "val_matrix": None if self.val_matrix is None else self.val_matrix.tolist(),
            "meta": self.meta,
            "directed_edges": [
                {"cause": c, "effect": e, "lag": t} for c, e, t in self.directed_edges()
            ],
            "undirected_edges": [
                {"a": a, "b": b, "lag": t} for a, b, t in self.undirected_edges()
            ],
            "n_edges": self.n_edges(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CausalGraph:
        def _array(key: str) -> np.ndarray | None:
            value = payload.get(key)
            return None if value is None else np.asarray(value, dtype=float)

        return cls(
            links=np.asarray(payload["links"], dtype=_MARK_DTYPE),
            var_names=list(payload["var_names"]),
            p_matrix=_array("p_matrix"),
            val_matrix=_array("val_matrix"),
            meta=dict(payload.get("meta", {})),
        )

    def __repr__(self) -> str:
        return (
            f"CausalGraph({self.n_vars} vars, tau_max={self.tau_max}, "
            f"{self.n_edges()} edges, algorithm={self.meta.get('algorithm', '?')})"
        )


@dataclass
class RunMetadata:
    """Provenance for one algorithm run.

    `assumptions_met` is deliberately three-valued. False is a recorded
    violation, not a crash: an algorithm that ran but was mis-specified still
    produces a graph, and hiding that would be worse than reporting it.
    """

    algorithm: str
    library: str
    library_version: str
    parameters: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    runtime_seconds: float | None = None
    assumptions_met: bool | None = None
    assumption_notes: list[str] = field(default_factory=list)
    completed: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "library": self.library,
            "library_version": self.library_version,
            "parameters": self.parameters,
            "seed": self.seed,
            "runtime_seconds": self.runtime_seconds,
            "assumptions_met": self.assumptions_met,
            "assumption_notes": self.assumption_notes,
            "completed": self.completed,
            "error": self.error,
        }
