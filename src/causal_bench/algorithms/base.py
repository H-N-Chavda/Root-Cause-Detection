"""The interface every algorithm implements.

One contract: take a numeric array plus variable names, return a `CausalGraph`.
The base class handles the parts that must not vary between algorithms -- seeding,
timing, standardisation, and turning an exception into a recorded failure rather
than a crash -- so a subclass only writes `_fit`.

Failure is data. If an algorithm cannot run under its own assumptions, the run
records that and continues; a missing result is a finding, a fabricated one is
not. `assumptions_met=False` marks a run that *completed* but was
mis-specified, which is a different thing from `completed=False`.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..graph.representation import CausalGraph, RunMetadata
from ..utils.logging import get_logger

log = get_logger(__name__)


class CausalDiscoveryAlgorithm(ABC):
    """Base class for a causal-discovery algorithm wrapper."""

    #: Short registry key, e.g. "pcmci_plus".
    name: str = "unnamed"
    #: Which library provides it, for the report.
    library: str = "unknown"
    #: True when the method models lags explicitly.
    handles_lags: bool = False
    #: True when the method accepts prior knowledge.
    supports_prior_knowledge: bool = False
    #: True when running at different tau_max values gives genuinely different
    #: results. PCMCI+ conditions on the whole lag window, so tau matters; LSTE
    #: tests each lag independently, so a tau-3 run already contains the tau-1
    #: and tau-2 answers and sweeping would only triple the cost.
    tau_sweep_is_informative: bool = True

    def __init__(self, seed: int = 0, **params: Any) -> None:
        self.seed = seed
        self.params = params

    # -- to implement -----------------------------------------------------

    @abstractmethod
    def _fit(
        self,
        data: np.ndarray,
        var_names: list[str],
        prior_knowledge: Any | None = None,
    ) -> CausalGraph:
        """Run the algorithm and convert its output into a `CausalGraph`."""

    def library_version(self) -> str:
        import importlib.metadata as md

        try:
            return md.version(self.library)
        except Exception:
            return "unknown"

    def check_assumptions(self, findings: dict[str, Any] | None) -> list[str]:
        """Notes on whether this method's assumptions hold, from the EDA output.

        Returns a list of violations; empty means nothing measured argues
        against it. Subclasses override to name the assumptions they care about.
        """
        return []

    # -- the fixed part ---------------------------------------------------

    def run(
        self,
        data: np.ndarray,
        var_names: Sequence[str],
        prior_knowledge: Any | None = None,
        standardize: bool = False,
        eda_findings: dict[str, Any] | None = None,
    ) -> CausalGraph:
        """Seed, time, run, and attach metadata. Never raises for an algorithm
        failure -- the failure is returned as an empty graph whose metadata says
        what went wrong."""
        var_names = list(var_names)
        data = np.asarray(data, dtype=float)

        # One seed for the whole run. numpy's global state covers libraries that
        # reach for it implicitly (tigramite's shuffle tests do); estimators
        # taking an explicit random_state get `self.seed` at their call sites.
        np.random.seed(self.seed)

        if standardize:
            # Column magnitudes span ~2500x on this data, and a kNN estimator
            # measures distances in the raw units. Correlation-based methods are
            # scale invariant so this is a no-op for them, but applying it
            # uniformly keeps every algorithm looking at the same numbers.
            data = _standardize(data)

        violations = self.check_assumptions(eda_findings)
        started = time.time()
        try:
            graph = self._fit(data, var_names, prior_knowledge)
            completed, error = True, None
        except Exception as exc:  # noqa: BLE001 - a failure is a recorded result
            log.error("%s failed: %s", self.name, exc)
            log.debug("%s traceback:\n%s", self.name, traceback.format_exc())
            graph = CausalGraph.empty(var_names, tau_max=0)
            completed, error = False, f"{type(exc).__name__}: {exc}"
        runtime = time.time() - started

        graph.meta = {
            **graph.meta,
            **RunMetadata(
                algorithm=self.name,
                library=self.library,
                library_version=self.library_version(),
                parameters={**self.params, "standardized": standardize},
                seed=self.seed,
                runtime_seconds=round(runtime, 3),
                assumptions_met=(len(violations) == 0) if completed else None,
                assumption_notes=violations,
                completed=completed,
                error=error,
            ).to_dict(),
            "with_prior_knowledge": prior_knowledge is not None,
        }
        log.info(
            "%s: %s in %.1fs (%d edges, assumptions_met=%s)",
            self.name,
            "completed" if completed else "FAILED",
            runtime,
            graph.n_edges(),
            graph.meta["assumptions_met"],
        )
        return graph


def _standardize(data: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per column.

    A zero-variance column would divide by zero; the loader already drops those,
    but the guard keeps this usable on an arbitrary array.
    """
    std = data.std(axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)
    return (data - data.mean(axis=0)) / std
