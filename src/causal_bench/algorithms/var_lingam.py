"""VAR-LiNGAM via the `lingam` package.

**Orientation convention -- the single most dangerous detail in this module.**
`lingam` returns coefficient matrices ``B_tau`` in which

    B_tau[row, col]  is the coefficient of  variable[col] at time t - tau
                     in the equation for  variable[row] at time t

so **row = effect, column = cause**. That is the transpose of the usual
adjacency reading, and of the shared graph object's
``links[i, j, tau] == "-->"`` meaning *i causes j*. The converter below
therefore reads ``B_tau[j, i]`` to decide the mark at ``[i, j, tau]``.

Verified empirically, not from memory: on a synthetic system with
``x1 = 0.8 * x0`` contemporaneously and ``x2(t) = 0.9 * x0(t-1)``, the fitted
matrices give ``B0[1, 0] = 0.78`` and ``B1[2, 0] = 0.91``, with the transposed
entries at zero. Getting this backwards transposes every graph and every SHD
still looks plausible.

**Identifiability.** LiNGAM recovers the contemporaneous order from
*non-Gaussian* innovations; ICA has no traction on Gaussian sources, because a
rotation of independent Gaussians is again independent Gaussians. Phase 2
measured the VAR(1) residuals at mean |excess kurtosis| 0.141 with zero of 31
equations clearing the threshold, so the lag-0 part is expected to carry no
information here. The bootstrap over lag-0 orientations demonstrates that
rather than asserting it, and the lagged and contemporaneous parts are reported
separately because only the first is trustworthy.

**Prior knowledge limit.** `DirectLiNGAM`'s `prior_knowledge` matrix constrains
the contemporaneous ``B0`` only. It cannot constrain a lagged edge: the lagged
coefficients come from an OLS/VAR pre-fit that the inner LiNGAM model never
sees.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..graph.representation import CausalGraph
from ..utils.logging import get_logger
from .base import CausalDiscoveryAlgorithm

log = get_logger(__name__)

#: |excess kurtosis| an innovation must exceed for ICA to have traction. Matches
#: the Phase 2 `eda.nongaussianity.abs_excess_kurtosis_threshold` default.
_KURTOSIS_LIMIT = 1.0


def adjacency_matrices_to_graph(matrices: np.ndarray, var_names: list[str]) -> CausalGraph:
    """`lingam`'s ``adjacency_matrices_`` into the shared graph object.

    `matrices` has shape ``(lags + 1, N, N)`` with `matrices[tau][effect, cause]`,
    so the conversion is a transpose per lag. A non-zero coefficient is an edge;
    `lingam`'s own pruning has already zeroed the ones it rejected.
    """
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(
            f"expected (lags+1, N, N) coefficient matrices, got {matrices.shape}"
        )
    n_lags = matrices.shape[0] - 1
    n = matrices.shape[1]
    if n != len(var_names):
        raise ValueError(f"{n} variables in the matrices, {len(var_names)} names")

    graph = CausalGraph.empty(list(var_names), tau_max=n_lags)
    values = np.zeros((n, n, n_lags + 1), dtype=float)

    for tau in range(n_lags + 1):
        b = matrices[tau]
        for i in range(n):
            for j in range(n):
                if tau == 0 and i == j:
                    continue  # LiNGAM's B0 has a zero diagonal by construction
                # b[j, i]: row j is the EFFECT, column i is the CAUSE.
                coefficient = float(b[j, i])
                values[i, j, tau] = coefficient
                if coefficient != 0.0:
                    graph.links[i, j, tau] = "-->"
                    if tau == 0:
                        graph.links[j, i, 0] = "<--"

    graph.val_matrix = values
    return graph


class VARLiNGAMAlgorithm(CausalDiscoveryAlgorithm):
    """VAR-LiNGAM: a VAR pre-fit plus DirectLiNGAM on the residuals."""

    name = "var_lingam"
    library = "lingam"
    handles_lags = True
    supports_prior_knowledge = True

    def __init__(
        self,
        seed: int = 0,
        tau_max: int = 1,
        criterion: str | None = "bic",
        prune: bool = True,
        measure: str = "pwling",
        bootstrap_samples: int = 100,
        run_bootstrap: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(
            seed=seed,
            tau_max=tau_max,
            criterion=criterion,
            prune=prune,
            measure=measure,
            bootstrap_samples=bootstrap_samples,
            run_bootstrap=run_bootstrap,
            **params,
        )

    def check_assumptions(self, findings: dict[str, Any] | None) -> list[str]:
        if not findings:
            return []
        notes = []
        residuals = findings["distribution"]["var_residuals"]
        if residuals.get("fitted") and not residuals["lingam_contemporaneous_identifiable"]:
            notes.append(
                f"non-Gaussian residuals: only {residuals['n_usable_for_ica']}/"
                f"{residuals['n_equations']} VAR residual equations clear the "
                f"non-Gaussianity thresholds (mean |excess kurtosis| "
                f"{residuals['mean_abs_excess_kurtosis']:.3f}, max negentropy "
                f"{residuals['max_negentropy']:.5f} nats). ICA cannot separate "
                "Gaussian sources, so the contemporaneous B0 is not identified"
            )
        stationarity = findings["temporal"]["stationarity"]["summary"]
        if stationarity["n_adf_stationary"] < stationarity["n_scored"]:
            notes.append(
                f"stationarity: ADF calls only "
                f"{stationarity['n_adf_stationary']}/{stationarity['n_scored']} "
                "columns stationary"
            )
        return notes

    def _fit(
        self,
        data: np.ndarray,
        var_names: list[str],
        prior_knowledge: Any | None = None,
    ) -> CausalGraph:
        import lingam

        inner = lingam.DirectLiNGAM(
            random_state=self.seed,
            prior_knowledge=prior_knowledge,
            measure=self.params["measure"],
        )
        model = lingam.VARLiNGAM(
            lags=self.params["tau_max"],
            criterion=self.params["criterion"],
            prune=self.params["prune"],
            lingam_model=inner,
            random_state=self.seed,
        )
        model.fit(data)

        graph = adjacency_matrices_to_graph(model.adjacency_matrices_, var_names)
        meta: dict[str, Any] = {
            "lags_requested": self.params["tau_max"],
            "lags_used": int(np.asarray(model.adjacency_matrices_).shape[0] - 1),
            "criterion": self.params["criterion"],
            "prior_knowledge_scope": (
                "contemporaneous B0 only; lagged coefficients come from the VAR "
                "pre-fit, which the inner DirectLiNGAM never sees"
            ),
            "causal_order": [int(k) for k in getattr(model, "causal_order_", [])],
        }
        meta.update(self._split_lagged_and_contemporaneous(graph))
        meta.update(self._residual_independence(model, var_names))
        if self.params["run_bootstrap"]:
            meta["lag0_bootstrap"] = self._bootstrap_lag0(model, data, var_names)
        graph.meta = meta
        return graph

    @staticmethod
    def _split_lagged_and_contemporaneous(graph: CausalGraph) -> dict[str, Any]:
        """Counts the two parts separately, because only the lagged one is
        trustworthy when the innovations are Gaussian."""
        contemporaneous = [e for e in graph.directed_edges() if e[2] == 0]
        lagged = [e for e in graph.directed_edges() if e[2] > 0]
        return {
            "n_contemporaneous_edges": len(contemporaneous),
            "n_lagged_edges": len(lagged),
            "contemporaneous_edges": [
                {"cause": c, "effect": e} for c, e, _ in contemporaneous
            ],
        }

    def _residual_independence(self, model: Any, var_names: list[str]) -> dict[str, Any]:
        """`get_error_independence_p_values` -- a second read on the model.

        LiNGAM assumes the error terms are mutually independent. Rejections here
        mean the fitted model does not explain the dependence structure, which is
        independent evidence from the non-Gaussianity check.
        """
        try:
            p_values = np.asarray(model.get_error_independence_p_values())
        except Exception as exc:
            log.warning("error-independence p-values unavailable: %s", exc)
            return {"error_independence": {"available": False, "error": str(exc)}}

        n = p_values.shape[0]
        upper = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rejected = [(i, j) for i, j in upper if p_values[i, j] < 0.05]
        return {
            "error_independence": {
                "available": True,
                "n_pairs": len(upper),
                "n_rejected_at_0.05": len(rejected),
                "fraction_rejected": len(rejected) / len(upper) if upper else 0.0,
                "min_p_value": float(np.min([p_values[i, j] for i, j in upper]))
                if upper
                else None,
                "note": "LiNGAM assumes mutually independent errors; rejections "
                "mean the fitted model leaves dependence unexplained",
            }
        }

    def _bootstrap_lag0(
        self, model: Any, data: np.ndarray, var_names: list[str]
    ) -> dict[str, Any]:
        """Resamples to show how stable the contemporaneous orientations are.

        Uses `VARLiNGAM.bootstrap`, which the library provides, rather than a
        hand-rolled resampler. The quantity that matters is the *disagreement*:
        if the innovations are Gaussian the contemporaneous order is not
        identified, so each resample can return a different order and the
        per-edge selection frequency collapses toward chance. That is a
        demonstration of non-identifiability rather than an assertion of it.
        """
        n_samples = self.params["bootstrap_samples"]
        try:
            result = model.bootstrap(data, n_sampling=n_samples)
            matrices = np.asarray(result.adjacency_matrices_)
        except Exception as exc:
            log.warning("VAR-LiNGAM bootstrap failed: %s", exc)
            return {"available": False, "error": str(exc), "n_samples": n_samples}

        # Verified against lingam 1.13.0: `bootstrap` returns
        # (n_sampling, N, N * (lags + 1)) -- the per-lag matrices concatenated
        # along the columns, NOT the (n_sampling, lags + 1, N, N) stack that
        # `adjacency_matrices_` uses on the fitted model. The lag-0 block is the
        # first N columns.
        n = len(var_names)
        if matrices.ndim != 3 or matrices.shape[1] != n:
            log.warning("unexpected bootstrap shape %s for %d variables", matrices.shape, n)
            return {
                "available": False,
                "error": f"unexpected bootstrap shape {matrices.shape}",
                "n_samples": n_samples,
            }
        b0 = matrices[:, :, :n]
        present = b0 != 0

        edges: list[dict[str, Any]] = []
        for effect in range(n):
            for cause in range(n):
                if effect == cause:
                    continue
                forward = float(present[:, effect, cause].mean())
                if forward > 0:
                    edges.append(
                        {
                            "cause": var_names[cause],
                            "effect": var_names[effect],
                            "selection_frequency": forward,
                        }
                    )
        edges.sort(key=lambda e: float(e["selection_frequency"]), reverse=True)

        # For each unordered pair that ever appears, how often the direction
        # flips between resamples. A value near 0.5 means the data does not
        # determine the direction at all.
        flips = []
        for i in range(n):
            for j in range(i + 1, n):
                a = present[:, j, i].mean()  # i -> j
                b = present[:, i, j].mean()  # j -> i
                if a + b > 0:
                    flips.append(min(a, b) / (a + b))

        stable = [e for e in edges if float(e["selection_frequency"]) >= 0.5]
        return {
            "available": True,
            "n_samples": n_samples,
            "n_lag0_edges_in_point_estimate": int(
                (np.asarray(model.adjacency_matrices_)[0] != 0).sum()
            ),
            "n_distinct_lag0_edges_seen": len(edges),
            "n_edges_selected_at_least_half_the_time": len(stable),
            "max_selection_frequency": edges[0]["selection_frequency"] if edges else 0.0,
            "median_selection_frequency": float(
                np.median([float(e["selection_frequency"]) for e in edges])
            )
            if edges
            else None,
            "mean_direction_flip_rate": float(np.mean(flips)) if flips else None,
            "top_edges": edges[:15],
            "interpretation": (
                "A selection frequency near 1.0 means a resample-stable edge. "
                "A direction flip rate near 0.5 means the two orientations are "
                "chosen about equally often, i.e. the contemporaneous direction "
                "is not identified from this data."
            ),
        }
