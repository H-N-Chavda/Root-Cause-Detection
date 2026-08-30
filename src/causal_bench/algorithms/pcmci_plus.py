"""PCMCI+ via tigramite.

PCMCI+ models the lag structure explicitly, so unlike PC it does not assume
i.i.d. samples: the autocorrelation that mis-specifies PC on this data is part
of what PCMCI+ estimates. It does assume stationarity, which Phase 2 found only
partly holds.

`tau_max` is not chosen here. It comes from the Phase 2 VAR order selection, and
because the criteria disagree (BIC low, AIC high) the runner sweeps the range
rather than committing to one.

The conditional independence test is `ParCorr` (partial correlation) for the
linear case, which Phase 2 supports: zero of 465 pairs showed distance
correlation exceeding |Pearson| by more than 0.10, and a boosted tree lost to a
linear fit out of sample. `GPDC` and `CMIknn` are available through config for
the nonlinear case, and are opt-in because both are orders of magnitude slower.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..graph.representation import CausalGraph
from .base import CausalDiscoveryAlgorithm

#: Config name -> tigramite independence test.
INDEPENDENCE_TESTS = {
    "parcorr": ("tigramite.independence_tests.parcorr", "ParCorr"),
    "gpdc": ("tigramite.independence_tests.gpdc", "GPDC"),
    "cmiknn": ("tigramite.independence_tests.cmiknn", "CMIknn"),
}


def build_independence_test(name: str, **kwargs: Any) -> Any:
    """Instantiates a tigramite CI test by config name."""
    key = name.lower()
    if key not in INDEPENDENCE_TESTS:
        raise ValueError(
            f"unknown independence test {name!r}; expected one of "
            f"{sorted(INDEPENDENCE_TESTS)}"
        )
    module_path, class_name = INDEPENDENCE_TESTS[key]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)(**kwargs)


class PCMCIPlusAlgorithm(CausalDiscoveryAlgorithm):
    """PCMCI+ on lagged data, returning a lagged graph."""

    name = "pcmci_plus"
    library = "tigramite"
    handles_lags = True
    supports_prior_knowledge = True

    def __init__(
        self,
        seed: int = 0,
        tau_max: int = 1,
        tau_min: int = 0,
        pc_alpha: float = 0.01,
        independence_test: str = "parcorr",
        independence_test_params: dict[str, Any] | None = None,
        contemp_collider_rule: str = "majority",
        conflict_resolution: bool = True,
        max_conds_dim: int | None = None,
        max_combinations: int = 1,
        max_conds_py: int | None = None,
        max_conds_px: int | None = None,
        fdr_method: str = "none",
        **params: Any,
    ) -> None:
        super().__init__(
            seed=seed,
            tau_max=tau_max,
            tau_min=tau_min,
            pc_alpha=pc_alpha,
            independence_test=independence_test,
            independence_test_params=independence_test_params or {},
            contemp_collider_rule=contemp_collider_rule,
            conflict_resolution=conflict_resolution,
            max_conds_dim=max_conds_dim,
            max_combinations=max_combinations,
            max_conds_py=max_conds_py,
            max_conds_px=max_conds_px,
            fdr_method=fdr_method,
            **params,
        )

    def check_assumptions(self, findings: dict[str, Any] | None) -> list[str]:
        if not findings:
            return []
        notes = []
        stationarity = findings["temporal"]["stationarity"]["summary"]
        regime = findings["temporal"]["regime"]
        if stationarity["n_adf_stationary"] < stationarity["n_scored"]:
            notes.append(
                f"stationarity: ADF calls only "
                f"{stationarity['n_adf_stationary']}/{stationarity['n_scored']} "
                f"columns stationary and KPSS only "
                f"{stationarity['n_kpss_stationary']}; the two disagree on "
                f"{stationarity['n_disagree']}"
            )
        shifted = max(
            regime["halves"]["n_columns_shifted"], regime["thirds"]["n_columns_shifted"]
        )
        if shifted:
            notes.append(
                f"stationarity: {shifted} column(s) shift their mean by more than "
                f"{regime['shift_threshold_sd']} SD between segments (max "
                f"{regime['thirds']['max_mean_shift_sd']:.2f} SD across thirds)"
            )
        collinearity = findings["conditioning"]["collinearity"]
        if (
            collinearity["ill_conditioned"]
            and self.params["independence_test"] == "parcorr"
        ):
            notes.append(
                f"well-conditioned covariance: condition number "
                f"{collinearity['condition_number']:.0f}; ParCorr inverts this "
                "matrix, so its p-values are unreliable"
            )
        return notes

    def _fit(
        self,
        data: np.ndarray,
        var_names: list[str],
        prior_knowledge: Any | None = None,
    ) -> CausalGraph:
        import tigramite.data_processing as pp
        from tigramite.pcmci import PCMCI

        test = build_independence_test(
            self.params["independence_test"],
            **self.params["independence_test_params"],
        )
        dataframe = pp.DataFrame(data, var_names=var_names)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=test, verbosity=0)

        results = pcmci.run_pcmciplus(
            link_assumptions=prior_knowledge,
            tau_min=self.params["tau_min"],
            tau_max=self.params["tau_max"],
            pc_alpha=self.params["pc_alpha"],
            contemp_collider_rule=self.params["contemp_collider_rule"],
            conflict_resolution=self.params["conflict_resolution"],
            max_conds_dim=self.params["max_conds_dim"],
            max_combinations=self.params["max_combinations"],
            max_conds_py=self.params["max_conds_py"],
            max_conds_px=self.params["max_conds_px"],
            fdr_method=self.params["fdr_method"],
        )

        # tigramite already returns the shared convention: graph[i, j, tau] is
        # the link from (i, t - tau) to (j, t). No transpose, no relabelling.
        return CausalGraph(
            links=np.asarray(results["graph"]),
            var_names=var_names,
            p_matrix=np.asarray(results["p_matrix"])
            if results.get("p_matrix") is not None
            else None,
            val_matrix=np.asarray(results["val_matrix"])
            if results.get("val_matrix") is not None
            else None,
            meta={
                "collider_rule": self.params["contemp_collider_rule"],
                "independence_test": self.params["independence_test"],
                "tau_max": self.params["tau_max"],
                "tau_min": self.params["tau_min"],
            },
        )
