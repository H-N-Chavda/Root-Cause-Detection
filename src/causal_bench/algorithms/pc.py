"""PC via tigramite's non-time-series entry point.

`PCMCI.run_pcalg_non_timeseries_data` is standard PC on i.i.d. data: it calls
`run_pcalg` with tau_min = tau_max = 0 and strips the lag dimension. Two of its
defaults matter enough to record in the report:

* the skeleton search is **PC-stable**, so it is already order independent;
* `contemp_collider_rule` defaults to ``'majority'``, i.e. Majority-Rule PC,
  which resolves the separating-set order dependence that plain PC leaves open.
  This changes the orientations relative to the plain rule, so which rule ran is
  recorded in the metadata.

**PC is mis-specified on this data.** It assumes i.i.d. samples, and Phase 2
measured lag-1 autocorrelation up to 0.98 with a binding effective sample size
of 36 out of 1499 rows. It is run anyway, as the baseline that shows why the
temporal methods exist, and the violation is recorded rather than hidden.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..graph.representation import CausalGraph
from .base import CausalDiscoveryAlgorithm

#: Above this lag-1 autocorrelation the i.i.d. assumption is not defensible.
_AUTOCORRELATION_LIMIT = 0.3


class PCAlgorithm(CausalDiscoveryAlgorithm):
    """Standard PC (PC-stable + majority collider rule) on non-time-series data."""

    name = "pc"
    library = "tigramite"
    handles_lags = False
    supports_prior_knowledge = True

    def __init__(
        self,
        seed: int = 0,
        pc_alpha: float = 0.01,
        max_conds_dim: int | None = None,
        max_combinations: int | None = None,
        contemp_collider_rule: str = "majority",
        conflict_resolution: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(
            seed=seed,
            pc_alpha=pc_alpha,
            max_conds_dim=max_conds_dim,
            max_combinations=max_combinations,
            contemp_collider_rule=contemp_collider_rule,
            conflict_resolution=conflict_resolution,
            **params,
        )

    def check_assumptions(self, findings: dict[str, Any] | None) -> list[str]:
        if not findings:
            return []
        notes = []
        serial = findings["temporal"]["serial_dependence"]["summary"]
        ess = findings["conditioning"]["effective_sample_size"]
        if serial["max_abs_acf_lag_1"] > _AUTOCORRELATION_LIMIT:
            notes.append(
                f"independent samples: max lag-1 autocorrelation "
                f"{serial['max_abs_acf_lag_1']:.3f} exceeds {_AUTOCORRELATION_LIMIT}; "
                f"binding effective n is {ess['binding_n_eff']:.0f} of "
                f"{ess['n_rows']} rows, so nominal p-values are anti-conservative"
            )
        collinearity = findings["conditioning"]["collinearity"]
        if collinearity["ill_conditioned"]:
            notes.append(
                f"well-conditioned covariance: condition number "
                f"{collinearity['condition_number']:.0f} exceeds "
                f"{collinearity['condition_number_threshold']:.0f}, so the "
                "partial correlations the Fisher-z test inverts amplify noise"
            )
        return notes

    def _fit(
        self,
        data: np.ndarray,
        var_names: list[str],
        prior_knowledge: Any | None = None,
    ) -> CausalGraph:
        import tigramite.data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI

        dataframe = pp.DataFrame(data, var_names=var_names)
        # ParCorr is the partial-correlation (Fisher-z) test.
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)

        if prior_knowledge is None:
            results = pcmci.run_pcalg_non_timeseries_data(
                pc_alpha=self.params["pc_alpha"],
                max_conds_dim=self.params["max_conds_dim"],
                max_combinations=self.params["max_combinations"],
                contemp_collider_rule=self.params["contemp_collider_rule"],
                conflict_resolution=self.params["conflict_resolution"],
            )
        else:
            # `run_pcalg_non_timeseries_data` takes no `link_assumptions`
            # argument in tigramite 5.2.10.1 -- verified against the installed
            # signature. Its own body is a call to `run_pcalg` with
            # tau_min = tau_max = 0, so calling that directly is the same code
            # path with prior knowledge added, not a different algorithm.
            results = pcmci.run_pcalg(
                link_assumptions=prior_knowledge,
                pc_alpha=self.params["pc_alpha"],
                tau_min=0,
                tau_max=0,
                max_conds_dim=self.params["max_conds_dim"],
                max_combinations=self.params["max_combinations"],
                mode="standard",
                contemp_collider_rule=self.params["contemp_collider_rule"],
                conflict_resolution=self.params["conflict_resolution"],
            )

        graph = CausalGraph(
            links=np.asarray(results["graph"]),
            var_names=var_names,
            p_matrix=np.asarray(results.get("p_matrix"))
            if results.get("p_matrix") is not None
            else None,
            val_matrix=np.asarray(results.get("val_matrix"))
            if results.get("val_matrix") is not None
            else None,
            meta={
                "collider_rule": self.params["contemp_collider_rule"],
                "skeleton": "PC-stable (order independent)",
                "n_ambiguous_triples": len(results.get("ambiguous_triples", [])),
                "entry_point": (
                    "run_pcalg_non_timeseries_data"
                    if prior_knowledge is None
                    else "run_pcalg(tau_min=0, tau_max=0)"
                ),
            },
        )
        return graph
