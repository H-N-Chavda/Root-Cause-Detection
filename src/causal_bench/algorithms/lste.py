"""Lag-specific transfer entropy (LSTE), built on tigramite's CMIknn.

Transfer entropy from X to Y at lag tau is a conditional mutual information::

    TE_{X->Y}(tau) = I( Y_t ; X_{t-tau} | Y_{t-1}, ..., Y_{t-k} )

which is exactly what a conditional-independence test estimates. There is no
maintained single-call Python implementation of the lag-specific form, so this
module is glue code; the estimator and the significance machinery are the
library's.

**Why this option, over IDTxl and JIDT.**

* *IDTxl* is the closest feature match -- multivariate TE with built-in lag
  selection -- but it is not on PyPI, so it would pin the build to a git
  revision, and the task allows exactly one new package for LSTE. It also
  duplicates significance machinery this project already has.
* *JIDT via jpype1* is the most mature toolkit, and the reason to reject it is
  operational rather than statistical: it puts a Java runtime in the middle of a
  reproducible Python build, which is a large recurring cost for one algorithm.
* *tigramite's CMIknn* adds **no** dependency at all -- tigramite is already
  here for PC and PCMCI+ -- and reuses the shuffle-test significance the rest of
  the project already relies on. That is what this module uses.

**Why the kNN estimator and not the Gaussian one.** Under a Gaussian estimator,
transfer entropy is numerically equivalent to linear Granger causality, so a
Gaussian LSTE on this data would largely reproduce the VAR result and add
nothing to the comparison. Phase 2 found the marginals close to Gaussian (max
|excess kurtosis| 0.359), which makes that equivalence bite hard here. The
Kraskov-style k-nearest-neighbour estimator in `CMIknn` is what keeps LSTE a
distinct method: it is non-parametric in both the functional form and the noise
distribution, so it can see dependence that a partial correlation cannot. The
cost is speed -- a shuffle test runs in seconds, not microseconds -- which is
why the sweep parameters are all in config.

**Multiple testing.** Unlike PC and PCMCI+, LSTE tests every ordered pair at
every lag with no sequential pruning, so nothing else controls its false
positive count: at 31 variables, 3 lags and alpha 0.05 that is 2790 tests and
~140 expected false positives under the null. Benjamini-Hochberg is therefore
applied by default, and the uncorrected count is reported alongside it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..graph.representation import CausalGraph
from ..utils.logging import get_logger
from .base import CausalDiscoveryAlgorithm

log = get_logger(__name__)


def build_embedding(
    data: np.ndarray, source: int, target: int, lag: int, history: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assembles the (array, xyz) pair CMIknn wants for one TE test.

    Returns rows stacked as ``[X, Y, Z...]`` with the matching ``xyz`` codes
    (0 = X, 1 = Y, 2 = conditioning set), which is tigramite's internal calling
    convention for `get_dependence_measure`.

    ``X`` is the source at ``t - lag``, ``Y`` is the target at ``t``, and ``Z``
    is the target's own past over ``t-1 .. t-history``. Conditioning on the
    target's history is what makes this transfer entropy rather than plain
    lagged mutual information: without it, a target that is merely
    autocorrelated would show "information transfer" from anything correlated
    with its own past.
    """
    if lag < 1:
        raise ValueError(f"transfer entropy needs lag >= 1, got {lag}")
    if history < 1:
        raise ValueError(f"history must be >= 1, got {history}")

    offset = max(lag, history)
    n_rows = data.shape[0]
    y = data[offset:, target]
    x = data[offset - lag : n_rows - lag, source]
    z = [data[offset - h : n_rows - h, target] for h in range(1, history + 1)]

    array = np.vstack([x, y, *z])
    xyz = np.array([0, 1] + [2] * len(z))
    return array, xyz


class LSTEAlgorithm(CausalDiscoveryAlgorithm):
    """Lag-specific transfer entropy over every ordered pair and lag."""

    name = "lste"
    library = "tigramite"
    handles_lags = True
    supports_prior_knowledge = True
    # Each (source, target, lag) is tested on its own, so the tau-3 run already
    # contains the tau-1 and tau-2 results. Sweeping would triple a run that
    # costs ~1.5 s per test x 2790 tests.
    tau_sweep_is_informative = False

    def __init__(
        self,
        seed: int = 0,
        tau_max: int = 3,
        tau_min: int = 1,
        alpha: float = 0.05,
        history: int | None = None,
        knn: float = 0.1,
        sig_samples: int = 100,
        workers: int = -1,
        fdr_method: str = "fdr_bh",
        **params: Any,
    ) -> None:
        super().__init__(
            seed=seed,
            tau_max=tau_max,
            tau_min=tau_min,
            alpha=alpha,
            # Default the conditioning history to tau_max: conditioning on less
            # of the target's past than the lags being tested would leave the
            # autocorrelation the test is meant to control for.
            history=history if history is not None else tau_max,
            knn=knn,
            sig_samples=sig_samples,
            workers=workers,
            fdr_method=fdr_method,
            **params,
        )

    def check_assumptions(self, findings: dict[str, Any] | None) -> list[str]:
        if not findings:
            return []
        notes = []
        stationarity = findings["temporal"]["stationarity"]["summary"]
        if stationarity["n_adf_stationary"] < stationarity["n_scored"]:
            notes.append(
                f"stationarity: ADF calls only "
                f"{stationarity['n_adf_stationary']}/{stationarity['n_scored']} "
                "columns stationary; a kNN density estimate over a drifting "
                "series mixes two regimes"
            )
        ess = findings["conditioning"]["effective_sample_size"]
        # A kNN estimator needs more data than a parametric one, not less, so
        # the effective sample size binds harder here than for ParCorr.
        if ess["binding_n_eff"] < 100:
            notes.append(
                f"adequate sample size: binding effective n is "
                f"{ess['binding_n_eff']:.0f} of {ess['n_rows']} rows; a "
                "k-nearest-neighbour estimator needs more effective samples "
                "than a parametric test, not fewer"
            )
        return notes

    def _fit(
        self,
        data: np.ndarray,
        var_names: list[str],
        prior_knowledge: Any | None = None,
    ) -> CausalGraph:
        from tigramite.independence_tests.cmiknn import CMIknn

        n_vars = data.shape[1]
        tau_max = self.params["tau_max"]
        tau_min = max(1, self.params["tau_min"])
        allowed = _allowed_links(prior_knowledge, n_vars, tau_max, tau_min)

        test = CMIknn(
            knn=self.params["knn"],
            significance="shuffle_test",
            sig_samples=self.params["sig_samples"],
            workers=self.params["workers"],
            # Deterministic given the global numpy seed the base class sets.
            seed=self.seed,
        )

        values = np.zeros((n_vars, n_vars, tau_max + 1))
        p_values = np.ones((n_vars, n_vars, tau_max + 1))
        tested: list[tuple[int, int, int]] = []

        total = sum(len(v) for v in allowed.values())
        log.info(
            "LSTE: %d conditional mutual information tests "
            "(%d vars, lags %d..%d, %d shuffles each)",
            total,
            n_vars,
            tau_min,
            tau_max,
            self.params["sig_samples"],
        )

        done = 0
        for (source, target), lags in allowed.items():
            for lag in lags:
                array, xyz = build_embedding(
                    data, source, target, lag, self.params["history"]
                )
                value = test.get_dependence_measure(array, xyz)
                p = test.get_shuffle_significance(array, xyz, value)
                values[source, target, lag] = float(value)
                p_values[source, target, lag] = float(p)
                tested.append((source, target, lag))
                done += 1
                if done % 200 == 0:
                    log.info("LSTE: %d/%d tests done", done, total)

        selected, correction = _select_edges(
            tested, p_values, self.params["alpha"], self.params["fdr_method"]
        )

        graph = CausalGraph.empty(list(var_names), tau_max=tau_max)
        for source, target, lag in selected:
            # TE is directional by construction: information flows from the
            # earlier source to the later target, so every edge is `-->` and no
            # orientation step is needed or possible.
            graph.links[source, target, lag] = "-->"

        # Forced prior-knowledge edges are asserted, not tested.
        for source, target, lag in _forced_links(prior_knowledge, tau_max, tau_min):
            graph.links[source, target, lag] = "-->"

        graph.p_matrix = p_values
        graph.val_matrix = values
        graph.meta = {
            "estimator": "CMIknn (Kraskov-style k-nearest-neighbour CMI)",
            "estimator_rationale": (
                "a Gaussian estimator makes transfer entropy numerically "
                "equivalent to linear Granger causality, which on near-Gaussian "
                "data would reproduce the VAR result; the kNN estimator is what "
                "keeps LSTE a distinct, non-parametric method"
            ),
            "significance": "shuffle_test",
            "sig_samples": self.params["sig_samples"],
            "knn": self.params["knn"],
            "history": self.params["history"],
            "tau_min": tau_min,
            "tau_max": tau_max,
            "n_tests": len(tested),
            "alpha": self.params["alpha"],
            # A shuffle test cannot report a p-value below 1/(sig_samples + 1).
            # With many tests this floor can make an FDR correction vacuous:
            # Benjamini-Hochberg rejects rank k only when p <= alpha * k / m, so
            # nothing can be rejected unless at least alpha^-1 * m * floor tests
            # sit at the floor. Recorded so an empty LSTE graph is read as a
            # resolution limit rather than as an absence of signal.
            "p_value_resolution_floor": 1.0 / (self.params["sig_samples"] + 1),
            "min_tests_at_floor_for_fdr": int(
                np.ceil(
                    len(tested) / (self.params["sig_samples"] + 1) / self.params["alpha"]
                )
            )
            if tested
            else 0,
            "n_tests_at_resolution_floor": int(
                sum(
                    p_values[s, t, lag] <= 1.0 / (self.params["sig_samples"] + 1) + 1e-12
                    for s, t, lag in tested
                )
            ),
            **correction,
        }
        if correction.get("n_significant_corrected") == 0 and correction.get(
            "n_significant_uncorrected", 0
        ):
            log.warning(
                "LSTE: %d tests significant uncorrected but 0 survive %s; the "
                "shuffle test's p-value floor of %.4g may be the binding "
                "constraint, not the absence of signal",
                correction["n_significant_uncorrected"],
                correction["fdr_method"],
                1.0 / (self.params["sig_samples"] + 1),
            )
        return graph


def _allowed_links(
    prior_knowledge: Any | None, n_vars: int, tau_max: int, tau_min: int
) -> dict[tuple[int, int], list[int]]:
    """Which (source, target) pairs to test at which lags.

    Without prior knowledge this is every ordered pair at every lag, excluding
    self-links: a variable's own past is already the conditioning set, so
    testing X -> X would condition the source on itself.

    With prior knowledge, forbidden links are skipped -- which is also the only
    way prior knowledge can make this algorithm *faster* rather than slower.
    """
    lags = list(range(tau_min, tau_max + 1))
    allowed: dict[tuple[int, int], list[int]] = {}
    for source in range(n_vars):
        for target in range(n_vars):
            if source == target:
                continue
            allowed[(source, target)] = list(lags)

    if not prior_knowledge:
        return allowed

    for target, links in prior_knowledge.items():
        for (source, lag), mark in links.items():
            if source == target:
                continue
            lag = abs(int(lag))
            if lag < tau_min or lag > tau_max:
                continue
            if mark == "":
                remaining = [t for t in allowed.get((source, target), []) if t != lag]
                if remaining:
                    allowed[(source, target)] = remaining
                else:
                    allowed.pop((source, target), None)
    return allowed


def _forced_links(
    prior_knowledge: Any | None, tau_max: int, tau_min: int
) -> list[tuple[int, int, int]]:
    """Links prior knowledge asserts outright, which are not tested."""
    if not prior_knowledge:
        return []
    forced = []
    for target, links in prior_knowledge.items():
        for (source, lag), mark in links.items():
            lag = abs(int(lag))
            if mark == "-->" and source != target and tau_min <= lag <= tau_max:
                forced.append((int(source), int(target), lag))
    return forced


def _select_edges(
    tested: list[tuple[int, int, int]],
    p_values: np.ndarray,
    alpha: float,
    fdr_method: str,
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """Applies the significance threshold, with optional FDR control.

    `statsmodels.stats.multitest.multipletests` does the correction; no custom
    code is needed. The uncorrected count is reported alongside, so the cost of
    the correction is visible rather than assumed.
    """
    flat = np.array([p_values[s, t, lag] for s, t, lag in tested])
    uncorrected = [edge for edge, p in zip(tested, flat, strict=True) if p < alpha]

    if fdr_method in ("none", None, ""):
        return uncorrected, {
            "fdr_method": "none",
            "n_significant_uncorrected": len(uncorrected),
            "n_significant_corrected": None,
            "expected_false_positives_uncorrected": round(alpha * len(tested), 1),
        }

    from statsmodels.stats.multitest import multipletests

    rejected, adjusted, _, _ = multipletests(flat, alpha=alpha, method=fdr_method)
    selected = [edge for edge, keep in zip(tested, rejected, strict=True) if keep]
    return selected, {
        "fdr_method": fdr_method,
        "n_significant_uncorrected": len(uncorrected),
        "n_significant_corrected": len(selected),
        "expected_false_positives_uncorrected": round(alpha * len(tested), 1),
        "min_adjusted_p_value": float(adjusted.min()) if len(adjusted) else None,
        # Both selections are kept: the corrected one builds the graph, the
        # uncorrected one shows what the correction cost.
        "uncorrected_edges": [
            {"source": int(s), "target": int(t), "lag": int(lag)}
            for s, t, lag in uncorrected
        ],
    }
