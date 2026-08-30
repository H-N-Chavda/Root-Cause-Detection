"""End-to-end smoke tests on a synthetic system with a known graph.

This is the test that matters. If an algorithm cannot recover a graph generated
here, the wrapper is broken, and no amount of Tennessee Eastman output would
reveal it -- a transposed converter produces a wrong graph and a plausible SHD.

The thresholds are deliberately loose. The claim being made is "the wrapper is
wired up correctly", not "this algorithm is accurate": PC is mis-specified on
autocorrelated data even when the data is synthetic, and bivariate transfer
entropy returns indirect paths as edges by construction. Each threshold below
says which of those it is allowing for.
"""

from __future__ import annotations

import pytest

from causal_bench.algorithms.registry import build
from causal_bench.discovery.targets import build_targets
from causal_bench.graph.conversions import lagged_to_summary
from causal_bench.scoring.shd import align_to, structural_hamming_distance

pytestmark = pytest.mark.slow


def _recover(name, data, names, truth, **params):
    algorithm = build(name, seed=0, **params)
    graph = algorithm.run(data, names)
    assert graph.meta["completed"], f"{name} failed: {graph.meta.get('error')}"
    summary = graph if graph.tau_max == 0 else lagged_to_summary(graph)
    score = structural_hamming_distance(align_to(summary, names), truth.gt_projected)
    return graph, summary, score


@pytest.fixture(scope="module")
def targets(synthetic_truth, synthetic_names):
    return build_targets(synthetic_truth, synthetic_names)


def test_targets_match_the_generating_graph(targets):
    """No latent nodes here, so induced and projected coincide."""
    assert targets.provenance["gt_projected_edges"] == 4
    assert targets.provenance["gt_induced_edges"] == 4
    assert targets.latent_nodes == []


def test_pcmci_plus_recovers_the_graph(synthetic_data, synthetic_names, targets):
    """PCMCI+ models the lags explicitly, so it should recover this exactly."""
    _, summary, score = _recover(
        "pcmci_plus",
        synthetic_data,
        synthetic_names,
        targets,
        tau_max=2,
        pc_alpha=0.01,
    )
    assert score.true_positives == 4
    assert score.false_negatives == 0
    assert score.shd == 0.0


def test_var_lingam_recovers_the_graph(synthetic_data, synthetic_names, targets):
    """The innovations are Laplace, so the contemporaneous order is identified
    and VAR-LiNGAM should get the whole graph including directions."""
    _, summary, score = _recover(
        "var_lingam",
        synthetic_data,
        synthetic_names,
        targets,
        tau_max=2,
        run_bootstrap=False,
    )
    assert score.true_positives == 4
    assert score.false_negatives == 0
    assert score.shd == 0.0


def test_pc_recovers_the_skeleton(synthetic_data, synthetic_names, targets):
    """PC assumes i.i.d. samples and this series is autocorrelated, so it is
    mis-specified even here. The skeleton is what is asserted; orientations are
    a CPDAG's business and are checked against the CPDAG target instead."""
    _, summary, score = _recover(
        "pc", synthetic_data, synthetic_names, targets, pc_alpha=0.01
    )
    assert score.false_negatives <= 1
    assert score.true_positives + score.unoriented_edges >= 3


def test_pc_scores_better_against_the_cpdag_target(
    synthetic_data, synthetic_names, targets
):
    """Decision 4 in action: a CPDAG's undirected edges are abstentions, and
    scoring them against a DAG charges PC for a limit no observational method
    can beat."""
    graph = build("pc", seed=0, pc_alpha=0.01).run(synthetic_data, synthetic_names)
    aligned = align_to(graph, synthetic_names)

    against_dag = structural_hamming_distance(aligned, targets.gt_projected)
    against_cpdag = structural_hamming_distance(aligned, targets.gt_projected_cpdag)

    assert against_cpdag.shd <= against_dag.shd


def test_lste_recovers_every_true_edge(synthetic_data, synthetic_names, targets):
    """Bivariate transfer entropy finds every real edge but also reports
    indirect paths as edges, because it conditions only on the target's own
    past and never on the other sources. So recall is what is asserted here,
    not precision -- the density is a property of the method, not a bug in the
    wrapper, and it is reported as a finding rather than tuned away."""
    _, summary, score = _recover(
        "lste",
        synthetic_data,
        synthetic_names,
        targets,
        tau_max=2,
        alpha=0.05,
        sig_samples=100,
        workers=-1,
        fdr_method="fdr_bh",
    )
    assert score.false_negatives == 0
    assert score.true_positives == 4


def test_pc_manual_and_tigramite_pc_agree_on_the_skeleton(synthetic_data, synthetic_names):
    """The correctness oracle. Skeletons must match; orientations legitimately
    differ, because pc_manual uses the plain collider rule and tigramite
    defaults to the majority rule, so those are reported rather than asserted.
    """
    manual = build("pc_manual", seed=0, pc_alpha=0.01, max_cond_set_size=None).run(
        synthetic_data, synthetic_names
    )
    tigramite = build("pc", seed=0, pc_alpha=0.01).run(synthetic_data, synthetic_names)

    assert manual.meta["completed"] and tigramite.meta["completed"]
    assert manual.adjacency_pairs() == tigramite.adjacency_pairs()

    manual_directed = {(c, e) for c, e, _ in manual.directed_edges()}
    tigramite_directed = {(c, e) for c, e, _ in tigramite.directed_edges()}
    if manual_directed != tigramite_directed:
        print(
            "orientation differences (expected: plain vs majority collider "
            f"rule): only in pc_manual {manual_directed - tigramite_directed}, "
            f"only in tigramite {tigramite_directed - manual_directed}"
        )


def test_every_registered_algorithm_runs_and_emits_the_shared_object(
    synthetic_data, synthetic_names, targets
):
    """Whatever else it gets right, each wrapper must return the same type with
    consistent metadata, or the runner cannot treat them uniformly."""
    params = {
        "pc": {"pc_alpha": 0.05},
        "pc_manual": {"pc_alpha": 0.05, "max_cond_set_size": 2},
        "pcmci_plus": {"tau_max": 1, "pc_alpha": 0.05},
        "var_lingam": {"tau_max": 1, "run_bootstrap": False},
        "lste": {"tau_max": 1, "sig_samples": 60, "workers": -1, "fdr_method": "none"},
    }
    for name, kwargs in params.items():
        graph = build(name, seed=0, **kwargs).run(synthetic_data, synthetic_names)
        assert graph.meta["completed"], f"{name}: {graph.meta.get('error')}"
        assert graph.var_names == synthetic_names
        assert graph.links.shape[:2] == (5, 5)
        assert graph.meta["algorithm"] == name
        assert graph.meta["runtime_seconds"] is not None
        summary = graph if graph.tau_max == 0 else lagged_to_summary(graph)
        assert summary.tau_max == 0
        structural_hamming_distance(
            align_to(summary, synthetic_names), targets.gt_projected
        )


def test_var_lingam_bootstrap_is_stable_on_non_gaussian_innovations(
    synthetic_data, synthetic_names
):
    """The contrast for the Tennessee Eastman result: with Laplace innovations
    the contemporaneous order *is* identified, so the bootstrap should find
    stable edges. If it did not, the bootstrap itself would be uninformative
    and the Tennessee finding would prove nothing."""
    graph = build(
        "var_lingam", seed=0, tau_max=1, bootstrap_samples=40, run_bootstrap=True
    ).run(synthetic_data, synthetic_names)

    bootstrap = graph.meta["lag0_bootstrap"]
    assert bootstrap["available"]
    assert bootstrap["n_edges_selected_at_least_half_the_time"] >= 2
    assert bootstrap["max_selection_frequency"] > 0.9
