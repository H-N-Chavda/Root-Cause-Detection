"""Linear versus general dependence, on systems with a known functional form."""

from __future__ import annotations

from causal_bench.eda import linearity

SEED = 0


def _pair(result, a, b):
    for entry in result["pairs"]:
        if {entry["a"], entry["b"]} == {a, b}:
            return entry
    raise AssertionError(f"pair {a}-{b} not found")


def test_linear_system_shows_no_dependence_gap(linear_frame, eda_config):
    """When the truth is linear, distance correlation must not see more than
    Pearson does."""
    result = linearity.pairwise_dependence(linear_frame, eda_config.linearity, SEED)
    assert result["n_pairs"] == 3
    assert result["n_pairs_above_gap_threshold"] == 0
    assert result["max_gap"] < eda_config.linearity.dcor_pearson_gap

    direct = _pair(result, "X1", "X2")
    assert abs(direct["pearson"]) > 0.8
    assert direct["distance_correlation"] > 0.8


def test_nonlinear_system_shows_a_large_dependence_gap(nonlinear_frame, eda_config):
    """X2 = X1^2 with symmetric X1: Pearson is blind, distance correlation is
    not. This is precisely the structure a Fisher-z test cannot detect."""
    result = linearity.pairwise_dependence(nonlinear_frame, eda_config.linearity, SEED)
    quadratic = _pair(result, "X1", "X2")

    assert abs(quadratic["pearson"]) < 0.15
    assert quadratic["distance_correlation"] > 0.4
    assert quadratic["gap"] > eda_config.linearity.dcor_pearson_gap
    assert quadratic["mutual_information"] > 0.3

    # It must also be the pair the ranking surfaces first.
    top = result["most_nonlinear_pairs"][0]
    assert {top["a"], top["b"]} == {"X1", "X2"}


def test_independent_columns_have_low_dependence_on_every_measure(
    gaussian_frame, eda_config
):
    result = linearity.pairwise_dependence(gaussian_frame, eda_config.linearity, SEED)
    for entry in result["pairs"]:
        assert abs(entry["pearson"]) < 0.2
        assert entry["distance_correlation"] < 0.2
        assert entry["mutual_information"] < 0.15


def test_tree_does_not_beat_linear_on_a_linear_system(linear_frame, eda_config):
    result = linearity.linear_versus_nonparametric(linear_frame, eda_config.linearity, SEED)
    assert result["split"].startswith("time-ordered")
    assert result["n_with_nonlinear_gain"] == 0
    assert result["mean_r2_gap"] <= eda_config.linearity.r2_gap_threshold


def test_tree_beats_linear_on_a_nonlinear_system(nonlinear_frame, eda_config):
    result = linearity.linear_versus_nonparametric(
        nonlinear_frame, eda_config.linearity, SEED
    )
    x2 = result["per_variable"]["X2"]
    assert x2["r2_gap"] > eda_config.linearity.r2_gap_threshold
    assert x2["nonlinear_gain"] is True
    assert x2["r2_tree_oos"] > x2["r2_linear_oos"]


def test_ramsey_reset_rejects_a_misspecified_linear_fit(nonlinear_frame, eda_config):
    result = linearity.linear_versus_nonparametric(
        nonlinear_frame, eda_config.linearity, SEED
    )
    reset = result["per_variable"]["X2"]["ramsey_reset"]
    assert reset["p_value"] is not None
    assert reset["reject"] is True


def test_ramsey_reset_accepts_a_correct_linear_fit(linear_frame, eda_config):
    result = linearity.linear_versus_nonparametric(linear_frame, eda_config.linearity, SEED)
    assert result["n_reset_rejections"] == 0


def test_results_are_reproducible(nonlinear_frame, eda_config):
    """Same input, same seed, same numbers -- including the kNN mutual
    information and the boosted tree, which are the two stochastic pieces."""
    first = linearity.analyse(nonlinear_frame, eda_config.linearity, SEED)
    second = linearity.analyse(nonlinear_frame, eda_config.linearity, SEED)
    assert first == second
