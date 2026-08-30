"""Collinearity, effective sample size, conditioning capacity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_bench.eda import conditioning


def test_independent_columns_are_well_conditioned(gaussian_frame, eda_config):
    result = conditioning.collinearity(gaussian_frame, eda_config.conditioning)
    assert result["ill_conditioned"] is False
    assert result["full_rank"] is True
    assert result["condition_number"] < 10
    assert result["high_correlation_pairs"] == []
    assert result["vif"]["n_above_threshold"] == 0
    assert result["vif"]["max"] < eda_config.conditioning.vif_threshold


def test_collinear_pair_blows_up_every_diagnostic(collinear_frame, eda_config):
    """X2 = X1 + 0.01 noise. The condition number, the smallest eigenvalue, the
    VIFs and the pair list must all agree that this is a problem."""
    result = conditioning.collinearity(collinear_frame, eda_config.conditioning)

    assert result["ill_conditioned"] is True
    assert result["condition_number"] > eda_config.conditioning.condition_number_threshold
    assert result["min_eigenvalue"] < 0.01

    assert len(result["high_correlation_pairs"]) == 1
    pair = result["high_correlation_pairs"][0]
    assert {pair["a"], pair["b"]} == {"X1", "X2"}
    assert abs(pair["correlation"]) > 0.99
    assert pair["above_high_threshold"] is True

    assert set(result["vif"]["columns_above_threshold"]) == {"X1", "X2"}
    assert result["vif"]["per_variable"]["X3"] < eda_config.conditioning.vif_threshold


def test_high_correlation_pairs_are_sorted_by_magnitude(eda_config):
    generator = np.random.default_rng(0)
    base = generator.normal(size=400)
    frame = pd.DataFrame(
        {
            "X1": base,
            "X2": base + generator.normal(scale=0.30, size=400),  # weaker
            "X3": base + generator.normal(scale=0.01, size=400),  # stronger
        }
    )
    pairs = conditioning.collinearity(frame, eda_config.conditioning)[
        "high_correlation_pairs"
    ]
    magnitudes = [abs(p["correlation"]) for p in pairs]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_white_noise_keeps_its_full_sample_size(gaussian_frame, eda_config):
    result = conditioning.effective_sample_size(gaussian_frame, eda_config.temporal)
    assert result["shrinkage_factor"] > 0.5
    assert result["binding_n_eff"] > 0.5 * result["n_rows"]


def test_autocorrelation_shrinks_the_effective_sample_size(ar_frame, eda_config):
    """The AR(2) has a positive lag-1 autocorrelation, so both estimators must
    report fewer effective observations than rows."""
    result = conditioning.effective_sample_size(ar_frame, eda_config.temporal)
    assert result["min_n_eff_ar1"] < result["n_rows"]
    assert result["binding_n_eff"] < result["n_rows"]
    assert result["shrinkage_factor"] < 1.0


def test_conditioning_capacity_falls_with_the_effective_sample_size(eda_config):
    """The whole point of the effective-n correction: the same row count buys a
    smaller conditioning set once serial correlation is accounted for."""
    result = conditioning.sample_size_adequacy(
        n_vars=30, n_eff=50.0, n_rows=1500, cfg=eda_config.conditioning
    )
    at_raw = result["at_raw_n"]["practical_max_conditioning_set"]
    at_eff = result["at_effective_n"]["practical_max_conditioning_set"]
    assert at_eff < at_raw
    assert result["recommended_max_conditioning_set"] == at_eff


def test_conditioning_capacity_is_capped_by_the_graph(eda_config):
    """With 5 variables PC can condition on at most 3, however many rows there
    are."""
    result = conditioning.sample_size_adequacy(
        n_vars=5, n_eff=100000.0, n_rows=100000, cfg=eda_config.conditioning
    )
    assert result["graph_ceiling"] == 3
    assert result["recommended_max_conditioning_set"] == 3
