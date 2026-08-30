"""Synthetic fixtures with known answers.

These tests verify the EDA code, not any dataset. Every fixture is generated
from a known process, so the expected verdict is known in advance and a test
failure means the measurement is wrong rather than that the data is unusual.

Sizes are kept small (a few hundred rows, a handful of columns) because these
run on every commit; the properties being asserted are all detectable well
below the real datasets' size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_bench.config import EdaConfig

SEED = 0
N = 600


@pytest.fixture
def eda_config() -> EdaConfig:
    """Config defaults, i.e. the same thresholds the real run uses."""
    return EdaConfig()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def gaussian_frame() -> pd.DataFrame:
    """Independent standard normals: must pass every Gaussian check."""
    generator = np.random.default_rng(SEED)
    return pd.DataFrame(
        generator.normal(size=(N, 4)), columns=[f"X{i + 1}" for i in range(4)]
    )


@pytest.fixture
def laplace_frame() -> pd.DataFrame:
    """Laplace marginals: excess kurtosis 3 in the limit, so the Gaussian
    verdict must come out false and the negentropy must be clearly positive."""
    generator = np.random.default_rng(SEED)
    return pd.DataFrame(
        generator.laplace(size=(N, 4)), columns=[f"X{i + 1}" for i in range(4)]
    )


@pytest.fixture
def linear_frame() -> pd.DataFrame:
    """X1 -> X2 -> X3, purely linear and additive.

    Pearson and distance correlation should agree closely, so the gap is small
    and a tree must not beat a linear fit out of sample.
    """
    generator = np.random.default_rng(SEED)
    x1 = generator.normal(size=N)
    x2 = 0.9 * x1 + generator.normal(scale=0.3, size=N)
    x3 = 0.9 * x2 + generator.normal(scale=0.3, size=N)
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})


@pytest.fixture
def nonlinear_frame() -> pd.DataFrame:
    """X2 = X1^2 with symmetric X1, so Pearson is ~0 while the dependence is
    deterministic. This is the case a correlation-based independence test is
    blind to, and the dCor-minus-Pearson gap must be large."""
    generator = np.random.default_rng(SEED)
    x1 = generator.uniform(-2, 2, size=N)
    x2 = x1**2 + generator.normal(scale=0.1, size=N)
    x3 = generator.normal(size=N)
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})


@pytest.fixture
def ar_frame() -> pd.DataFrame:
    """A stationary AR(2) system with no cross-variable structure.

    Coefficients 0.6 and -0.3 give roots outside the unit circle, so the series
    is stationary, and the second PACF coefficient is large enough that a lag
    order of 2 is recoverable at this sample size.
    """
    generator = np.random.default_rng(SEED)
    n_vars, burn = 3, 200
    total = N + burn
    series = np.zeros((total, n_vars))
    noise = generator.normal(size=(total, n_vars))
    for t in range(2, total):
        series[t] = 0.6 * series[t - 1] - 0.3 * series[t - 2] + noise[t]
    return pd.DataFrame(series[burn:], columns=[f"X{i + 1}" for i in range(n_vars)])


@pytest.fixture
def random_walk_frame() -> pd.DataFrame:
    """A unit-root process: ADF must fail to reject, KPSS must reject."""
    generator = np.random.default_rng(SEED)
    return pd.DataFrame(
        np.cumsum(generator.normal(size=(N, 2)), axis=0), columns=["X1", "X2"]
    )


@pytest.fixture
def collinear_frame() -> pd.DataFrame:
    """X2 is X1 plus a whisper of noise: correlation ~0.9999, so the condition
    number must blow up and the VIF for both must be very large."""
    generator = np.random.default_rng(SEED)
    x1 = generator.normal(size=N)
    return pd.DataFrame(
        {
            "X1": x1,
            "X2": x1 + generator.normal(scale=0.01, size=N),
            "X3": generator.normal(size=N),
        }
    )


@pytest.fixture
def regime_shift_frame() -> pd.DataFrame:
    """A clean level shift of 4 SD at the midpoint, with no other structure."""
    generator = np.random.default_rng(SEED)
    series = generator.normal(size=N)
    series[N // 2 :] += 4.0
    return pd.DataFrame({"X1": series, "X2": generator.normal(size=N)})


@pytest.fixture
def delayed_frame() -> pd.DataFrame:
    """X2 is X1 delayed by exactly 7 samples, so the peak cross-correlation
    must sit at lag 7 with X1 leading."""
    generator = np.random.default_rng(SEED)
    x1 = generator.normal(size=N + 7)
    x2 = np.roll(x1, 7) + generator.normal(scale=0.05, size=N + 7)
    return pd.DataFrame({"X1": x1[7:], "X2": x2[7:]})


@pytest.fixture
def dirty_frame() -> pd.DataFrame:
    """Every integrity defect at once: a constant column, an exact duplicate
    column, a binary column, a missing value and a duplicated row."""
    generator = np.random.default_rng(SEED)
    base = generator.normal(size=40)
    frame = pd.DataFrame(
        {
            "X1": base,
            "X2": base.copy(),
            "X3": np.full(40, 7.0),
            "X4": generator.integers(0, 2, size=40).astype(float),
            "X5": generator.normal(size=40),
        }
    )
    frame.loc[0, "X5"] = np.nan
    frame.loc[39] = frame.loc[38]
    return frame
