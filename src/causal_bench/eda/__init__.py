"""Exploratory data analysis.

The EDA is the gate that decides which causal-discovery algorithm can produce a
meaningful answer on a given dataset. The four algorithms this project
benchmarks hold conflicting assumptions -- PC and PCMCI+ want Gaussian data,
VAR-LiNGAM wants non-Gaussian innovations, some want linearity, some model time
explicitly -- so "which method applies here" is a measurement, not a preference.

Nothing in this package is specific to any dataset: every entry point takes a
dataframe and a config.

    from causal_bench.eda import runner
    findings = runner.run(dataset_path, config.eda, out_dir)
"""

from . import (
    conditioning,
    distribution,
    figures,
    linearity,
    report,
    runner,
    structure,
    suitability,
    temporal,
)

__all__ = [
    "conditioning",
    "distribution",
    "figures",
    "linearity",
    "report",
    "runner",
    "structure",
    "suitability",
    "temporal",
]
