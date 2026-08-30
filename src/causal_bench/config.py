"""Typed settings loaded from configs/default.yaml.

Dataclasses rather than pydantic: pydantic is not in the environment and the
task forbids new runtime dependencies beyond the YAML reader, so validation is
hand-written in each `_from_mapping`. The public surface is the same either
way -- `Config.load()` returns a validated object or raises `ConfigError`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from . import paths


class ConfigError(ValueError):
    """Raised when the configuration file is missing keys or malformed."""


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key '{key}' in {where}")
    return mapping[key]


def _as_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _as_positive_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    number = float(value)
    if not 0.0 < number < 1.0:
        raise ConfigError(f"{where} must lie strictly between 0 and 1, got {number}")
    return number


def _as_cond_size(value: Any, where: str) -> int | None:
    """A conditioning-set cap: a non-negative integer, or None for convergence."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer or null, got {value!r}")
    if value < 0:
        raise ConfigError(f"{where} must be >= 0, got {value}")
    return value


def _as_sequence(value: Any, where: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{where} must be a list, got {type(value).__name__}")
    if not value:
        raise ConfigError(f"{where} must not be empty")
    return value


def _num(
    data: Mapping[str, Any],
    key: str,
    default: float,
    where: str,
    *,
    integer: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Any:
    """One numeric key with a default, a type and an optional range.

    The EDA block carries ~40 thresholds; hand-writing a validator for each
    would bury the structure it is meant to document, so they share this.
    """
    value = data.get(key, default)
    if isinstance(value, bool):
        raise ConfigError(f"{where}.{key} must be a number, got a boolean")
    if integer:
        if not isinstance(value, int):
            raise ConfigError(f"{where}.{key} must be an integer, got {value!r}")
    elif not isinstance(value, (int, float)):
        raise ConfigError(f"{where}.{key} must be a number, got {value!r}")
    number = value if integer else float(value)
    if minimum is not None and number < minimum:
        raise ConfigError(f"{where}.{key} must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{where}.{key} must be <= {maximum}, got {number}")
    return number


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    output_prefix: str = "run"

    @classmethod
    def _from_mapping(cls, data: Mapping[str, Any]) -> RunConfig:
        seed = data.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigError(f"run.seed must be an integer, got {seed!r}")
        prefix = str(data.get("output_prefix", "run"))
        if not prefix or "/" in prefix:
            raise ConfigError(f"run.output_prefix must be a bare name, got {prefix!r}")
        return cls(seed=seed, output_prefix=prefix)


@dataclass(frozen=True)
class AlgorithmConfig:
    alpha: float = 0.01
    max_cond_set_size: int | None = 2

    @classmethod
    def _from_mapping(cls, data: Mapping[str, Any]) -> AlgorithmConfig:
        return cls(
            alpha=_as_positive_float(
                _require(data, "alpha", "algorithm"), "algorithm.alpha"
            ),
            max_cond_set_size=_as_cond_size(
                data.get("max_cond_set_size", 2), "algorithm.max_cond_set_size"
            ),
        )


@dataclass(frozen=True)
class CaseConfig:
    """One benchmark case: a title, a dataset file and a ground-truth file.

    `dataset` and `ground_truth` stay as written in the YAML; resolving them to
    absolute paths is `paths`' job, not this module's.
    """

    name: str
    dataset: str
    ground_truth: str

    @classmethod
    def _from_mapping(cls, data: Mapping[str, Any], index: int) -> CaseConfig:
        where = f"cases[{index}]"
        return cls(
            name=str(_require(data, "name", where)),
            dataset=str(_require(data, "dataset", where)),
            ground_truth=str(_require(data, "ground_truth", where)),
        )

    @property
    def dataset_path(self) -> Path:
        return paths.dataset_path(self.dataset)

    @property
    def ground_truth_path(self) -> Path:
        return paths.ground_truth_path(self.ground_truth)


@dataclass(frozen=True)
class SensitivityConfig:
    enabled: bool = True
    alpha_sweep: tuple[float, ...] = ()
    cond_sweep: tuple[int | None, ...] = ()
    thin_sweep: tuple[int, ...] = ()

    @classmethod
    def _from_mapping(cls, data: Mapping[str, Any]) -> SensitivityConfig:
        enabled = bool(data.get("enabled", True))
        if not enabled:
            return cls(enabled=False)

        alphas = _as_sequence(
            _require(data, "alpha_sweep", "sensitivity"), "sensitivity.alpha_sweep"
        )
        conds = _as_sequence(
            _require(data, "cond_sweep", "sensitivity"), "sensitivity.cond_sweep"
        )
        thins = _as_sequence(
            _require(data, "thin_sweep", "sensitivity"), "sensitivity.thin_sweep"
        )

        thin_values = []
        for i, value in enumerate(thins):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(
                    f"sensitivity.thin_sweep[{i}] must be an integer >= 1, got {value!r}"
                )
            thin_values.append(value)

        return cls(
            enabled=True,
            alpha_sweep=tuple(
                _as_positive_float(v, f"sensitivity.alpha_sweep[{i}]")
                for i, v in enumerate(alphas)
            ),
            cond_sweep=tuple(
                _as_cond_size(v, f"sensitivity.cond_sweep[{i}]")
                for i, v in enumerate(conds)
            ),
            thin_sweep=tuple(thin_values),
        )


# ---------------------------------------------------------------------------
# Exploratory data analysis
#
# One frozen dataclass per section of the `eda:` block in the YAML. Every
# threshold that turns a measured number into a verdict lives here and is echoed
# into eda_report.json, so a report always carries the criteria it was judged by.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureConfig:
    near_constant_cv: float = 1e-6
    min_unique: int = 10
    discrete_unique_ratio: float = 0.05

    @classmethod
    def _from_mapping(cls, d: Mapping[str, Any]) -> StructureConfig:
        w = "eda.structure"
        return cls(
            near_constant_cv=_num(d, "near_constant_cv", 1e-6, w, minimum=0.0),
            min_unique=_num(d, "min_unique", 10, w, integer=True, minimum=1),
            discrete_unique_ratio=_num(
                d, "discrete_unique_ratio", 0.05, w, minimum=0.0, maximum=1.0
            ),
        )


@dataclass(frozen=True)
class DistributionConfig:
    skew_threshold: float = 0.5
    excess_kurtosis_threshold: float = 1.0
    outlier_modified_z: float = 3.5
    outlier_iqr_multiplier: float = 1.5

    @classmethod
    def _from_mapping(cls, d: Mapping[str, Any]) -> DistributionConfig:
        w = "eda.distribution"
        return cls(
            skew_threshold=_num(d, "skew_threshold", 0.5, w, minimum=0.0),
            excess_kurtosis_threshold=_num(
                d, "excess_kurtosis_threshold", 1.0, w, minimum=0.0
            ),
            outlier_modified_z=_num(d, "outlier_modified_z", 3.5, w, minimum=0.0),
            outlier_iqr_multiplier=_num(d, "outlier_iqr_multiplier", 1.5, w, minimum=0.0),
        )


@dataclass(frozen=True)
class NonGaussianityConfig:
    var_lag: int = 1
    abs_excess_kurtosis_threshold: float = 1.0
    negentropy_threshold: float = 0.01
    min_fraction_nongaussian: float = 0.5

    @classmethod
    def _from_mapping(cls, d: Mapping[str, Any]) -> NonGaussianityConfig:
        w = "eda.nongaussianity"
        return cls(
            var_lag=_num(d, "var_lag", 1, w, integer=True, minimum=1),
            abs_excess_kurtosis_threshold=_num(
                d, "abs_excess_kurtosis_threshold", 1.0, w, minimum=0.0
            ),
            negentropy_threshold=_num(d, "negentropy_threshold", 0.01, w, minimum=0.0),
            min_fraction_nongaussian=_num(
                d, "min_fraction_nongaussian", 0.5, w, minimum=0.0, maximum=1.0
            ),
        )


@dataclass(frozen=True)
class LinearityConfig:
    dcor_pearson_gap: float = 0.10
    top_pairs: int = 5
    mi_n_neighbors: int = 3
    test_fraction: float = 0.3
    r2_gap_threshold: float = 0.05
    gbt_n_estimators: int = 200
    gbt_max_depth: int = 3
    gbt_learning_rate: float = 0.05

    @classmethod
    def _from_mapping(cls, d: Mapping[str, Any]) -> LinearityConfig:
        w = "eda.linearity"
        return cls(
            dcor_pearson_gap=_num(d, "dcor_pearson_gap", 0.10, w, minimum=0.0),
            top_pairs=_num(d, "top_pairs", 5, w, integer=True, minimum=1),
            mi_n_neighbors=_num(d, "mi_n_neighbors", 3, w, integer=True, minimum=1),
            test_fraction=_num(d, "test_fraction", 0.3, w, minimum=0.05, maximum=0.95),
            r2_gap_threshold=_num(d, "r2_gap_threshold", 0.05, w, minimum=0.0),
            gbt_n_estimators=_num(d, "gbt_n_estimators", 200, w, integer=True, minimum=1),
            gbt_max_depth=_num(d, "gbt_max_depth", 3, w, integer=True, minimum=1),
            gbt_learning_rate=_num(
                d, "gbt_learning_rate", 0.05, w, minimum=1e-6, maximum=1.0
            ),
        )


@dataclass(frozen=True)
class TemporalConfig:
    max_lag: int = 50
    var_max_lag: int = 10
    ljung_box_lags: int = 20
    regime_shift_sd_threshold: float = 0.5
    changepoint_min_size: int = 100
    changepoint_max_breaks: int = 5
    changepoint_penalty: float = 50.0
    xcorr_max_lag: int = 50

    @classmethod
    def _from_mapping(cls, d: Mapping[str, Any]) -> TemporalConfig:
        w = "eda.temporal"
        return cls(
            max_lag=_num(d, "max_lag", 50, w, integer=True, minimum=1),
            var_max_lag=_num(d, "var_max_lag", 10, w, integer=True, minimum=1),
            ljung_box_lags=_num(d, "ljung_box_lags", 20, w, integer=True, minimum=1),
            regime_shift_sd_threshold=_num(
                d, "regime_shift_sd_threshold", 0.5, w, minimum=0.0
            ),
            changepoint_min_size=_num(
                d, "changepoint_min_size", 100, w, integer=True, minimum=2
            ),
            changepoint_max_breaks=_num(
                d, "changepoint_max_breaks", 5, w, integer=True, minimum=0
            ),
            changepoint_penalty=_num(d, "changepoint_penalty", 50.0, w, minimum=0.0),
            xcorr_max_lag=_num(d, "xcorr_max_lag", 50, w, integer=True, minimum=1),
        )


@dataclass(frozen=True)
class ConditioningConfig:
    high_corr_threshold: float = 0.95
    report_corr_threshold: float = 0.90
    vif_threshold: float = 10.0
    condition_number_threshold: float = 1000.0
    samples_per_parameter: int = 10

    @classmethod
    def _from_mapping(cls, d: Mapping[str, Any]) -> ConditioningConfig:
        w = "eda.conditioning"
        high = _num(d, "high_corr_threshold", 0.95, w, minimum=0.0, maximum=1.0)
        report = _num(d, "report_corr_threshold", 0.90, w, minimum=0.0, maximum=1.0)
        if report > high:
            raise ConfigError(
                f"{w}.report_corr_threshold ({report}) must not exceed "
                f"{w}.high_corr_threshold ({high})"
            )
        return cls(
            high_corr_threshold=high,
            report_corr_threshold=report,
            vif_threshold=_num(d, "vif_threshold", 10.0, w, minimum=1.0),
            condition_number_threshold=_num(
                d, "condition_number_threshold", 1000.0, w, minimum=1.0
            ),
            samples_per_parameter=_num(
                d, "samples_per_parameter", 10, w, integer=True, minimum=1
            ),
        )


@dataclass(frozen=True)
class ScalingConfig:
    magnitude_ratio_threshold: float = 10.0

    @classmethod
    def _from_mapping(cls, d: Mapping[str, Any]) -> ScalingConfig:
        return cls(
            magnitude_ratio_threshold=_num(
                d, "magnitude_ratio_threshold", 10.0, "eda.scaling", minimum=1.0
            )
        )


@dataclass(frozen=True)
class FiguresConfig:
    enabled: bool = True
    dpi: int = 120
    max_grid_vars: int = 12

    @classmethod
    def _from_mapping(cls, d: Mapping[str, Any]) -> FiguresConfig:
        w = "eda.figures"
        return cls(
            enabled=bool(d.get("enabled", True)),
            dpi=_num(d, "dpi", 120, w, integer=True, minimum=10),
            max_grid_vars=_num(d, "max_grid_vars", 12, w, integer=True, minimum=1),
        )


@dataclass(frozen=True)
class EdaConfig:
    """Every parameter and threshold the EDA uses."""

    seed: int = 0
    significance: float = 0.05
    structure: StructureConfig = field(default_factory=StructureConfig)
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    nongaussianity: NonGaussianityConfig = field(default_factory=NonGaussianityConfig)
    linearity: LinearityConfig = field(default_factory=LinearityConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    figures: FiguresConfig = field(default_factory=FiguresConfig)

    @classmethod
    def _from_mapping(cls, d: Mapping[str, Any]) -> EdaConfig:
        seed = d.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigError(f"eda.seed must be an integer, got {seed!r}")
        return cls(
            seed=seed,
            significance=_num(d, "significance", 0.05, "eda", minimum=1e-12, maximum=0.5),
            structure=StructureConfig._from_mapping(
                _as_mapping(d.get("structure", {}), "eda.structure")
            ),
            distribution=DistributionConfig._from_mapping(
                _as_mapping(d.get("distribution", {}), "eda.distribution")
            ),
            nongaussianity=NonGaussianityConfig._from_mapping(
                _as_mapping(d.get("nongaussianity", {}), "eda.nongaussianity")
            ),
            linearity=LinearityConfig._from_mapping(
                _as_mapping(d.get("linearity", {}), "eda.linearity")
            ),
            temporal=TemporalConfig._from_mapping(
                _as_mapping(d.get("temporal", {}), "eda.temporal")
            ),
            conditioning=ConditioningConfig._from_mapping(
                _as_mapping(d.get("conditioning", {}), "eda.conditioning")
            ),
            scaling=ScalingConfig._from_mapping(
                _as_mapping(d.get("scaling", {}), "eda.scaling")
            ),
            figures=FiguresConfig._from_mapping(
                _as_mapping(d.get("figures", {}), "eda.figures")
            ),
        )

    def thresholds(self) -> dict[str, Any]:
        """Flat {section.key: value} dump, embedded verbatim in the JSON report."""
        from dataclasses import asdict, fields

        out: dict[str, Any] = {"seed": self.seed, "significance": self.significance}
        for f in fields(self):
            value = getattr(self, f.name)
            # `is_dataclass` also admits a dataclass *class*; only instances
            # can be flattened, and every field here holds an instance.
            if is_dataclass(value) and not isinstance(value, type):
                for key, inner in asdict(value).items():
                    out[f"{f.name}.{key}"] = inner
        return out


@dataclass(frozen=True)
class DiscoveryConfig:
    """Parameters for `causal-bench run` (phase 3).

    `tau_max` and `standardize` default to None, meaning "take it from the EDA
    report". A lag order chosen here rather than measured would defeat the point
    of the Phase 2 handoff, so the runner refuses to invent one.
    """

    seed: int = 0
    algorithms: tuple[str, ...] = ("pc", "pcmci_plus", "var_lingam", "lste")
    tau_max: int | None = None
    standardize: bool | None = None
    tau_sweep: bool = True
    undirected_weight: float = 0.5
    prior_knowledge: str | None = None
    run_with_prior_knowledge: bool = True
    algorithm_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def _from_mapping(cls, d: Mapping[str, Any]) -> DiscoveryConfig:
        w = "discovery"
        seed = d.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigError(f"{w}.seed must be an integer, got {seed!r}")

        algorithms = d.get("algorithms", cls.algorithms)
        if isinstance(algorithms, str) or not isinstance(algorithms, Sequence):
            raise ConfigError(f"{w}.algorithms must be a list, got {algorithms!r}")
        if not algorithms:
            raise ConfigError(f"{w}.algorithms must not be empty")

        tau_max = d.get("tau_max")
        if tau_max is not None:
            tau_max = _as_cond_size(tau_max, f"{w}.tau_max")
            if tau_max is not None and tau_max < 1:
                raise ConfigError(f"{w}.tau_max must be >= 1, got {tau_max}")

        standardize = d.get("standardize")
        if standardize is not None and not isinstance(standardize, bool):
            raise ConfigError(f"{w}.standardize must be a boolean or null")

        params = d.get("algorithm_params", {}) or {}
        if not isinstance(params, Mapping):
            raise ConfigError(f"{w}.algorithm_params must be a mapping")
        for name, entry in params.items():
            if not isinstance(entry, Mapping):
                raise ConfigError(f"{w}.algorithm_params.{name} must be a mapping")

        prior = d.get("prior_knowledge")
        return cls(
            seed=seed,
            algorithms=tuple(str(a) for a in algorithms),
            tau_max=tau_max,
            standardize=standardize,
            tau_sweep=bool(d.get("tau_sweep", True)),
            undirected_weight=_num(
                d, "undirected_weight", 0.5, w, minimum=0.0, maximum=1.0
            ),
            prior_knowledge=None if prior is None else str(prior),
            run_with_prior_knowledge=bool(d.get("run_with_prior_knowledge", True)),
            algorithm_params={str(k): dict(v) for k, v in params.items()},
        )

    def params_for(self, algorithm: str) -> dict[str, Any]:
        return dict(self.algorithm_params.get(algorithm, {}))


@dataclass(frozen=True)
class Config:
    """Fully validated run configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    cases: tuple[CaseConfig, ...] = ()
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    eda: EdaConfig = field(default_factory=EdaConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    source: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], source: Path | None = None) -> Config:
        data = _as_mapping(data, "config root")
        cases = _as_sequence(_require(data, "cases", "config root"), "cases")
        return cls(
            run=RunConfig._from_mapping(_as_mapping(data.get("run", {}), "run")),
            algorithm=AlgorithmConfig._from_mapping(
                _as_mapping(_require(data, "algorithm", "config root"), "algorithm")
            ),
            cases=tuple(
                CaseConfig._from_mapping(_as_mapping(c, f"cases[{i}]"), i)
                for i, c in enumerate(cases)
            ),
            sensitivity=SensitivityConfig._from_mapping(
                _as_mapping(data.get("sensitivity", {}), "sensitivity")
            ),
            eda=EdaConfig._from_mapping(_as_mapping(data.get("eda", {}), "eda")),
            discovery=DiscoveryConfig._from_mapping(
                _as_mapping(data.get("discovery", {}), "discovery")
            ),
            source=source,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Loads and validates a YAML config. `None` uses configs/default.yaml."""
        resolved = Path(path).expanduser() if path else paths.DEFAULT_CONFIG_PATH
        if not resolved.exists():
            raise ConfigError(f"config file not found: {resolved}")
        try:
            raw = yaml.safe_load(resolved.read_text())
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse {resolved}: {exc}") from exc
        if raw is None:
            raise ConfigError(f"config file is empty: {resolved}")
        return cls.from_mapping(raw, source=resolved.resolve())

    def describe(self) -> str:
        """One-block summary for the CLI run header."""
        cap = (
            "converged"
            if self.algorithm.max_cond_set_size is None
            else str(self.algorithm.max_cond_set_size)
        )
        return "\n".join(
            [
                f"  config      : {self.source or '<defaults>'}",
                f"  alpha       : {self.algorithm.alpha}",
                f"  max cond set: {cap}",
                f"  seed        : {self.run.seed}",
                f"  cases       : {len(self.cases)}",
                f"  sensitivity : {'on' if self.sensitivity.enabled else 'off'}",
            ]
        )
