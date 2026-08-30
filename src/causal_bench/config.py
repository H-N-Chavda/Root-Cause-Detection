"""Typed settings loaded from configs/default.yaml.

Dataclasses rather than pydantic: pydantic is not in the environment and the
task forbids new runtime dependencies beyond the YAML reader, so validation is
hand-written in each `_from_mapping`. The public surface is the same either
way -- `Config.load()` returns a validated object or raises `ConfigError`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class Config:
    """Fully validated run configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    cases: tuple[CaseConfig, ...] = ()
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
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
