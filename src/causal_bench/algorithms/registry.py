"""Name -> algorithm class lookup, so the CLI and config never import a class.

Adding an algorithm means adding one entry here and nothing else: the runner,
the CLI's `--algorithms` flag and the config all resolve through this table.
"""

from __future__ import annotations

from typing import Any

from .base import CausalDiscoveryAlgorithm
from .lste import LSTEAlgorithm
from .pc import PCAlgorithm
from .pc_manual import PCManualAlgorithm
from .pcmci_plus import PCMCIPlusAlgorithm
from .var_lingam import VARLiNGAMAlgorithm

REGISTRY: dict[str, type[CausalDiscoveryAlgorithm]] = {
    "pc": PCAlgorithm,
    "pc_manual": PCManualAlgorithm,
    "pcmci_plus": PCMCIPlusAlgorithm,
    "var_lingam": VARLiNGAMAlgorithm,
    "lste": LSTEAlgorithm,
}

#: The four benchmark entries. `pc_manual` is a correctness oracle, not a
#: competitor, so it is excluded from the default set.
DEFAULT_ALGORITHMS = ("pc", "pcmci_plus", "var_lingam", "lste")


def available() -> list[str]:
    return sorted(REGISTRY)


def get_algorithm_class(name: str) -> type[CausalDiscoveryAlgorithm]:
    key = name.strip().lower()
    if key not in REGISTRY:
        raise KeyError(f"unknown algorithm {name!r}; available: {available()}")
    return REGISTRY[key]


def build(name: str, **params: Any) -> CausalDiscoveryAlgorithm:
    """Instantiates an algorithm, dropping config keys it does not accept.

    Config blocks carry per-algorithm keys that other algorithms have no use
    for; filtering here keeps a shared config block from being a TypeError.
    """
    import inspect

    cls = get_algorithm_class(name)
    # Filter against the NAMED parameters only. Every wrapper also declares
    # **params so it can forward extras into its metadata, which means a
    # "does it take **kwargs" check would never filter anything and a stray
    # config key would be silently recorded as if it had been used.
    named = {
        key
        for key, parameter in inspect.signature(cls.__init__).parameters.items()
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        and key != "self"
    }
    dropped = sorted(set(params) - named)
    if dropped:
        from ..utils.logging import get_logger

        get_logger(__name__).debug(
            "%s does not take %s; dropped from its parameters", name, dropped
        )
    return cls(**{k: v for k, v in params.items() if k in named})
