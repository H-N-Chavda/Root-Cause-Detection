"""Single source of truth for every filesystem path this package uses.

No other module may build a data path. Import the constants or the helper
functions below instead, so that relocating the repository, installing the
package non-editably, or pointing it at a different data drive is a change in
one file (or one environment variable) rather than a grep-and-pray.

Environment overrides, all optional and all absolute:

    CAUSAL_BENCH_ROOT         repository root; the other three default under it
    CAUSAL_BENCH_DATA_DIR     read-only input tree (raw/ and ground_truth/)
    CAUSAL_BENCH_CONFIG_DIR   directory holding default.yaml
    CAUSAL_BENCH_RESULTS_DIR  writable output tree

`data/` is read only. Everything this package generates goes under
`results/`, in a timestamped run directory produced by `new_run_dir`.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

#: The distribution/import name. Defined here so a rename touches one line
#: plus the directory itself; every other module derives it from this.
PACKAGE_NAME = "causal_bench"

#: Prefix for every environment-variable override read by this module.
ENV_PREFIX = PACKAGE_NAME.upper()

PACKAGE_DIR = Path(__file__).resolve().parent


def _env_dir(suffix: str) -> Path | None:
    value = os.environ.get(f"{ENV_PREFIX}_{suffix}")
    return Path(value).expanduser().resolve() if value else None


def _default_root() -> Path:
    """Repository root inferred from the package location.

    Under the src/ layout (and an editable install, which keeps it) the package
    sits at ``<root>/src/causal_bench``, so the root is two levels up. A wheel
    installed into site-packages has no repository around it; in that case the
    caller must set CAUSAL_BENCH_ROOT (or the specific *_DIR override) and the
    absence is reported when a path is actually used, not at import time.
    """
    return PACKAGE_DIR.parent.parent


REPO_ROOT = _env_dir("ROOT") or _default_root()

DATA_DIR = _env_dir("DATA_DIR") or REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

CONFIG_DIR = _env_dir("CONFIG_DIR") or REPO_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"

RESULTS_DIR = _env_dir("RESULTS_DIR") or REPO_ROOT / "results"


def dataset_path(name: str) -> Path:
    """Absolute path to a raw dataset file.

    `name` may be a bare file name resolved under `RAW_DATA_DIR`, or an
    absolute/explicit path, which is returned unchanged so a caller can point
    the CLI at a file outside the repository.
    """
    return _resolve(name, RAW_DATA_DIR, "dataset")


def ground_truth_path(name: str) -> Path:
    """Absolute path to a ground-truth adjacency file."""
    return _resolve(name, GROUND_TRUTH_DIR, "ground truth")


def _resolve(name: str, base: Path, kind: str) -> Path:
    candidate = Path(name).expanduser()
    if candidate.is_absolute() or len(candidate.parts) > 1:
        return candidate.resolve()
    path = (base / candidate).resolve()
    if not path.exists() and not base.exists():
        raise FileNotFoundError(
            f"{kind} directory {base} does not exist. Set {ENV_PREFIX}_DATA_DIR "
            f"or {ENV_PREFIX}_ROOT when the package is installed outside the "
            "repository."
        )
    return path


def new_run_dir(base: Path | None = None, prefix: str = "run") -> Path:
    """Creates and returns a fresh timestamped output directory.

    Output never lands in `data/`; this is the only place the package creates
    directories for its own artefacts.
    """
    root = Path(base).expanduser().resolve() if base is not None else RESULTS_DIR
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / f"{prefix}-{stamp}"
    suffix = 1
    while path.exists():
        path = root / f"{prefix}-{stamp}-{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def describe() -> str:
    """Human-readable dump of the resolved paths, for the CLI run summary."""
    rows = [
        ("repo root", REPO_ROOT),
        ("data", DATA_DIR),
        ("configs", CONFIG_DIR),
        ("results", RESULTS_DIR),
    ]
    return "\n".join(f"  {label:<12}: {value}" for label, value in rows)
