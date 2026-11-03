"""Parameters and tuning of the unit-wise PCA -> PC -> causation-based T² pipeline, plus the .env loader (paths only)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PROJECT = Path(__file__).resolve().parent
REQUIRED_ENV = ["DATA_ROOT", "UNIT_FOLDERS", "FLARE_FOLDER", "ALL_TAGS_FOLDER", "CONTROL_FOLDER", "OUT_DIR",
                "FLARE_1_TAG", "FLARE_2_TAG", "FLARE_RECOVERY_TAG"]

# ---- TIME WINDOW ----
WINDOW_START = "2025-07-01"            # rows with WINDOW_START <= time < WINDOW_END are read; None = no bound
WINDOW_END = "2025-11-01"              # 4 months: July to October 2025

# ---- UNIT-WISE PCA ----
N_PCS_PER_UNIT = 16                    # x = PCs kept per unit (capped by the unit's tag count and numerical rank)
VAR_EXPLAINED_THRESHOLD = None         # e.g. 0.90: keep the fewest PCs reaching this share of the unit's variance;
                                       # when set it replaces N_PCS_PER_UNIT

# ---- FLARE TAG (flare_net = FLARE_1_TAG + FLARE_2_TAG - FLARE_RECOVERY_TAG, names in .env) ----
FLARE_NET = "flare_net"
FLARE_RECOVERY_MISSING = "zero"        # recovery tag has no value: "zero" = no recovery, "drop" = row left out

# ---- FLARE EVENTS AND WINDOWS ----
FLARE_USE_MAD_THRESHOLD = False        # True: threshold = median + K * 1.4826 * MAD; False: FLARE_THRESHOLD below
FLARE_THRESHOLD = 1.5                  # t/h; flaring = flare_net > this value (used when the flag above is False)
FLARE_MAD_K = 3.0                      # K of the median + K * MAD rule (used when the flag above is True)
EVENT_GAP = 5                          # samples; flagged rows at most this far apart form one run
FLARE_MIN_DURATION_MIN = 30            # a run is a flare event only if it lasts at least this long; shorter runs are
                                       # not events but are kept out of NOC
FLARE_PRE_WINDOW_MIN = 30              # flare region = [onset - 30 min, onset); an event is "found" if T² > limit there
FLARE_BUFFER_MIN = 120                 # NOC excludes [onset - 120 min, end + 120 min] of every event
TRAIN_FRAC = 0.60                      # NOC is drawn from the first 60 % of the window
TA_CHECK_START = "2025-05-01"          # TA's account (2 flares of about 5 min in May 2025); skipped when outside the window
TA_CHECK_END = "2025-06-01"
TA_FLARE_COUNT = 2
TA_FLARE_MINUTES = 5

# ---- CLEANING (unchanged from the earlier pipeline) ----
SENTINELS = [-999.0, -9999.0, -99999.0, 99999.0]
FFILL_MAX_GAP = 5                      # samples; only NaN runs this short are forward-filled
MAX_MISSING = 0.30                     # drop tags with more missing than this after the fill
MIN_ROW_COMPLETE = 0.90                # keep rows with at least this share of tags observed
NEAR_CONSTANT_SHARE = 0.99             # near-constant = one value covers more than 99 % of samples

# ---- PC ALGORITHM ----
PC_ALPHA = 0.01                        # Fisher-Z significance level
PC_ROW_STEP = 10                       # PC uses every 10th row
PC_MAX_K = 3                           # largest conditioning-set size allowed
PC_TEST_BUDGET = 3e7                   # upper bound on CI tests used to pick the depth cap
COLLINEAR_R = 0.999                    # |r| at or above this: one node of the pair dropped
EIG_MIN = 1e-6                         # smallest relative eigenvalue kept (PC input and per-unit PCA rank)

# ---- T² (Li, Jin & Shi 2008) ----
ALPHA0 = 0.05                          # significance level of the overall T² limit (report Eq. 13, p = PCs, k = 0)
ALPHA_TERM = 0.05                      # significance level of each per-PC limit (report Eq. 13, p = 1, k = parents)
SIGMA2_MIN = 1e-6                      # floor for sigma²_j·PA(j) (numerical guard)
IDENTITY_TOL = 1e-6                    # max relative error allowed in the sum identity check

# ---- PLOTS (plots.py) ----
PLOT_SECTION_HOURS = 48                # each section row of Figures 1-2 covers this many hours at 1-min resolution
PLOT_PC_SECTION_HOURS = 168            # section length in the 'individual PCs' figures (one week per row)
PLOT_PEAK_MIN_SEP_MIN = 60             # labelled T² peaks are at least this far apart
PLOT_MAX_LABELS_PER_SECTION = 8        # labelled T² peaks per section row
PLOT_TOP_TAGS = 2                      # tags named per PC label
PLOT_ZOOM_BEFORE_MIN = 120             # Figure 3 window: onset - 120 min ...
PLOT_ZOOM_AFTER_MIN = 30               # ... to onset + 30 min
PLOT_TOP_PCS_ZOOM = 5                  # PCs drawn separately in the T² share panel; the rest is "other PCs"
PLOT_MAX_EVENTS = 30                   # events in Figure 3 (largest flare_net peaks if there are more)
PLOT_MAX_PCS = 40                      # PCs in Figure 4 (most often labelled first)
PLOT_WIDTH_IN = 22
PLOT_ROW_HEIGHT_IN = 3.6               # height of one row (section / PC) in Figures 1, 2 and 4
PLOT_ROW_GAP = 0.75                    # vertical gap between rows, as a fraction of the row height
PLOT_EVENT_HEIGHT_IN = 10              # height of one event block in Figure 3
PLOT_DPI = 120
PLOT_COLORS = {"flare": "#c0392b", "t2": "#1f5fa8", "event": "#e67e22", "preflare": "#d62728", "noc": "#2ca02c",
               "limit": "#222222", "missing": "#9e9e9e"}


def load_paths() -> SimpleNamespace:
    """Read the data paths and flare tag names from unit_pca_pc_t2/.env and check the settings in this file.

    Returns:
        Namespace with data_root (folder holding the unit folders), out (output folder), unit_folders (the 13 unit
        folder names), flare_folder, all_tags_folder, control_folder, and flare_1_tag / flare_2_tag /
        flare_recovery_tag (exact column names in flare_data used for flare_net). Relative paths are resolved
        against unit_pca_pc_t2/.

    Raises:
        SystemExit: If a required .env entry is empty, or FLARE_RECOVERY_MISSING / N_PCS_PER_UNIT is invalid.
    """
    from dotenv import load_dotenv
    load_dotenv(PROJECT / ".env")
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k, "").strip()]
    if missing:
        sys.exit(f"Set {', '.join(missing)} in unit_pca_pc_t2/.env before running the pipeline.")
    if FLARE_RECOVERY_MISSING not in ("drop", "zero") or N_PCS_PER_UNIT < 1:
        sys.exit("config.py: FLARE_RECOVERY_MISSING must be 'drop' or 'zero' and N_PCS_PER_UNIT >= 1.")
    env = {k: os.environ[k].strip().strip("'\"") for k in REQUIRED_ENV}
    return SimpleNamespace(
        data_root=resolve_path(env["DATA_ROOT"]), out=resolve_path(env["OUT_DIR"]),
        unit_folders=[s.strip() for s in env["UNIT_FOLDERS"].split(",") if s.strip()],
        flare_folder=env["FLARE_FOLDER"], all_tags_folder=env["ALL_TAGS_FOLDER"], control_folder=env["CONTROL_FOLDER"],
        flare_1_tag=env["FLARE_1_TAG"], flare_2_tag=env["FLARE_2_TAG"], flare_recovery_tag=env["FLARE_RECOVERY_TAG"])


def resolve_path(value: str) -> Path:
    """Turn a path from .env into an absolute path.

    Args:
        value: Path text from .env (DATA_ROOT or OUT_DIR).

    Returns:
        The path itself if absolute, else the path relative to unit_pca_pc_t2/.
    """
    p = Path(value)
    return p if p.is_absolute() else (PROJECT / p).resolve()


def window_bounds() -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """The analysis window as timestamps.

    Returns:
        A tuple (WINDOW_START, WINDOW_END) as pd.Timestamp, None for an unset bound.
    """
    return (pd.Timestamp(WINDOW_START) if WINDOW_START else None, pd.Timestamp(WINDOW_END) if WINDOW_END else None)
