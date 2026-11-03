"""Parameters and tuning of the unit-wise PCA -> PC -> causation-based T² pipeline, plus the .env loader (paths only)."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PROJECT = Path(__file__).resolve().parent
REQUIRED_ENV = ["DATA_ROOT", "UNIT_FOLDERS", "FLARE_FOLDER", "ALL_TAGS_FOLDER", "CONTROL_FOLDER", "OUT_DIR",
                "FLARE_1_TAG", "FLARE_2_TAG", "FLARE_RECOVERY_TAG"]

# ---- TIME WINDOW ----
WINDOW_START = "2025-05-01"            # rows with WINDOW_START <= time < WINDOW_END are read; None = no bound
WINDOW_END = "2025-06-01"

# ---- UNIT-WISE PCA ----
N_PCS_PER_UNIT = 16                    # x = PCs kept per unit (capped by the unit's tag count and numerical rank)
VAR_EXPLAINED_THRESHOLD = None         # e.g. 0.90: keep the fewest PCs reaching this share of the unit's variance;
                                       # when set it replaces N_PCS_PER_UNIT

# ---- FLARE TAG (flare_net = FLARE_1_TAG + FLARE_2_TAG - FLARE_RECOVERY_TAG, names in .env) ----
FLARE_NET = "flare_net"
FLARE_RECOVERY_MISSING = "zero"        # recovery tag has no value: "zero" = no recovery, "drop" = row left out

# ---- FLARE EVENTS AND WINDOWS ----
FLARE_MAD_K = 3.0                      # flaring = flare_net > median + K * 1.4826 * MAD
EVENT_GAP = 5                          # samples; flagged rows at most this far apart form one event
FLARE_PRE_WINDOW_MIN = 30              # flare region = [onset - 30 min, onset)
FLARE_BUFFER_MIN = 120                 # NOC excludes [onset - 120 min, end + 120 min] of every event
TRAIN_FRAC = 0.60                      # NOC is drawn from the first 60 % of the window
TA_CHECK_START = "2025-05-01"          # TA's account to cross-check: 2 flares of about 5 min in May 2025
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
ALPHA0 = 0.05                          # T² limit chi²_p(1 - ALPHA0); per-term Bonferroni alpha' = ALPHA0 / p
SIGMA2_MIN = 1e-6                      # floor for sigma²_j·PA(j) (numerical guard)
IDENTITY_TOL = 1e-6                    # max relative error allowed in the sum identity check


def load_paths():
    """Read folder paths and flare tag names from .env and stop if any is missing."""
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


def resolve_path(value):
    """Absolute path; a relative one is read relative to this folder."""
    p = Path(value)
    return p if p.is_absolute() else (PROJECT / p).resolve()


def window_bounds():
    """WINDOW_START and WINDOW_END as timestamps (None when unset)."""
    return (pd.Timestamp(WINDOW_START) if WINDOW_START else None, pd.Timestamp(WINDOW_END) if WINDOW_END else None)
