"""Shared paths, constants, logging, data readers and plotting helpers for the PCA -> PC -> T² pipeline."""
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ----------------------------------------------------------------------------- paths
PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs"
DATA = OUT / "data"
FIG = OUT / "figures"
TAB = OUT / "tables"
for _d in (DATA, FIG, TAB):
    _d.mkdir(parents=True, exist_ok=True)


def _read_env(path):
    """Minimal KEY=VALUE reader for the project .env file (avoids an extra dependency)."""
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip("'\"")
    return env


# The only absolute path in the project. Set in pca_pc_t2/.env (an environment variable overrides it).
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or _read_env(PROJECT / ".env").get("DATA_ROOT", ""))

# ----------------------------------------------------------------------------- constants
SEED = 42
UNITS = ["APS_NVPS_LEU", "DAK", "DIU", "ISOM", "ISOM_3", "ISOM_4", "LOUP",
         "NSU_NHT_CCR", "PDA", "PG", "SEU1", "SEU2", "SEU3"]
TAG_FOLDER = "All_unit_tags"
FLARE_FOLDER = "flare_data"
CONTROL_FOLDER = "unit_wise_control_data"
EXPECTED_N_PARAMS = 418

K = 26                      # number of PCs (user-specified)
SENTINELS = [-999.0, -9999.0, -99999.0, 99999.0]   # replaced by NaN in Stage 1 (user decision)
FFILL_MAX_GAP = 5           # samples; only NaN runs this short are forward-filled (Stage 1)
MAX_MISSING = 0.30          # drop tags with more missing than this after the fill
TRAIN_FRAC = 0.60           # TRAIN = first 60 % of the time span (minus flare windows ± buffer)
FLARE_BUFFER = pd.Timedelta(hours=2)    # excluded around every flare window from TRAIN (user decision)
MATCH_WINDOW = pd.Timedelta(hours=6)    # an alarm counts for a flare if it starts <= 6 h before onset (user decision)
FLARE_MAD_K = 3.0           # continuous flare signal: flaring = value > median + 3 * MAD (user decision)
FLAG_HINTS = ("flag", "label", "event", "flaring", "alarm")

ALPHAS = (0.05, 0.01)
NEAR_EMPTY_EDGES = 5        # fewer edges than this among the 26 PCs = "near-empty" (§3.3)
BOOT_B = 100
STABLE_FREQ = 0.7
KCI_N = 300                 # subsample size for the KCI variant (§3.3 b)
LAG_TAU = 2                 # lags 0..2 for the lagged diagnostic (§3.3 c)
MAX_STEP_SECONDS = 600      # §5: a step longer than 10 min is reduced
EVENT_GAP = 5               # samples; alarms closer than this are merged into one event
PLOT_MAX_POINTS = 20000     # longer time series are binned before plotting

EDGE_DIRECTED, EDGE_UNDIRECTED, EDGE_BIDIRECTED = "→", "—", "↔"
PCS = [f"PC{i + 1}" for i in range(K)]

# Fixed colour per unit (Okabe-Ito + Tol muted, best effort for 13 colour-blind-safe colours).
UNIT_COLORS = dict(zip(UNITS, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
                               "#F0E442", "#000000", "#332288", "#882255", "#44AA99", "#999933",
                               "#888888"]))
TRAIN_COLOR, TEST_COLOR, FLARE_COLOR = "#0072B2", "#E69F00", "#D55E00"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
                     "savefig.bbox": "tight"})

# ----------------------------------------------------------------------------- logging and gates
_T0 = time.time()


def log(msg=""):
    line = f"[{time.time() - _T0:8.1f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def stage(name):
    log(f"===== {name} | started {time.strftime('%Y-%m-%d %H:%M:%S')} =====")


def gate(ok, msg):
    """Stage gate (§6): log the check; on failure stop the whole pipeline."""
    log(f"CHECK {'PASS' if ok else 'FAIL'}: {msg}")
    if not ok:
        log("GATE FAILED - pipeline stopped. Send this log to the project owner before changing anything.")
        sys.exit(1)


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.4g}"
    return str(v)


def md_table(df):
    """Markdown table without the optional 'tabulate' dependency."""
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save_json(obj, name):
    (DATA / name).write_text(json.dumps(obj, indent=2, default=str))


def load_json(name):
    return json.loads((DATA / name).read_text())


# ----------------------------------------------------------------------------- raw data readers
def parquet_files(folder):
    return sorted(p for p in Path(folder).rglob("*.parquet") if p.is_file())


def read_table(path):
    """Read one parquet file; a non-default index (e.g. a DatetimeIndex) becomes a column."""
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    df.columns = [str(c) for c in df.columns]
    return df


def to_datetime_or_none(s):
    """Parse a column as timestamps (tz-aware -> UTC, naive); None if < 95 % parse as plausible times."""
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s)
    elif pd.api.types.is_numeric_dtype(s):
        out = None
        for unit in ("s", "ms"):            # epoch seconds / milliseconds
            cand = pd.to_datetime(s, unit=unit, errors="coerce")
            years = cand.dt.year.dropna()
            if len(years) and years.between(1990, 2100).mean() > 0.95:
                out = cand
                break
        if out is None:
            return None
    else:
        out = pd.to_datetime(s, errors="coerce")
    if out.notna().mean() < 0.95:
        return None
    if out.dt.tz is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def find_time_column(df):
    """(column name, parsed timestamps) of the timestamp column, or (None, None).
    datetime-typed columns are tried first, then columns whose name contains 'time'/'date'."""
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    hinted = [c for c in df.columns if c not in dt_cols and ("time" in c.lower() or "date" in c.lower())]
    for c in dt_cols + hinted:
        parsed = to_datetime_or_none(df[c])
        if parsed is not None:
            return c, parsed
    return None, None


def numeric_part(df):
    """Numeric tag columns of a raw table. Text columns that are mostly numbers (historian text such
    as 'Bad Input' mixed in) are coerced and the text becomes NaN.
    Returns (numeric frame, dropped non-numeric column names, {column: text values coerced})."""
    out, dropped, coerced = {}, [], {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            out[c] = s.astype("float64")
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            n_valid = int(s.notna().sum())
            if n_valid and num.notna().sum() >= 0.5 * n_valid:
                out[c] = num.astype("float64")
                coerced[c] = n_valid - int(num.notna().sum())
            else:
                dropped.append(c)
        else:
            dropped.append(c)
    return pd.DataFrame(out, index=df.index), dropped, coerced


def median_interval(index):
    ts = pd.DatetimeIndex(index).unique().sort_values()
    return ts.to_series().diff().median() if len(ts) > 1 else pd.NaT


def load_file(path):
    """One parquet file -> (numeric frame indexed by timestamp, info dict). Frame is None without a timestamp."""
    df = read_table(path)
    tcol, t = find_time_column(df)
    info = {"file": str(Path(path).relative_to(DATA_ROOT)), "rows": len(df), "columns": df.shape[1],
            "time_column": tcol, "time_dtype": str(df[tcol].dtype) if tcol else None}
    if tcol is None:
        return None, info
    num, dropped, coerced = numeric_part(df.drop(columns=[tcol]))
    num.index = pd.DatetimeIndex(t.to_numpy(), name="timestamp")
    num = num[~num.index.isna()]
    info.update({"numeric_columns": num.shape[1], "non_numeric_dropped": "; ".join(dropped),
                 "text_values_coerced": int(sum(coerced.values())),
                 "start": num.index.min(), "end": num.index.max(),
                 "median_interval": median_interval(num.index),
                 "duplicated_timestamps": int(num.index.duplicated().sum())})
    return num, info


def load_folder(folder):
    """All parquet files under a folder stacked in time (rows); columns are the union of the files."""
    frames, infos = [], []
    for p in parquet_files(folder):
        num, info = load_file(p)
        infos.append(info)
        if num is not None and num.shape[1]:
            frames.append(num)
    if not frames:
        return pd.DataFrame(), infos
    return pd.concat(frames, axis=0, sort=False).sort_index(kind="stable"), infos


def read_tag_map():
    return pd.read_csv(DATA / "tag_unit_map.csv", dtype=str, keep_default_na=False)


def read_flare_events():
    p = DATA / "flare_events.csv"
    if not p.exists():
        return pd.DataFrame({"event_id": pd.Series(dtype=int), "start": pd.Series(dtype="datetime64[ns]"),
                             "end": pd.Series(dtype="datetime64[ns]")})
    ev = pd.read_csv(p)
    ev["start"], ev["end"] = pd.to_datetime(ev["start"]), pd.to_datetime(ev["end"])   # datetime even when empty
    return ev


def present_units(columns, unit_of):
    """The units (in fixed UNITS order) that own at least one of the given columns."""
    have = {unit_of[c] for c in columns}
    return [u for u in UNITS if u in have]


# ----------------------------------------------------------------------------- shared analysis helpers
def flags_to_events(times, flag, max_gap):
    """Group flagged timestamps into events; flags at most max_gap apart are merged.
    Returns a frame with start, end, n_flagged."""
    t = pd.DatetimeIndex(times)[np.asarray(flag, dtype=bool)].sort_values()
    if len(t) == 0:
        return pd.DataFrame({"start": pd.Series(dtype="datetime64[ns]"),
                             "end": pd.Series(dtype="datetime64[ns]"), "n_flagged": pd.Series(dtype=int)})
    new = np.r_[True, np.diff(t.asi8) > pd.Timedelta(max_gap).value]
    g = pd.Series(t, index=np.cumsum(new))
    by = g.groupby(level=0)
    return pd.DataFrame({"start": by.min(), "end": by.max(), "n_flagged": by.size()}).reset_index(drop=True)


def loading_mass(P, columns, unit_of, units):
    """Share of each PC's squared loadings carried by each unit's tags (rows PCs, columns units, rows sum to 1)."""
    sq = pd.DataFrame(np.asarray(P) ** 2, index=columns, columns=[f"PC{i + 1}" for i in range(P.shape[1])])
    m = sq.groupby(pd.Series(columns, index=columns).map(unit_of)).sum().T
    return m.reindex(columns=units, fill_value=0.0)


# ----------------------------------------------------------------------------- plotting helpers
FIG_INDEX = []      # (figure name, takeaway) for RESULTS.md


def save_fig(fig, name, title, takeaway, table, index=False, bottom=0.0):
    """Title + one-line takeaway subtitle, 300 dpi PNG + vector PDF, and the plotted numbers as CSV.
    bottom reserves a figure fraction below the axes (for a figure-level legend)."""
    w, h = fig.get_size_inches()
    sub = textwrap.fill(takeaway, max(40, int(w * 11)))
    n_lines = sub.count("\n") + 1
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1 - 0.12 / h, va="top")
    fig.text(0.5, 1 - 0.45 / h, sub, ha="center", va="top", fontsize=11, style="italic")
    fig.tight_layout(rect=(0, bottom, 1, 1 - (0.75 + 0.2 * n_lines) / h))
    fig.savefig(FIG / f"{name}.png", dpi=300)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    table.to_csv(TAB / f"{name}.csv", index=index)
    FIG_INDEX.append((name, takeaway))
    log(f"  saved {name}")


def shade_flares(ax, events, label=True):
    for k, (s, e) in enumerate(zip(events["start"], events["end"])):
        ax.axvspan(s, e, color=FLARE_COLOR, alpha=0.25, lw=0,
                   label="flare event" if (label and k == 0) else None)


def unit_handles(units):
    return [Patch(facecolor=UNIT_COLORS.get(u, "#BBBBBB"), edgecolor="k", label=u) for u in units]


def bin_for_plot(df, how):
    """Bin a time-indexed frame to at most PLOT_MAX_POINTS bins (empty bins stay NaN so lines break).
    Returns (frame, description of the binning)."""
    if len(df) <= PLOT_MAX_POINTS:
        return df, "raw samples"
    span = df.index.max() - df.index.min()
    rule = max(pd.Timedelta(minutes=1), (span / PLOT_MAX_POINTS).ceil("min"))
    return df.resample(rule).agg(how), f"{int(rule.total_seconds() // 60)}-min bins, {how}"
