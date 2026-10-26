"""Shared settings, logging, raw-data readers and graph helpers for the causal PC + T² pipeline.

The raw-data readers and the missing-value helper are copied unchanged from the earlier pipeline
(pca_pc_t2/src/common.py and pca_pc_t2/src/s1_build_matrix.py), so both pipelines read the data the same way.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")    # Windows redirects default to cp1252, which has no '→'
except AttributeError:
    pass

# ----------------------------------------------------------------------------- paths (all from .env)
PROJECT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT / ".env")
_REQUIRED = ["DATA_ROOT", "UNIT_FOLDERS", "FLARE_FOLDER", "ALL_TAGS_FOLDER", "CONTROL_FOLDER", "OUT_DIR"]
_missing = [k for k in _REQUIRED if not os.environ.get(k, "").strip()]
if _missing:
    sys.exit(f"Set {', '.join(_missing)} in causal_t2/.env before running the pipeline.")


def _path(value):
    p = Path(value.strip().strip("'\""))
    return p if p.is_absolute() else (PROJECT / p).resolve()


DATA_ROOT = _path(os.environ["DATA_ROOT"])
UNIT_FOLDERS = [s.strip() for s in os.environ["UNIT_FOLDERS"].split(",") if s.strip()]
FLARE_FOLDER = os.environ["FLARE_FOLDER"].strip()
ALL_TAGS_FOLDER = os.environ["ALL_TAGS_FOLDER"].strip()
CONTROL_FOLDER = os.environ["CONTROL_FOLDER"].strip()
OUT = _path(os.environ["OUT_DIR"])

# Optional time window: only rows with WINDOW_START <= timestamp < WINDOW_END are read (empty = all data).
WINDOW_START = pd.Timestamp(os.environ["WINDOW_START"].strip()) if os.environ.get("WINDOW_START", "").strip() else None
WINDOW_END = pd.Timestamp(os.environ["WINDOW_END"].strip()) if os.environ.get("WINDOW_END", "").strip() else None
WINDOW_TEXT = (f"{WINDOW_START or 'start of data'} to {WINDOW_END or 'end of data'} (end excluded)"
               if WINDOW_START is not None or WINDOW_END is not None else "all data (no window)")


def in_window(times):
    """Boolean mask: timestamps inside [WINDOW_START, WINDOW_END)."""
    t = pd.DatetimeIndex(times)
    mask = np.ones(len(t), dtype=bool)
    if WINDOW_START is not None:
        mask &= t >= WINDOW_START
    if WINDOW_END is not None:
        mask &= t < WINDOW_END
    return mask
DATA = OUT / "data"           # cached intermediate files
TAB = OUT / "tables"
GRAPH = OUT / "graph"
FIG = OUT / "figures"
LOGS = OUT / "logs"
for _d in (DATA, TAB, GRAPH, FIG / "per_tag", LOGS):
    _d.mkdir(parents=True, exist_ok=True)
REGISTRY = OUT / "tag_registry.csv"

# ----------------------------------------------------------------------------- settings
# Reused unchanged from the earlier pipeline (pca_pc_t2/src/common.py), decisions of the project owner:
SENTINELS = [-999.0, -9999.0, -99999.0, 99999.0]   # -> NaN
FFILL_MAX_GAP = 5             # samples; only NaN runs this short are forward-filled
MAX_MISSING = 0.30            # drop tags with more missing than this after the fill
MIN_ROW_COMPLETE = 0.90       # keep rows with at least this share of tags observed
NEAR_CONSTANT_SHARE = 0.99    # near-constant = one value covers > 99 % of samples (s0_audit definition)
TRAIN_FRAC = 0.60             # reference (normal operation) = first 60 % of time minus flare windows +- buffer
FLARE_BUFFER = pd.Timedelta(hours=2)
FLARE_MAD_K = 3.0             # flaring = any flare signal > median + 3 * MAD
EVENT_GAP = 5                 # samples; flare flags closer than this are merged into one event
MATCH_WINDOW = pd.Timedelta(hours=6)   # an alarm counts for a flare if it starts <= 6 h before onset

# New in this pipeline:
PC_ALPHA = 0.01               # Fisher-Z significance level of the PC algorithm
PC_ROW_STEP = 10              # PC uses every 10th row (1-min samples are strongly autocorrelated)
PC_MAX_K = 3                  # largest conditioning-set size allowed
PC_TEST_BUDGET = 3e7          # upper bound on CI tests; the depth cap is lowered until the estimate fits
COLLINEAR_R = 0.999           # |r| at or above this = collinear pair, one tag dropped
EIG_MIN = 1e-6                # smallest eigenvalue allowed in the correlation matrix used by PC
ALPHA0 = 0.05                 # overall false-alarm rate: T² limit chi2_p(1 - ALPHA0); per tag Bonferroni ALPHA0 / p
SIGMA2_MIN = 1e-6             # floor for the conditional variance sigma²_j·PA(j) (numerical guard)
EVENT_DIAG_WINDOW = pd.Timedelta(minutes=30)  # rows after the first out-of-control T² used to rank tags per event
TOP_N = 15                    # tags shown per flare event
ZOOM_N = 5                    # root-cause tags with a zoomed graph view
PLOT_MAX_POINTS = 4000        # time series are binned to at most this many points before plotting
SEED = 42

EDGE_DIRECTED, EDGE_UNDIRECTED, EDGE_BIDIRECTED = "->", "--", "<->"

# ----------------------------------------------------------------------------- logging, gates, json
_T0 = time.time()


def log(msg=""):
    print(f"[{time.time() - _T0:8.1f}s] {msg}", flush=True)


def stage(name):
    log(f"===== {name} | started {time.strftime('%Y-%m-%d %H:%M:%S')} =====")


def gate(ok, msg):
    """Stage check (Definition of Done): log it; on failure stop with exit code 1."""
    log(f"CHECK {'PASS' if ok else 'FAIL'}: {msg}")
    if not ok:
        log("CHECK FAILED - stage stopped.")
        sys.exit(1)


def save_json(obj, name):
    (DATA / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _append_json_list(name, stage_name, msg):
    p = DATA / name
    items = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    items = [i for i in items if not (i["stage"] == stage_name and i["text"] == msg)]
    items.append({"stage": stage_name, "text": msg})
    p.write_text(json.dumps(items, indent=2), encoding="utf-8")


def assumption(stage_name, msg):
    """Log an assumption / choice; SUMMARY.md lists all of them."""
    log(f"ASSUMPTION: {msg}")
    _append_json_list("assumptions.json", stage_name, msg)


def deviation(stage_name, msg):
    """Log a fallback that departs from the plan; SUMMARY.md lists them under 'Deviations'."""
    log(f"DEVIATION: {msg}")
    _append_json_list("deviations.json", stage_name, msg)


def clear_notes(stage_name):
    """Drop this stage's old assumptions / deviations before a rerun."""
    for name in ("assumptions.json", "deviations.json"):
        p = DATA / name
        if p.exists():
            items = [i for i in json.loads(p.read_text(encoding="utf-8")) if i["stage"] != stage_name]
            p.write_text(json.dumps(items, indent=2), encoding="utf-8")


def done(stage_name, seconds):
    (DATA / f"{stage_name}.done").write_text(f"{seconds:.1f}\n", encoding="utf-8")


def read_registry():
    reg = pd.read_csv(REGISTRY, dtype=str, keep_default_na=False)
    reg["is_flare"] = reg["is_flare"].eq("True")
    return reg


def write_registry(reg):
    reg.to_csv(REGISTRY, index=False)


def safe_name(tag):
    """File-name-safe version of a tag (flare tags contain '/', '*', '(' ...)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(tag)).strip("_")[:120]


# ----------------------------------------------------------------------------- raw data readers (copied)
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
    """(column name, parsed timestamps) of the timestamp column, or (None, None)."""
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    hinted = [c for c in df.columns if c not in dt_cols and ("time" in c.lower() or "date" in c.lower())]
    for c in dt_cols + hinted:
        parsed = to_datetime_or_none(df[c])
        if parsed is not None:
            return c, parsed
    return None, None


def numeric_part(df):
    """Numeric tag columns of a raw table. Mostly-numeric text columns are coerced, the text becomes NaN.
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
    info = {"file": Path(path).relative_to(DATA_ROOT).as_posix(), "rows": len(df), "time_column": tcol}
    if tcol is None:
        return None, info
    num, dropped, coerced = numeric_part(df.drop(columns=[tcol]))
    num.index = pd.DatetimeIndex(t.to_numpy(), name="timestamp")
    num = num[~num.index.isna() & in_window(num.index)]
    info.update({"rows_in_window": len(num), "numeric_columns": num.shape[1], "non_numeric_dropped": "; ".join(dropped),
                 "text_values_coerced": int(sum(coerced.values())), "median_interval": median_interval(num.index)})
    return num, info


def flags_to_events(times, flag, max_gap):
    """Group flagged timestamps into events; flags at most max_gap apart are merged (start, end, n_flagged)."""
    t = pd.DatetimeIndex(times)[np.asarray(flag, dtype=bool)].sort_values()
    if len(t) == 0:
        return pd.DataFrame({"start": pd.Series(dtype="datetime64[ns]"),
                             "end": pd.Series(dtype="datetime64[ns]"), "n_flagged": pd.Series(dtype=int)})
    new = np.r_[True, np.diff(t.asi8) > pd.Timedelta(max_gap).value]
    g = pd.Series(t, index=np.cumsum(new))
    by = g.groupby(level=0)
    return pd.DataFrame({"start": by.min(), "end": by.max(), "n_flagged": by.size()}).reset_index(drop=True)


def ffill_short_gaps(df, max_gap):
    """Forward-fill only NaN runs of length <= max_gap that follow a valid value (longer gaps stay NaN).
    Returns (filled frame, histogram of NaN-run lengths, number of filled values).
    Copied from pca_pc_t2/src/s1_build_matrix.py."""
    out = df.copy()
    run_hist = pd.Series(dtype="float64")
    n_filled = 0
    for c in df.columns:
        na = df[c].isna()
        if not na.any():
            continue
        run_id = (~na).cumsum()                     # NaNs share the id of the last valid value before them
        lengths = na.groupby(run_id).sum()
        lengths = lengths[lengths > 0]
        run_hist = run_hist.add(lengths.value_counts(), fill_value=0)
        run_len = run_id.map(lengths).fillna(0)
        fill = na & (run_len <= max_gap) & (run_id > 0)
        out.loc[fill, c] = df[c].ffill()[fill]
        n_filled += int(fill.sum())
    run_hist.index = run_hist.index.astype(int)
    return out, run_hist.astype(int).sort_index(), n_filled


def read_flare_events():
    ev = pd.read_csv(DATA / "flare_events.csv")
    ev["start"], ev["end"] = pd.to_datetime(ev["start"]), pd.to_datetime(ev["end"])
    return ev


# ----------------------------------------------------------------------------- graph helpers
def edge_list(G, names):
    """Edges of a causal-learn graph matrix: G[i,j] = -1 and G[j,i] = 1 means i -> j;
    both -1 = undirected; both 1 = bidirected (conflicting orientations). Copied from pca_pc_t2 s3."""
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = G[i, j], G[j, i]
            if a == 0 and b == 0:
                continue
            if a == -1 and b == 1:
                rows.append((names[i], names[j], EDGE_DIRECTED))
            elif a == 1 and b == -1:
                rows.append((names[j], names[i], EDGE_DIRECTED))
            elif a == -1 and b == -1:
                rows.append((names[i], names[j], EDGE_UNDIRECTED))
            elif a == 1 and b == 1:
                rows.append((names[i], names[j], EDGE_BIDIRECTED))
            else:
                rows.append((names[i], names[j], f"other({a},{b})"))
    return pd.DataFrame(rows, columns=["source", "target", "type"])
