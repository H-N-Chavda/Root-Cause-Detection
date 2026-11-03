"""All logic of the pipeline: loading, cleaning, flare tags and events, unit-wise PCA, PC algorithm, DAG, T² and its
decomposition, figures, the root-cause table and the summary.

Data flow (main.py): raw parquet files -> joined 1-min matrix -> flare_net and flare events -> cleaned matrix ->
NOC-standardized matrix Z -> per-unit PCA scores -> PC algorithm on the scores -> DAG among PCs (+ every PC -> flare_net)
-> per-PC causal terms T_j·PA(j) and overall T² -> pre-flare windows. plots.py and summary.py only read saved outputs.
"""
from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import time
from math import comb
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import f as f_dist
from scipy.stats import norm

import config as C

EDGE_DIRECTED, EDGE_UNDIRECTED, EDGE_BIDIRECTED = "->", "--", "<->"
_T0 = time.time()
_LOG = {"file": None}


# ---- LOGGING AND CHECKS ----
def check_packages() -> None:
    """Stop the run with a pip command if a required Python package is not installed.

    Raises:
        SystemExit: If any of pandas, pyarrow, numpy, scipy, networkx, causal-learn, python-dotenv, scikit-learn or
            matplotlib is missing.
    """
    need = {"pandas": "pandas", "pyarrow": "pyarrow", "numpy": "numpy", "scipy": "scipy", "networkx": "networkx",
            "causallearn": "causal-learn", "dotenv": "python-dotenv", "sklearn": "scikit-learn", "matplotlib": "matplotlib"}
    missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        sys.exit("Missing packages. Install with:  pip install " + " ".join(missing))


def make_output_dirs(out: Path) -> dict[str, Path]:
    """Create the output sub-folders (tables, graph, t2, logs, figures) under the output folder.

    Args:
        out: Output folder, OUT_DIR from .env as resolved by config.load_paths().

    Returns:
        The sub-folder paths by name, e.g. {"tables": out/"tables", ...}.
    """
    dirs = {name: out / name for name in ("tables", "graph", "t2", "logs", "figures")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def start_log(path: Path) -> None:
    """Open a log file; from now on every log() line is written there as well as to the screen.

    Args:
        path: Log file to (over)write, e.g. outputs/logs/run.log.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    _LOG["file"] = open(path, "w", encoding="utf-8")


def log(msg: str = "") -> None:
    """Print a message with the seconds since start and append it to the open log file.

    Args:
        msg: Text to log.
    """
    line = f"[{time.time() - _T0:8.1f}s] {msg}"
    print(line, flush=True)
    if _LOG["file"] is not None:
        _LOG["file"].write(line + "\n")
        _LOG["file"].flush()


def gate(ok: bool, msg: str) -> None:
    """Log a pass/fail check and stop the run if it failed.

    Args:
        ok: Result of the check.
        msg: What was checked, written to the log either way.

    Raises:
        SystemExit: With exit code 1 if ok is False.
    """
    log(f"CHECK {'PASS' if ok else 'FAIL'}: {msg}")
    if not ok:
        log("CHECK FAILED - run stopped.")
        sys.exit(1)


# ---- LOADING ----
def norm_unit(name: str) -> str:
    """Unit-name key that ignores case and punctuation, so 'SEU-1' and 'SEU1' match.

    Args:
        name: A unit name or any text (e.g. a file path).

    Returns:
        The text upper-cased with everything except A-Z and 0-9 removed.
    """
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def unit_from_text(text: str, unit_folders: list[str]) -> str | None:
    """Find which unit a file belongs to from its path; the longest match wins (ISOM_3 over ISOM).

    Args:
        text: File path relative to DATA_ROOT, used for tags in the control / all-tags folders.
        unit_folders: Unit folder names from UNIT_FOLDERS in .env.

    Returns:
        The matching unit name, or None if no unit name appears in the path.
    """
    key = norm_unit(text)
    hits = [u for u in unit_folders if norm_unit(u) in key]
    return max(hits, key=len) if hits else None


def tag_key(name: str) -> str:
    """Key used to spot the same tag in two folders: tag name without spaces, upper-cased.

    Args:
        name: Column name of a tag in a raw parquet file.

    Returns:
        The normalized key.
    """
    return re.sub(r"\s+", "", str(name)).upper()


def in_window(times: Any) -> np.ndarray:
    """Mark which timestamps fall inside the analysis window [WINDOW_START, WINDOW_END) from config.py.

    Args:
        times: Timestamps of one raw file (anything pd.DatetimeIndex accepts).

    Returns:
        Boolean array, True for timestamps inside the window (all True when the window is unset).
    """
    start, end = C.window_bounds()
    t = pd.DatetimeIndex(times)
    mask = np.ones(len(t), dtype=bool)
    if start is not None:
        mask &= t >= start
    if end is not None:
        mask &= t < end
    return mask


def window_text() -> str:
    """Readable description of the analysis window from config.py, for logs and the summary.

    Returns:
        E.g. '2025-07-01 00:00:00 to 2025-11-01 00:00:00 (end excluded)'.
    """
    start, end = C.window_bounds()
    if start is None and end is None:
        return "all data (no window)"
    return f"{start or 'start of data'} to {end or 'end of data'} (end excluded)"


def parquet_files(folder: Path) -> list[Path]:
    """List every parquet file below a data folder.

    Args:
        folder: A folder under DATA_ROOT, e.g. DATA_ROOT/LOUP.

    Returns:
        Sorted list of parquet file paths (sub-folders included).
    """
    return sorted(p for p in Path(folder).rglob("*.parquet") if p.is_file())


def read_table(path: Path) -> pd.DataFrame:
    """Read one raw parquet file; a timestamp kept in the index becomes an ordinary column.

    Args:
        path: Raw parquet file of one tag (or a few tags) from the plant data.

    Returns:
        The table with string column names and a plain 0..n-1 index.
    """
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    df.columns = [str(c) for c in df.columns]
    return df


def to_datetime_or_none(s: pd.Series) -> tuple[pd.Series | None, str]:
    """Try to read a column as timestamps (datetime, epoch seconds/milliseconds or text).

    Args:
        s: One column of a raw table that may hold the timestamps.

    Returns:
        A tuple (timestamps, tz): timestamps as naive UTC datetimes (tz-aware values are converted to UTC), or None
        when fewer than 95 % of the values parse as plausible times; tz is the original time zone name, '' if naive.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s)
    elif pd.api.types.is_numeric_dtype(s):
        out = None
        for unit in ("s", "ms"):
            cand = pd.to_datetime(s, unit=unit, errors="coerce")
            years = cand.dt.year.dropna()
            if len(years) and years.between(1990, 2100).mean() > 0.95:
                out = cand
                break
        if out is None:
            return None, ""
    else:
        out = pd.to_datetime(s, errors="coerce")
    if out.notna().mean() < 0.95:
        return None, ""
    tz = ""
    if out.dt.tz is not None:
        tz = str(out.dt.tz)
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out, tz


def find_time_column(df: pd.DataFrame) -> tuple[str | None, pd.Series | None, str]:
    """Find the timestamp column of a raw table: datetime columns first, then columns named like 'time' or 'date'.

    Args:
        df: Raw table from read_table().

    Returns:
        A tuple (column name, parsed timestamps, original time zone), or (None, None, '') if no column parses.
    """
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    hinted = [c for c in df.columns if c not in dt_cols and ("time" in c.lower() or "date" in c.lower())]
    for c in dt_cols + hinted:
        parsed, tz = to_datetime_or_none(df[c])
        if parsed is not None:
            return c, parsed, tz
    return None, None, ""


def numeric_part(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Keep the numeric tag columns of a raw table; text columns that are mostly numbers are converted.

    Args:
        df: Raw table without its timestamp column.

    Returns:
        A tuple (numeric table as float64, names of dropped non-numeric columns, {column: number of text values that
        became NaN}).
    """
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


def median_interval(index: Any) -> pd.Timedelta:
    """Typical sampling interval of one file: the median gap between its distinct timestamps.

    Args:
        index: Timestamps of one raw file inside the window.

    Returns:
        The median gap, or NaT if the file has fewer than two timestamps.
    """
    ts = pd.DatetimeIndex(index).unique().sort_values()
    return ts.to_series().diff().median() if len(ts) > 1 else pd.NaT


def to_ns_index(values: Any) -> pd.DatetimeIndex:
    """Build a DatetimeIndex in nanoseconds, so every time difference in the pipeline has the same unit.

    Mixed units (µs from parquet, ns from Timedelta) once merged the whole of May into one flare event.

    Args:
        values: Timestamps in any resolution.

    Returns:
        A DatetimeIndex named 'timestamp' with nanosecond resolution.
    """
    idx = pd.DatetimeIndex(values, name="timestamp")
    return idx.as_unit("ns") if hasattr(idx, "as_unit") else idx


def load_file(path: Path, data_root: Path) -> tuple[pd.DataFrame | None, dict]:
    """Read one raw parquet file into a numeric table indexed by time, keeping only rows inside the window.

    Args:
        path: Raw parquet file under DATA_ROOT.
        data_root: DATA_ROOT from .env (used to report the file path relative to it).

    Returns:
        A tuple (table, info): table has one float column per tag and a nanosecond DatetimeIndex, or None if the file
        has no timestamp column; info holds the file name, row counts, time zone and median sampling interval.
    """
    df = read_table(path)
    tcol, t, tz = find_time_column(df)
    info = {"file": Path(path).relative_to(data_root).as_posix(), "rows": len(df), "time_column": tcol, "tz": tz}
    if tcol is None:
        return None, info
    num, _, _ = numeric_part(df.drop(columns=[tcol]))
    num.index = to_ns_index(t.to_numpy())
    num = num[~num.index.isna() & in_window(num.index)]
    info.update({"rows_in_window": len(num), "median_interval": median_interval(num.index)})
    return num, info


def load_all_tags(P: SimpleNamespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[pd.Timedelta]]:
    """Load every tag of the flare folder, the 13 unit folders and the optional extra folders, and build the registry.

    A tag found in two folders is kept from the first folder in the order flare -> units -> control -> all-tags.

    Args:
        P: Paths and names from config.load_paths() (.env): data_root, unit_folders, flare_folder, control_folder,
            all_tags_folder.

    Returns:
        A tuple (registry, folder_frames, intervals): registry has one row per tag with columns tag, source_folder,
        unit, is_flare, file; folder_frames maps each folder to its raw tags (float32, original timestamps);
        intervals lists each file's median sampling interval.

    Raises:
        SystemExit: If DATA_ROOT, the unit folders, the flare folder or any data inside the window is missing.
    """
    gate(P.data_root.is_dir(), "DATA_ROOT in .env is an existing folder")
    log(f"time window: {window_text()}")
    present = sorted(p.name for p in P.data_root.iterdir() if p.is_dir())
    units = [u for u in P.unit_folders if u in present]
    log(f"unit folders found: {len(units)} of {len(P.unit_folders)}; missing: "
        f"{[u for u in P.unit_folders if u not in present] or 'none'}")
    gate(len(units) > 0, "at least one unit folder exists")
    gate(P.flare_folder in present, f"flare folder '{P.flare_folder}' exists")
    sources = [P.flare_folder] + units + [f for f in (P.control_folder, P.all_tags_folder) if f in present]
    reg_rows, dup_rows, intervals, folder_frames, seen, zones = [], [], [], {}, {}, set()
    for folder in sources:
        parts, files = {}, parquet_files(P.data_root / folder)
        for p in files:
            num, info = load_file(p, P.data_root)
            if num is None:
                log(f"  skipped {info['file']}: no timestamp column")
                continue
            zones.add(info["tz"] or "naive")
            if pd.notna(info["median_interval"]):
                intervals.append(info["median_interval"])
            for c in num.columns:
                k = tag_key(c)
                if k in seen and seen[k][0] != folder:
                    dup_rows.append({"tag": c, "duplicate_file": info["file"], "kept_folder": seen[k][0]})
                    continue
                if k not in seen:
                    seen[k] = (folder, c)
                    unit = folder if folder == P.flare_folder or folder in units else (
                        unit_from_text(info["file"], P.unit_folders) or folder)
                    reg_rows.append({"tag": c, "source_folder": folder, "unit": unit,
                                     "is_flare": folder == P.flare_folder, "file": info["file"]})
                parts.setdefault(seen[k][1], []).append(num[c].astype("float32"))
            del num
        if parts:
            cols = {}
            for c, ps in parts.items():
                s = pd.concat(ps).sort_index(kind="stable")
                cols[c] = s[~s.index.duplicated(keep="last")]
            folder_frames[folder] = pd.DataFrame(cols)
        log(f"  {folder}: {len(files)} files, {len(parts)} new tags")
    log(f"tags loaded: {len(reg_rows)} unique, {len(dup_rows)} duplicate copies skipped; "
        f"timestamp time zones in the files: {sorted(zones)} (tz-aware stamps are converted to UTC)")
    gate(len(reg_rows) > 0 and len(intervals) > 0, "at least one timestamped tag was loaded")
    gate(any(len(f) for f in folder_frames.values()), f"the time window ({window_text()}) contains data rows")
    order = {f: i for i, f in enumerate(units + [P.control_folder, P.all_tags_folder, P.flare_folder])}
    reg = pd.DataFrame(reg_rows)
    reg = reg.assign(_o=reg["source_folder"].map(order)).sort_values(["_o", "unit", "tag"], kind="stable")
    reg = reg.drop(columns="_o").reset_index(drop=True)
    gate(reg["is_flare"].any(), "at least one flare tag was loaded")
    return reg, folder_frames, intervals


def join_and_resample(folder_frames: dict[str, pd.DataFrame], intervals: list[pd.Timedelta],
                      reg: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timedelta]:
    """Put every tag on one common time grid: sentinels -> NaN, resample to the most common interval, join on time.

    Args:
        folder_frames: Raw tags per folder from load_all_tags() (emptied by this function to save memory).
        intervals: Median sampling interval of each file, from load_all_tags().
        reg: Tag registry from load_all_tags(); its tag order becomes the column order.

    Returns:
        A tuple (X, interval): X is the joined matrix (rows = grid timestamps, columns = tags, float32, NaN where a tag
        has no value); interval is the grid step (1 min for this plant).

    Raises:
        SystemExit: If no positive sampling interval could be measured.
    """
    iv = pd.to_timedelta(pd.Series(intervals)).dt.round("1s")
    counts = iv[iv > pd.Timedelta(0)].value_counts()
    gate(len(counts) > 0, "a positive sampling interval could be measured")
    interval = max(counts[counts == counts.max()].index)
    frames, n_sent = [], 0
    for folder in list(folder_frames):
        f = folder_frames.pop(folder).replace([np.inf, -np.inf], np.nan)
        is_sent = f.isin(C.SENTINELS)
        n_sent += int(is_sent.to_numpy().sum())
        frames.append(f.mask(is_sent).resample(interval).mean().astype("float32"))
    X = pd.concat(frames, axis=1).sort_index()
    X = X[list(reg["tag"])]
    X.index = to_ns_index(X.index)
    log(f"joined matrix {X.shape} on a {interval} grid (bin mean); sentinel values set to NaN: {n_sent}")
    return X, interval


# ---- FLARE TAG ----
def build_flare_net(X: pd.DataFrame, reg: pd.DataFrame,
                    P: SimpleNamespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, str]]:
    """Replace all flare tags by one column flare_net = flare_1 + flare_2 - recovery (t/h).

    Args:
        X: Joined matrix from join_and_resample().
        reg: Tag registry from load_all_tags().
        P: Paths and names from config.load_paths(); flare_1_tag, flare_2_tag and flare_recovery_tag are the exact
            flare_data column names from .env.

    Returns:
        A tuple (X, reg, net, status): X without the raw flare tags and with a flare_net column; reg with a flare_net
        row added; net is the raw flare_net series on the full grid (used to find flare events); status records what
        happened to each raw flare tag.

    Raises:
        SystemExit: If a formula tag is not a flare tag or a tag is already called flare_net.
    """
    formula = [P.flare_1_tag, P.flare_2_tag, P.flare_recovery_tag]
    flare_loaded = list(reg.loc[reg["is_flare"], "tag"])
    gate(all(t in flare_loaded for t in formula), f"flare formula tags are flare tags: {formula} (loaded: {flare_loaded})")
    gate(C.FLARE_NET not in X.columns, f"no loaded tag is already named '{C.FLARE_NET}'")
    for t in formula:
        log(f"  {t}: {100 * X[t].isna().mean():.1f}% missing on the grid, median {X[t].median():.3f}")
    rec = X[P.flare_recovery_tag]
    if C.FLARE_RECOVERY_MISSING == "zero":
        rec = rec.fillna(0.0)
    net = (X[P.flare_1_tag] + X[P.flare_2_tag] - rec).astype("float32").rename(C.FLARE_NET)
    formula_text = f"{C.FLARE_NET} = {P.flare_1_tag} + {P.flare_2_tag} - {P.flare_recovery_tag}"
    log(f"{formula_text}; recovery with no value: {C.FLARE_RECOVERY_MISSING}; {100 * net.isna().mean():.1f}% missing, "
        f"median {net.median():.3f}; flare tags removed from the matrix: {flare_loaded}")
    X = X.drop(columns=flare_loaded)
    X[C.FLARE_NET] = net
    reg = pd.concat([reg, pd.DataFrame([{"tag": C.FLARE_NET, "source_folder": P.flare_folder, "unit": P.flare_folder,
                                         "is_flare": True, "file": formula_text}])], ignore_index=True)
    status = {t: "used_in_flare_net" if t in formula else "not used: flare tag outside the flare_net formula"
              for t in flare_loaded}
    return X, reg, net, status


# ---- FLARE EVENTS ----
def detect_flare_events(net: pd.Series, interval: pd.Timedelta) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Find flare events: minutes with flare_net above the threshold, merged into runs, kept if they last 30 min.

    Minutes above the threshold at most EVENT_GAP samples apart form one run. A run is a flare event only if it lasts
    at least FLARE_MIN_DURATION_MIN; shorter runs are returned separately (kept out of NOC, but not events).

    Args:
        net: Raw flare_net (t/h) on the full grid, from build_flare_net() (before any cleaning).
        interval: Grid step from join_and_resample().

    Returns:
        A tuple (events, short, thr). events has one row per flare event: event_id (1, 2, ... in time order), start
        (onset = first minute above the threshold), end (last such minute), n_flagged, peak_flare_net, duration_min.
        short has the same columns without event_id for runs shorter than FLARE_MIN_DURATION_MIN. thr is the
        threshold in t/h: FLARE_THRESHOLD, or median + FLARE_MAD_K * 1.4826 * MAD when FLARE_USE_MAD_THRESHOLD.
    """
    med = float(net.median())
    mad = 1.4826 * float((net - med).abs().median())
    thr = med + C.FLARE_MAD_K * mad if C.FLARE_USE_MAD_THRESHOLD else float(C.FLARE_THRESHOLD)
    flagged = (net > thr).to_numpy()
    t = pd.DatetimeIndex(net.index)[flagged]
    v = net.to_numpy()[flagged]
    max_gap = (C.EVENT_GAP * interval).to_timedelta64()
    new = np.r_[True, np.diff(t.values) > max_gap] if len(t) else np.zeros(0, dtype=bool)
    g = pd.DataFrame({"t": t, "v": v, "eid": np.cumsum(new)}).groupby("eid")
    runs = pd.DataFrame({"start": g["t"].min(), "end": g["t"].max(), "n_flagged": g.size(),
                         "peak_flare_net": g["v"].max()}).reset_index(drop=True)
    runs["duration_min"] = (runs["end"] - runs["start"] + interval) / pd.Timedelta(minutes=1)
    long = (runs["duration_min"] >= C.FLARE_MIN_DURATION_MIN).to_numpy()
    events, short = runs[long].reset_index(drop=True), runs[~long].reset_index(drop=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    log(f"flare threshold {thr:.3f} ({'median + K x MAD' if C.FLARE_USE_MAD_THRESHOLD else 'FLARE_THRESHOLD, manual'}; "
        f"median {med:.3f}, MAD {mad:.3f}); flagged rows {100 * flagged.mean():.2f}%; runs above it: {len(runs)}, of "
        f"which {len(events)} last >= {C.FLARE_MIN_DURATION_MIN} min (flare events) and {len(short)} are shorter")
    return events, short, thr


def crosscheck_flare_events_ta(events: pd.DataFrame) -> None:
    """Compare the events found in the TA's period (TA_CHECK_START..TA_CHECK_END) with the TA's account; logs only.

    Args:
        events: Flare events from detect_flare_events().
    """
    start, end = pd.Timestamp(C.TA_CHECK_START), pd.Timestamp(C.TA_CHECK_END)
    ws, we = C.window_bounds()
    if (ws is not None and end <= ws) or (we is not None and start >= we):
        log(f"TA cross-check skipped: the TA's period [{start:%Y-%m-%d}, {end:%Y-%m-%d}) is outside the time window "
            f"({window_text()})")
        return
    sel = events[(events["start"] >= start) & (events["start"] < end)]
    log(f"flare events with onset in [{start:%Y-%m-%d}, {end:%Y-%m-%d}): {len(sel)}")
    for ev in sel.itertuples(index=False):
        log(f"  event {ev.event_id}: onset {ev.start}, end {ev.end}, {ev.duration_min:.0f} min, "
            f"{ev.n_flagged} flagged rows, peak flare_net {ev.peak_flare_net:.3f}")
    same = len(sel) == C.TA_FLARE_COUNT and bool((sel["duration_min"] <= 3 * C.TA_FLARE_MINUTES).all())
    total = float(sel["duration_min"].sum()) if len(sel) else 0.0
    log(f"TA CROSS-CHECK {'MATCH' if same else 'MISMATCH'}: data give {len(sel)} events, {total:.0f} event minutes in "
        f"total; TA said {C.TA_FLARE_COUNT} flares of about {C.TA_FLARE_MINUTES} min"
        + ("" if same else " -> report this difference to the TA"))


def label_flare_rows(idx: pd.DatetimeIndex, events: pd.DataFrame, short: pd.DataFrame, interval: pd.Timedelta,
                     n_tags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Label each row of the cleaned matrix: flare event, short flare, pre-flare window, or NOC (normal operation).

    NOC = rows in the first TRAIN_FRAC of the time span, outside every flare event ± FLARE_BUFFER_MIN and outside
    every short flare. If that leaves fewer than 5 x n_tags rows, the buffer is dropped (logged as a deviation).

    Args:
        idx: Timestamps of the cleaned matrix, from clean_matrix().
        events: Flare events (>= 30 min) from detect_flare_events().
        short: Shorter runs above the threshold from detect_flare_events().
        interval: Grid step from join_and_resample().
        n_tags: Number of columns of the cleaned matrix (sets the minimum NOC size).

    Returns:
        A tuple (flare, short_flare, pre_eid, noc) of arrays aligned with idx: flare and short_flare are booleans for
        rows inside a flare event / short flare; pre_eid is the event id of the 30-min pre-flare window a row belongs
        to (0 = none); noc is True for baseline rows.

    Raises:
        SystemExit: If NOC is still too small, or a NOC row lies inside a flare, short flare or pre-flare window.
    """
    flare = np.zeros(len(idx), dtype=bool)
    short_flare = np.zeros(len(idx), dtype=bool)
    near_flare = np.zeros(len(idx), dtype=bool)
    buffer = pd.Timedelta(minutes=C.FLARE_BUFFER_MIN)
    for s, e in zip(events["start"], events["end"]):
        flare |= (idx > s - interval) & (idx <= e)
        near_flare |= (idx > s - interval - buffer) & (idx <= e + buffer)
    for s, e in zip(short["start"], short["end"]):
        short_flare |= (idx > s - interval) & (idx <= e)
    pre_eid = extract_preflare_window(idx, events, interval, flare)
    cut = idx.min() + C.TRAIN_FRAC * (idx.max() - idx.min())
    noc = (idx < cut) & ~near_flare & ~short_flare
    rule = f"NOC = rows before {cut} (first {C.TRAIN_FRAC:.0%} of time) minus flare events ± {buffer} and short flares"
    min_noc = 5 * n_tags
    if noc.sum() < min_noc:
        noc = (idx < cut) & ~flare & ~short_flare & (pre_eid == 0)
        rule = f"NOC = rows before {cut} minus flare-event rows, short flares and pre-flare windows (no buffer)"
        log(f"DEVIATION: NOC with the ± {buffer} buffer had < {min_noc} rows (5 x tags); fell back to: {rule}")
    log(f"{rule}; flare-event rows {100 * flare.mean():.2f}%, short-flare rows {100 * short_flare.mean():.2f}%, "
        f"pre-flare rows {int((pre_eid > 0).sum())}, NOC rows {int(noc.sum())}")
    gate(noc.sum() >= min_noc, f"NOC rows {int(noc.sum())} >= {min_noc} (5 x tags)")
    gate(not (noc & (flare | short_flare | (pre_eid > 0))).any(),
         "no NOC row lies inside a flare event, a short flare or a pre-flare window")
    return flare, short_flare, pre_eid, noc


def preflare_window_length(interval: pd.Timedelta) -> int:
    """Number of grid samples in FLARE_PRE_WINDOW_MIN minutes (30 min -> 30 samples on a 1-min grid).

    Args:
        interval: Grid step from join_and_resample().

    Returns:
        Window length in samples.

    Raises:
        SystemExit: If the window is shorter than one sample.
    """
    n = int(round(pd.Timedelta(minutes=C.FLARE_PRE_WINDOW_MIN) / interval))
    gate(n >= 1, f"pre-flare window {C.FLARE_PRE_WINDOW_MIN} min = {n} samples of {interval}")
    return n


def extract_preflare_window(idx: pd.DatetimeIndex, events: pd.DataFrame, interval: pd.Timedelta,
                            flare: np.ndarray) -> np.ndarray:
    """Label the rows in [onset - 30 min, onset) of each flare event with that event's id.

    Rows inside any flare event are left out; a row in two windows keeps the earlier event's id.

    Args:
        idx: Timestamps of the cleaned matrix.
        events: Flare events from detect_flare_events().
        interval: Grid step from join_and_resample().
        flare: Flare-event rows from label_flare_rows().

    Returns:
        Integer array aligned with idx: event id of the pre-flare window a row belongs to, 0 if none.
    """
    n = preflare_window_length(interval)
    span = n * interval
    eid = np.zeros(len(idx), dtype=int)
    for ev in events.itertuples(index=False):
        w = (idx >= ev.start - span) & (idx < ev.start)
        n_in_flare, n_shared = int((w & flare).sum()), int((w & ~flare & (eid > 0)).sum())
        w &= ~flare & (eid == 0)
        eid[w] = ev.event_id
        if int(w.sum()) != n:
            log(f"  pre-flare window of event {ev.event_id}: {int(w.sum())} of {n} rows "
                f"({n_in_flare} inside a flare event, {n_shared} already in an earlier event's window, rest not on the grid)")
    return eid


def check_flare_source_agreement(flare_z: pd.Series, mu: float, sd: float, net: pd.Series, thr: float,
                                 flagged_rows: np.ndarray, out_csv: Path) -> None:
    """Check that flare_net in the cleaned matrix and the raw flare_net that defined the events are the same series.

    Args:
        flare_z: NOC-standardized flare_net column of Z, from standardize_with_noc().
        mu: NOC mean of flare_net used for that standardization (t/h).
        sd: NOC std of flare_net used for that standardization (t/h).
        net: Raw flare_net from build_flare_net().
        thr: Flare threshold from detect_flare_events().
        flagged_rows: Rows inside a flare event or a short flare, from label_flare_rows().
        out_csv: File for the mismatching rows (empty table if there are none).

    Raises:
        SystemExit: If any value, threshold flag or label disagrees.
    """
    gate(flare_z.index.isin(net.index).all(), "every matrix timestamp exists on the flare_net grid used for the events")
    raw = net.reindex(flare_z.index).to_numpy(dtype=np.float64)
    back = flare_z.to_numpy(dtype=np.float64) * sd + mu
    tol = 1e-4 * (1.0 + np.abs(np.nan_to_num(raw)))
    filled = np.isnan(raw)
    value_bad = ~filled & (np.abs(back - raw) > tol)
    flag_bad = ~filled & (np.abs(raw - thr) > tol) & ((back > thr) != (raw > thr))
    label_bad = ~filled & (raw > thr) & ~flagged_rows
    filled_flagged = filled & (back > thr)
    bad = value_bad | flag_bad | label_bad | filled_flagged
    rows = pd.DataFrame({"timestamp": flare_z.index[bad], "flare_net_events": raw[bad], "flare_net_matrix": back[bad],
                         "value_mismatch": value_bad[bad], "flag_mismatch": flag_bad[bad],
                         "flagged_outside_flare": label_bad[bad], "filled_value_above_threshold": filled_flagged[bad]})
    rows.to_csv(out_csv, index=False)
    for r in rows.head(20).itertuples(index=False):
        log(f"  MISMATCH {r.timestamp}: events {r.flare_net_events}, matrix {r.flare_net_matrix}")
    log(f"flare_net agreement: {int(filled.sum())} matrix rows filled by the short-gap forward fill (no raw value)")
    gate(not bad.any(), f"flare_net in the matrix equals the flare_net used for the events ({int(bad.sum())} mismatches)")


# ---- CLEANING ----
def ffill_short_gaps(df: pd.DataFrame, max_gap: int) -> tuple[pd.DataFrame, int]:
    """Forward-fill gaps of at most max_gap missing samples that follow a valid value; longer gaps stay NaN.

    Args:
        df: Joined matrix (rows = grid timestamps, columns = tags).
        max_gap: Longest run of NaN to fill, in samples (FFILL_MAX_GAP).

    Returns:
        A tuple (filled matrix, number of values filled).
    """
    out = df.copy()
    n_filled = 0
    for c in df.columns:
        na = df[c].isna()
        if not na.any():
            continue
        run_id = (~na).cumsum()
        lengths = na.groupby(run_id).sum()
        lengths = lengths[lengths > 0]
        run_len = run_id.map(lengths).fillna(0)
        fill = na & (run_len <= max_gap) & (run_id > 0)
        out.loc[fill, c] = df[c].ffill()[fill]
        n_filled += int(fill.sum())
    return out, n_filled


def clean_matrix(X: pd.DataFrame, reg: pd.DataFrame, status: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Clean the joined matrix; flare_net is never dropped.

    Steps: fill short gaps; drop tags with more than MAX_MISSING missing; drop rows without flare_net; drop constant
    and near-constant tags; drop rows with fewer than MIN_ROW_COMPLETE of tags observed; drop duplicate columns.

    Args:
        X: Matrix with flare_net, from build_flare_net().
        reg: Tag registry from build_flare_net() (is_flare protects flare_net).
        status: Per-tag status so far; drop reasons are added here.

    Returns:
        A tuple (cleaned matrix, updated status).

    Raises:
        SystemExit: If 1000 rows or fewer remain, or flare_net was lost.
    """
    is_flare = dict(zip(reg["tag"], reg["is_flare"]))

    def drop(frame: pd.DataFrame, cols: Any, reason: Any) -> pd.DataFrame:
        """Drop columns from frame, logging reason(c) and recording it in status."""
        for c in cols:
            status[c] = f"dropped: {reason(c)}"
            log(f"  drop {c}: {reason(c)}")
        return frame.drop(columns=list(cols))

    X, n_filled = ffill_short_gaps(X, C.FFILL_MAX_GAP)
    log(f"forward-filled {n_filled} values (NaN runs <= {C.FFILL_MAX_GAP} samples)")
    miss = X.isna().mean()
    X = drop(X, [c for c in miss[miss > C.MAX_MISSING].index if c != C.FLARE_NET],
             lambda c: f"{100 * miss[c]:.1f}% missing after ffill (> {C.MAX_MISSING:.0%})")
    n0 = len(X)
    X = X[X[C.FLARE_NET].notna()]
    log(f"rows without a {C.FLARE_NET} value dropped: {n0 - len(X)} of {n0}")
    nun = X.nunique()
    X = drop(X, nun[nun <= 1].index, lambda c: "constant")
    top_share = X.apply(lambda s: s.value_counts(normalize=True).iloc[0])
    near = [c for c in X.columns if top_share[c] > C.NEAR_CONSTANT_SHARE]
    X = drop(X, [c for c in near if not is_flare[c]],
             lambda c: f"near-constant ({100 * top_share[c]:.2f}% one value > {C.NEAR_CONSTANT_SHARE:.0%})")
    n0 = len(X)
    X = X[X.notna().mean(axis=1) >= C.MIN_ROW_COMPLETE]
    log(f"rows with < {C.MIN_ROW_COMPLETE:.0%} of tags observed dropped: {n0 - len(X)} of {n0}")
    gate(len(X) > 1000, f"{len(X)} rows remain after the row filter")
    fp = pd.Series({c: int(pd.util.hash_pandas_object(X[c], index=False).sum()) for c in X.columns})
    dups = {}
    for _, cols in fp.groupby(fp).groups.items():
        cols = sorted(cols, key=lambda c: (not is_flare[c], list(X.columns).index(c)))
        for c in cols[1:]:
            if X[c].equals(X[cols[0]]):
                dups[c] = cols[0]
    X = drop(X, list(dups), lambda c: f"duplicate column of {dups[c]}")
    gate(C.FLARE_NET in X.columns, f"{C.FLARE_NET} is still in the matrix")
    return X, status


def standardize_with_noc(X: pd.DataFrame, reg: pd.DataFrame, noc: np.ndarray,
                         status: dict[str, str]) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, str], list[str]]:
    """Z-score every tag with its NOC mean and std; leftover NaN become 0 (= the NOC mean).

    Tags constant within NOC cannot be standardized and are dropped, except flare_net, which is then standardized
    with all rows.

    Args:
        X: Cleaned matrix from clean_matrix().
        reg: Tag registry (is_flare).
        noc: NOC rows from label_flare_rows().
        status: Per-tag status; tags kept get 'in_matrix'.

    Returns:
        A tuple (Z, mu, sd, status, pc_only): Z is the standardized matrix (float32); mu and sd are the means / stds
        used per tag; pc_only lists flare tags standardized with all rows.

    Raises:
        SystemExit: If Z has a NaN/inf or a constant column.
    """
    is_flare = dict(zip(reg["tag"], reg["is_flare"]))
    nun = X[noc].nunique()
    const = list(nun[nun <= 1].index)
    pc_only = [c for c in const if is_flare[c]]
    for c in const:
        if not is_flare[c]:
            status[c] = "dropped: constant within NOC"
            log(f"  drop {c}: constant within NOC")
    X = X.drop(columns=[c for c in const if not is_flare[c]])
    mu, sd = X[noc].mean(), X[noc].std()
    for c in pc_only:
        mu[c], sd[c] = X[c].mean(), X[c].std()
        log(f"  {c} is constant within NOC: standardized with all rows")
    n_imputed = int(X.isna().to_numpy().sum())
    for c in X.columns:
        X[c] = ((X[c] - mu[c]) / sd[c]).fillna(0.0).astype("float32")
        status.setdefault(c, "in_matrix")
    arr = X.to_numpy()
    log(f"NOC-mean imputed {n_imputed} values = {100 * n_imputed / arr.size:.2f}% of the matrix; matrix {X.shape}")
    gate(bool(np.isfinite(arr).all()), "standardized matrix has no NaN / inf")
    gate(bool((arr.std(axis=0) > 0).all()), "standardized matrix has no constant column")
    return X, mu, sd, status, pc_only


# ---- UNIT-WISE PCA ----
def choose_n_pcs(evr: np.ndarray, ev: np.ndarray) -> tuple[int, int]:
    """Number of PCs to keep for one unit.

    Uses N_PCS_PER_UNIT, or the fewest PCs reaching VAR_EXPLAINED_THRESHOLD when that is set; never more than the
    unit's numerical rank (eigenvalues above EIG_MIN x the largest).

    Args:
        evr: Explained-variance ratio of each component of the unit's PCA (descending).
        ev: Explained variance (eigenvalue) of each component.

    Returns:
        A tuple (PCs to keep, numerical rank).
    """
    if C.VAR_EXPLAINED_THRESHOLD is None:
        k = C.N_PCS_PER_UNIT
    else:
        k = int(np.searchsorted(np.cumsum(evr), C.VAR_EXPLAINED_THRESHOLD) + 1)
    rank = int((ev > C.EIG_MIN * ev[0]).sum())
    return min(k, len(evr), rank), rank


def fit_unit_pca(Z: pd.DataFrame, reg: pd.DataFrame, unit_folders: list[str],
                 noc: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one PCA per unit on that unit's NOC rows and return the kept PC scores for all rows.

    flare_net is not part of any unit and never enters PCA. Scores are standardized with their NOC mean and std.

    Args:
        Z: NOC-standardized matrix from standardize_with_noc() (process tags + flare_net).
        reg: Tag registry (unit of each tag).
        unit_folders: Unit names from UNIT_FOLDERS in .env (sets the unit order).
        noc: NOC rows from label_flare_rows().

    Returns:
        A tuple (scores, per_unit, loadings): scores has one column per kept PC named '<unit>_PC<k>' (rows = Z's
        timestamps); per_unit has tags, rank, PCs kept and variance kept per unit; loadings has one row per PC and one
        column per process tag (0 outside the PC's unit).

    Raises:
        SystemExit: If fewer than 2 units have process tags, scores are not finite, or PCs of one unit correlate.
    """
    from sklearn.decomposition import PCA
    unit_of = dict(zip(reg["tag"], reg["unit"]))
    proc = [c for c in Z.columns if c != C.FLARE_NET]
    tag_unit = pd.Series([unit_of[t] for t in proc], index=proc)
    units = [u for u in unit_folders if u in set(tag_unit)] + sorted(set(tag_unit) - set(unit_folders))
    gate(len(units) >= 2, f"{len(units)} units with process tags (a cross-unit graph needs at least 2)")
    A = Z[proc].to_numpy(dtype=np.float64)
    An = A[noc]
    scores, names, pc_unit, load_rows, unit_rows = [], [], [], [], []
    for u in units:
        cols = np.flatnonzero((tag_unit == u).to_numpy())
        pca = PCA(svd_solver="full").fit(An[:, cols])
        k, rank = choose_n_pcs(pca.explained_variance_ratio_, pca.explained_variance_)
        sc = pca.transform(A[:, cols])[:, :k]
        m, s = sc[noc].mean(axis=0), sc[noc].std(axis=0, ddof=1)
        gate(bool(np.isfinite(sc).all()) and bool((s > 0).all()), f"{u}: {k} PC scores finite and not constant")
        sc = (sc - m) / s
        cum = float(pca.explained_variance_ratio_[:k].sum())
        log(f"  {u}: {len(cols)} tags, numerical rank {rank}, {k} PCs kept, {100 * cum:.1f}% of the unit's NOC variance")
        unit_rows.append({"unit": u, "n_tags": len(cols), "rank": rank, "n_pcs": k, "variance_kept": cum})
        for i in range(k):
            names.append(f"{u}_PC{i + 1}")
            pc_unit.append(u)
            scores.append(sc[:, i])
            load_rows.append(pd.Series(pca.components_[i], index=tag_unit.index[cols], name=names[-1]))
    S = pd.DataFrame(np.column_stack(scores), index=Z.index, columns=names)
    R = np.corrcoef(S.to_numpy()[noc], rowvar=False)
    pu = np.array(pc_unit)
    same = (pu[:, None] == pu[None, :]) & ~np.eye(len(pu), dtype=bool)
    gate(not same.any() or float(np.abs(R[same]).max()) < 1e-6, "PCs of the same unit are uncorrelated over NOC")
    rule = (f"N_PCS_PER_UNIT = {C.N_PCS_PER_UNIT}" if C.VAR_EXPLAINED_THRESHOLD is None
            else f"VAR_EXPLAINED_THRESHOLD = {C.VAR_EXPLAINED_THRESHOLD}")
    log(f"unit-wise PCA ({rule}): {len(names)} PCs from {len(units)} units")
    return S, pd.DataFrame(unit_rows), pd.DataFrame(load_rows).fillna(0.0)


# ---- PC ALGORITHM ----
def drop_dependent_nodes(D: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Remove PCs that would make the Fisher-Z test fail: constant, collinear (|r| >= COLLINEAR_R) or near-dependent.

    Args:
        D: PC scores on the PC-algorithm rows (every PC_ROW_STEP-th row), from run_pc_algorithm().
        names: PC names, one per column of D.

    Returns:
        A tuple (keep, D, R, drops): keep marks the PCs kept; D is re-standardized; R is the correlation matrix of D
        (dropped PCs zeroed); drops lists each dropped PC with its reason.
    """
    p = len(names)
    keep = np.ones(p, dtype=bool)
    drops = []
    sd = D.std(axis=0)
    for i in np.flatnonzero(sd == 0):
        keep[i] = False
        drops.append({"node": names[i], "reason": "constant in the PC row subsample", "partner": "", "value": 0.0})
    sd[sd == 0] = 1.0
    D = (D - D.mean(axis=0)) / sd
    R = np.corrcoef(D, rowvar=False)
    R[~keep, :] = 0.0
    R[:, ~keep] = 0.0
    np.fill_diagonal(R, 1.0)
    iu = np.triu_indices(p, 1)
    r = np.abs(R[iu])
    hits = np.flatnonzero(r >= C.COLLINEAR_R)
    for h in hits[np.argsort(-r[hits])]:
        i, j = iu[0][h], iu[1][h]
        if not (keep[i] and keep[j]):
            continue
        keep[j] = False
        drops.append({"node": names[j], "reason": f"collinear (|r| >= {C.COLLINEAR_R})", "partner": names[i],
                      "value": float(r[h])})
    while True:
        k = np.flatnonzero(keep)
        ev, V = np.linalg.eigh(R[np.ix_(k, k)])
        if ev[0] >= C.EIG_MIN:
            break
        v = k[int(np.argmax(np.abs(V[:, 0])))]
        keep[v] = False
        drops.append({"node": names[v], "reason": f"near-linear dependence (eigenvalue {ev[0]:.2e} < {C.EIG_MIN:g})",
                      "partner": "", "value": float(ev[0])})
    for d in drops:
        log(f"  drop {d['node']} before PC: {d['reason']} {d['partner']}")
    log(f"correlation matrix of the {int(keep.sum())} PC nodes: smallest eigenvalue {ev[0]:.3e}")
    return keep, D, R, drops


def choose_pc_depth(Rk: np.ndarray, n: int) -> int:
    """Largest conditioning-set size (depth) <= PC_MAX_K whose estimated number of CI tests fits PC_TEST_BUDGET.

    Args:
        Rk: Correlation matrix of the PCs entering PC, from drop_dependent_nodes().
        n: Number of rows PC uses.

    Returns:
        The depth cap max_k passed to causal-learn (at least 1).
    """
    zc = norm.ppf(1 - C.PC_ALPHA / 2)
    zs = np.arctanh(np.clip(np.abs(Rk), 0, 1 - 1e-12)) * np.sqrt(n - 3)
    A0 = zs > zc
    np.fill_diagonal(A0, False)
    deg = A0.sum(axis=1)
    bound = [int(sum(int(d) * comb(max(int(d) - 1, 0), m) for d in deg)) for m in range(C.PC_MAX_K + 1)]
    cum = np.cumsum(bound)
    fits = [m for m in range(1, C.PC_MAX_K + 1) if cum[m] <= C.PC_TEST_BUDGET]
    max_k = max(fits) if fits else 1
    log(f"marginal skeleton: {int(A0.sum() // 2)} edges, max degree {int(deg.max())}; CI-test bound by depth "
        f"{dict(enumerate(bound))}; depth cap max_k = {max_k}")
    return max_k


def edge_list(G: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Turn causal-learn's graph matrix into an edge table.

    In causal-learn, G[i,j] = -1 and G[j,i] = 1 means i -> j; both -1 means undirected; both 1 means bidirected.

    Args:
        G: Graph matrix cg.G.graph from causal-learn's pc().
        names: Node names in the same order as G's rows.

    Returns:
        Table with columns source, target, type ('->', '--', '<->' or 'other(a,b)').
    """
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


def run_pc_algorithm(scores: pd.DataFrame) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Learn the causal graph among the PCs with causal-learn's PC algorithm (Fisher-Z, stable). flare_net is not used.

    Args:
        scores: NOC-standardized PC scores of all units, from fit_unit_pca().

    Returns:
        A tuple (names, edges, drops): names are the PCs that entered PC; edges is the CPDAG edge table from
        edge_list(); drops lists PCs removed before PC with their reason.

    Raises:
        SystemExit: If the installed causal-learn has no depth cap (max_k).
    """
    from causallearn.search.ConstraintBased.PC import pc
    gate("max_k" in inspect.signature(pc).parameters, "installed causal-learn supports the depth cap max_k")
    names = list(scores.columns)
    D = scores.iloc[::C.PC_ROW_STEP].to_numpy(dtype=np.float64)
    log(f"PC uses every {C.PC_ROW_STEP}th row: {len(D)} rows x {len(names)} PCs (flare_net not included)")
    keep, D, R, drops = drop_dependent_nodes(D, names)
    k = np.flatnonzero(keep)
    names_k = [names[i] for i in k]
    Dk = np.ascontiguousarray(D[:, k])
    max_k = choose_pc_depth(R[np.ix_(k, k)], len(Dk))
    t0 = time.time()
    cg = pc(Dk, alpha=C.PC_ALPHA, indep_test="fisherz", stable=True, uc_rule=0, uc_priority=2, verbose=False,
            show_progress=False, max_k=max_k)
    edges = edge_list(np.array(cg.G.graph, dtype=int), names_k)
    log(f"PC (alpha {C.PC_ALPHA}, Fisher-Z, stable): {len(edges)} edges among {len(names_k)} PCs in "
        f"{time.time() - t0:.0f} s")
    return names_k, edges, pd.DataFrame(drops, columns=["node", "reason", "partner", "value"])


# ---- CPDAG TO DAG ----
def cpdag_to_dag(edges: pd.DataFrame, names: list[str]) -> tuple[dict[str, list[str]], pd.DataFrame, dict[str, int]]:
    """Orient the PC output (CPDAG) into a DAG among the PCs with the Dor-Tarsi procedure.

    Repeatedly removes a node with no outgoing edge whose undirected neighbours are adjacent to all its other
    neighbours, orienting its undirected edges into it. If no such node exists the best sink is forced; if there is
    no sink (directed cycle) the node with the fewest outgoing edges is removed and those edges reversed.

    Args:
        edges: CPDAG edge table from run_pc_algorithm().
        names: PCs that entered PC, from run_pc_algorithm().

    Returns:
        A tuple (parents, dag_edges, stats): parents maps each PC to its parents in the DAG; dag_edges has columns
        source, target; stats has n_edges, n_forced_steps and n_reversed_edges (edges reversed to break loops).

    Raises:
        SystemExit: If the result is not acyclic or does not keep the CPDAG's skeleton.
    """
    import networkx as nx
    out, inn, und = ({v: set() for v in names} for _ in range(3))
    for s, t, ty in edges[["source", "target", "type"]].itertuples(index=False):
        if ty == EDGE_DIRECTED:
            out[s].add(t)
            inn[t].add(s)
        else:
            und[s].add(t)
            und[t].add(s)
    dag, remaining, counts = set(), set(names), {"forced": 0, "reversed": 0}

    def adj(x: str) -> set[str]:
        """All remaining neighbours of x."""
        return out[x] | inn[x] | und[x]

    def remove(x: str) -> None:
        """Orient every remaining edge at x into x, then delete x from the remaining graph."""
        for y in list(und[x]):
            dag.add((y, x))
            und[y].discard(x)
        for y in list(inn[x]):
            dag.add((y, x))
            out[y].discard(x)
        for y in list(out[x]):
            dag.add((y, x))
            inn[y].discard(x)
            counts["reversed"] += 1
        out[x], inn[x], und[x] = set(), set(), set()
        remaining.discard(x)

    def violations(x: str) -> int:
        """Undirected neighbours of x that miss another neighbour of x (0 = Dor-Tarsi condition holds)."""
        ax = adj(x)
        return sum(1 for y in und[x] if not (ax - {y}) <= (adj(y) | {y}))

    order = {v: i for i, v in enumerate(names)}
    while remaining:
        sinks = sorted((v for v in remaining if not out[v]), key=order.get)
        if sinks:
            ok = [v for v in sinks if violations(v) == 0]
            if not ok:
                counts["forced"] += 1
            remove(ok[0] if ok else min(sinks, key=lambda v: (violations(v), order[v])))
        else:
            remove(min(remaining, key=lambda v: (len(out[v]), order[v])))
    g = nx.DiGraph()
    g.add_nodes_from(names)
    g.add_edges_from(dag)
    skel = {frozenset((s, t)) for s, t in edges[["source", "target"]].itertuples(index=False)}
    gate(nx.is_directed_acyclic_graph(g), "DAG is acyclic")
    gate({frozenset(e) for e in dag} == skel, f"DAG keeps exactly the CPDAG skeleton ({len(skel)} edges)")
    if counts["forced"] or counts["reversed"]:
        log(f"DEVIATION: CPDAG has no consistent extension: {counts['forced']} forced Dor-Tarsi steps, "
            f"{counts['reversed']} directed edges reversed to break cycles")
    parents = {v: sorted(g.predecessors(v), key=order.get) for v in names}
    dag_edges = pd.DataFrame(sorted(dag, key=lambda st: (order[st[1]], order[st[0]])), columns=["source", "target"])
    log(f"DAG among PCs: {len(dag)} edges, mean parents {np.mean([len(p) for p in parents.values()]):.2f}")
    return parents, dag_edges, {"n_edges": len(dag), "n_forced_steps": counts["forced"],
                                "n_reversed_edges": counts["reversed"]}


def connect_pcs_to_flare(dag_edges: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Final graph: the DAG among PCs plus one edge from every PC to flare_net (added, not learned).

    Args:
        dag_edges: DAG among PCs from cpdag_to_dag().
        names: PCs in the DAG.

    Returns:
        Edge table (source, target) of the final graph.
    """
    final = pd.concat([dag_edges, pd.DataFrame({"source": names, "target": C.FLARE_NET})], ignore_index=True)
    log(f"final graph: {len(dag_edges)} PC -> PC edges + {len(names)} PC -> {C.FLARE_NET} edges")
    return final


# ---- T2 ----
def noc_correlation(X: np.ndarray, noc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize the PCs with their NOC mean and std and compute their NOC correlation matrix Σ.

    Args:
        X: PC scores (rows = all timestamps, columns = PCs in the DAG), from fit_unit_pca(). No flare data.
        noc: NOC rows from label_flare_rows().

    Returns:
        A tuple (Xs, S): Xs is X standardized with NOC statistics (ddof 1); S is the NOC correlation matrix.

    Raises:
        SystemExit: If a PC is constant over NOC.
    """
    Xn = X[noc]
    m, s = Xn.mean(axis=0), Xn.std(axis=0, ddof=1)
    gate(bool((s > 0).all()), "every T² node varies over NOC")
    Xs = (X - m) / s
    Xn = Xs[noc]
    S = Xn.T @ Xn / (len(Xn) - 1)
    S = (S + S.T) / 2
    np.fill_diagonal(S, 1.0)
    return Xs, S


def t2_threshold(m: int, p: int, k: int, alpha: float) -> float:
    """T² upper limit from report Eq. 13 (corrected): p(m+1)(m-1) / (m(m-k-p)) * F(1-alpha; p, m-k-p).

    Args:
        m: Number of NOC rows used to estimate the mean and covariance.
        p: Number of variables in the statistic (all PCs for the overall T², 1 for one PC's term).
        k: Number of conditioning variables (0 for the overall T², the PC's number of parents for its term).
        alpha: Significance level (ALPHA0 or ALPHA_TERM in config.py).

    Returns:
        The limit; a value above it is out of control.

    Raises:
        SystemExit: If m - k - p <= 0.
    """
    gate(m - k - p > 0, f"T² limit needs m - k - p > 0 (m = {m}, k = {k}, p = {p})")
    return float(p * (m + 1) * (m - 1) / (m * (m - k - p)) * f_dist.ppf(1 - alpha, p, m - k - p))


def compute_t2(X: np.ndarray, Sigma: np.ndarray) -> tuple[np.ndarray, int]:
    """Hotelling T² = Zᵀ Σ⁻¹ Z for every row (paper Eq. 7); pseudo-inverse if Σ is singular.

    Args:
        X: Standardized PCs from noc_correlation() (rows = timestamps).
        Sigma: A covariance / correlation matrix of those PCs (the sample NOC Σ here).

    Returns:
        A tuple (T² per row, number of directions used).
    """
    try:
        L = np.linalg.cholesky(Sigma)
        return (solve_triangular(L, X.T, lower=True) ** 2).sum(axis=0), Sigma.shape[0]
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(Sigma)
        keep = vals > 1e-10 * vals.max()
        log(f"DEVIATION: Σ is singular; T² uses a pseudo-inverse on {int(keep.sum())} of {len(vals)} directions")
        return ((X @ (vecs[:, keep] / np.sqrt(vals[keep]))) ** 2).sum(axis=1), int(keep.sum())


# ---- T2 DECOMPOSITION ----
def fit_causal_regressions(S: np.ndarray, nodes: list[str],
                           parents: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Regress each PC on its parents in the DAG, using the NOC correlation matrix (paper p. 49, Eqs. 5 and 11).

    beta_j = Σ_PP⁻¹ Σ_Pj and sigma²_j = 1 - beta_jᵀ Σ_Pj, where P are PC j's parents.

    Args:
        S: NOC correlation matrix from noc_correlation().
        nodes: PC names in the order of S's rows.
        parents: Parents of each PC in the DAG among PCs, from cpdag_to_dag().

    Returns:
        A tuple (B, sigma2, coef, floored): B[k, j] is the coefficient of parent k for PC j; sigma2[j] is PC j's
        conditional variance; coef lists the coefficients as a table; floored lists PCs whose variance was raised to
        SIGMA2_MIN.
    """
    p = len(nodes)
    pos = {t: i for i, t in enumerate(nodes)}
    B = np.zeros((p, p))
    sigma2 = np.ones(p)
    coef, floored = [], []
    for j, t in enumerate(nodes):
        P = [pos[q] for q in parents[t]]
        if P:
            try:
                beta = np.linalg.solve(S[np.ix_(P, P)], S[P, j])
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(S[np.ix_(P, P)], S[P, j], rcond=None)[0]
                log(f"DEVIATION: parents of {t} are linearly dependent; least-squares path coefficients used")
            B[P, j] = beta
            sigma2[j] = 1.0 - beta @ S[P, j]
            coef += [{"node": t, "parent": nodes[q], "beta": float(b)} for q, b in zip(P, beta)]
        if sigma2[j] < C.SIGMA2_MIN:
            floored.append(t)
            sigma2[j] = C.SIGMA2_MIN
    if floored:
        log(f"DEVIATION: σ²_j·PA(j) below {C.SIGMA2_MIN:g} floored for {floored}")
    return B, sigma2, pd.DataFrame(coef, columns=["node", "parent", "beta"]), floored


def decompose_t2_causal(X: np.ndarray, B: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Per-PC causal terms T_j·PA(j) = (Z_j - sum_k beta_k Z_pa_k) / sigma_j (paper Eqs. 9, 10).

    Args:
        X: Standardized PCs from noc_correlation().
        B: Path coefficients from fit_causal_regressions().
        sigma2: Conditional variances from fit_causal_regressions().

    Returns:
        Array (rows = timestamps, columns = PCs): how far each PC moved beyond what its parents explain, in NOC
        standard deviations.
    """
    return (X - X @ B) / np.sqrt(sigma2)


def bn_implied_correlation(B: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Covariance of the PCs implied by the linear Gaussian network Z = BᵀZ + E (paper Eqs. 1-4).

    Args:
        B: Path coefficients from fit_causal_regressions().
        sigma2: Conditional variances from fit_causal_regressions().

    Returns:
        Σ_G = (I - Bᵀ)⁻¹ D (I - Bᵀ)⁻ᵀ with D = diag(sigma2).
    """
    A = np.linalg.inv(np.eye(len(sigma2)) - B.T)
    Sg = A @ np.diag(sigma2) @ A.T
    return (Sg + Sg.T) / 2


def check_sum_identity(T: np.ndarray, X: np.ndarray, B: np.ndarray, sigma2: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute the overall T² as the sum of the causal terms and check it equals Zᵀ Σ_G⁻¹ Z (paper Eq. 13).

    Args:
        T: Causal terms from decompose_t2_causal().
        X: Standardized PCs from noc_correlation().
        B: Path coefficients from fit_causal_regressions().
        sigma2: Conditional variances from fit_causal_regressions().

    Returns:
        A tuple (T² per row, max relative error of the identity).

    Raises:
        SystemExit: If the identity fails beyond IDENTITY_TOL (or the numerical-inverse tolerance).
    """
    sum_t2 = (T ** 2).sum(axis=1)
    eye = np.eye(len(sigma2))
    Q = (eye - B) @ np.diag(1.0 / sigma2) @ (eye - B.T)
    quad = ((X @ Q) * X).sum(axis=1)
    rel_q = float((np.abs(sum_t2 - quad) / np.maximum(quad, 1.0)).max())
    gate(rel_q < C.IDENTITY_TOL, f"sum identity Σ_j T²_j·PA(j) = Zᵀ (I-B) D⁻¹ (I-Bᵀ) Z = Zᵀ Σ_G⁻¹ Z: max relative "
                                 f"error {rel_q:.2e} < {C.IDENTITY_TOL:g}")
    Sg = bn_implied_correlation(B, sigma2)
    cond = float(np.linalg.cond(Sg))
    try:
        L = np.linalg.cholesky(Sg)
    except np.linalg.LinAlgError:
        log(f"Σ_G is numerically not positive definite (condition number {cond:.1e}); numerical inverse check skipped")
        return sum_t2, rel_q
    direct = (solve_triangular(L, X.T, lower=True) ** 2).sum(axis=0)
    rel_g = float((np.abs(sum_t2 - direct) / np.maximum(direct, 1.0)).max())
    tol = max(C.IDENTITY_TOL, 1e3 * cond * np.finfo(float).eps)
    gate(rel_g < tol, f"same identity with Σ_G = (I-Bᵀ)⁻¹ D (I-Bᵀ)⁻ᵀ inverted numerically (condition number {cond:.1e}): "
                      f"max relative error {rel_g:.2e} < {tol:.1e}")
    return sum_t2, rel_q


def check_noc_terms(T: np.ndarray, noc: np.ndarray, nodes: list[str], floored: list[str]) -> None:
    """Check every causal term has mean 0 and std 1 over NOC, as its N(0,1) reference requires.

    Args:
        T: Causal terms from decompose_t2_causal().
        noc: NOC rows from label_flare_rows().
        nodes: PC names in T's column order.
        floored: PCs with a floored variance (skipped), from fit_causal_regressions().

    Raises:
        SystemExit: If a mean or std is off by more than 1e-3.
    """
    ok = np.array([t not in floored for t in nodes])
    Tn = T[noc][:, ok]
    m, s = float(np.abs(Tn.mean(axis=0)).max()), float(np.abs(Tn.std(axis=0, ddof=1) - 1).max())
    gate(m < 1e-3 and s < 1e-3, f"NOC T_j·PA(j): max |mean| {m:.1e}, max |std - 1| {s:.1e}")


# ---- PRE-FLARE EVALUATION ----
def evaluate_preflare_windows(idx: pd.DatetimeIndex, pre_eid: np.ndarray, events: pd.DataFrame, T: np.ndarray,
                              t2: np.ndarray, t2_lim: float, abs_lim: np.ndarray, nodes: list[str]) -> pd.DataFrame:
    """Decide for each flare event whether it was found: T² above its limit at least once in the 30 min before onset.

    For found events, the first such minute is decomposed (paper Fig. 5): every PC whose |T_j·PA(j)| is above its own
    limit is listed.

    Args:
        idx: Timestamps of the matrix.
        pre_eid: Pre-flare window labels from label_flare_rows().
        events: Flare events from detect_flare_events().
        T: Causal terms from decompose_t2_causal().
        t2: Overall T² from check_sum_identity().
        t2_lim: Overall T² limit from t2_threshold().
        abs_lim: Per-PC limits on |T_j·PA(j)| (square roots of the per-PC t2_threshold() values).
        nodes: PC names in T's column order.

    Returns:
        One row per event: window_rows, rows_t2_above_limit, detected, first_ooc, minutes_before_onset,
        t2_at_first_ooc and significant_pcs (PC name and T value, largest first).
    """
    rows = []
    for ev in events.itertuples(index=False):
        w = np.flatnonzero(pre_eid == ev.event_id)
        ooc = t2[w] > t2_lim
        row = {"event_id": ev.event_id, "onset": ev.start, "window_rows": len(w), "rows_t2_above_limit": int(ooc.sum()),
               "detected": bool(ooc.any()), "first_ooc": pd.NaT, "minutes_before_onset": np.nan,
               "t2_at_first_ooc": np.nan, "significant_pcs": ""}
        if ooc.any():
            i0 = w[int(np.argmax(ooc))]
            sig = np.flatnonzero(np.abs(T[i0]) > abs_lim)
            sig = sig[np.argsort(-np.abs(T[i0, sig]))]
            row.update({"first_ooc": idx[i0], "minutes_before_onset": (ev.start - idx[i0]) / pd.Timedelta(minutes=1),
                        "t2_at_first_ooc": float(t2[i0]),
                        "significant_pcs": "; ".join(f"{nodes[j]} ({T[i0, j]:+.2f})" for j in sig)})
        rows.append(row)
    out = pd.DataFrame(rows, columns=["event_id", "onset", "window_rows", "rows_t2_above_limit", "detected",
                                      "first_ooc", "minutes_before_onset", "t2_at_first_ooc", "significant_pcs"])
    n_found = int(out["detected"].sum())
    log(f"flare events found (T² > {t2_lim:.2f} at least once in the {C.FLARE_PRE_WINDOW_MIN} min before onset): "
        f"{n_found} of {len(out)}" + (f" ({100 * n_found / len(out):.0f}%)" if len(out) else ""))
    return out


# ---- PLOT INPUTS AND PEAK LABELS ----
def load_plot_inputs(out: Path) -> dict[str, Any]:
    """Read the saved pipeline outputs that the figures, the root-cause table and the summary need.

    Args:
        out: Output folder (OUT_DIR from .env) holding a finished main.py run.

    Returns:
        Dict with flare (raw flare_net and its inputs), t2 (all-row T² table), events, terms (all-row causal terms),
        loadings, dag (final graph edges), cpdag (PC output), thr, t2_lim, term_lim (per-PC |T| limit by name),
        limits (limits.csv row) and interval.

    Raises:
        SystemExit: If any of the files is missing.
    """
    files = {"flare": out / "tables" / "flare_net_raw.parquet", "events": out / "tables" / "flare_events.csv",
             "limits": out / "t2" / "limits.csv", "t2": out / "t2" / "t2_all_rows.parquet",
             "terms": out / "t2" / "terms_all_rows.parquet", "loadings": out / "tables" / "pca_loadings.csv",
             "dag": out / "graph" / "dag_edges.csv", "cpdag": out / "graph" / "pc_edges.csv",
             "term_limits": out / "t2" / "term_limits.csv"}
    missing = [p.name for p in files.values() if not p.exists()]
    gate(not missing, f"pipeline outputs exist (run main.py first); missing: {missing or 'none'}")
    limits = pd.read_csv(files["limits"]).iloc[0]
    return {"flare": pd.read_parquet(files["flare"]), "t2": pd.read_parquet(files["t2"]),
            "events": pd.read_csv(files["events"], parse_dates=["start", "end"]),
            "terms": pd.read_parquet(files["terms"]), "loadings": pd.read_csv(files["loadings"], index_col=0),
            "dag": pd.read_csv(files["dag"], dtype=str), "cpdag": pd.read_csv(files["cpdag"], dtype=str),
            "thr": float(limits["flare_threshold"]), "t2_lim": float(limits["t2_limit"]),
            "term_lim": pd.read_csv(files["term_limits"]).set_index("node")["abs_T_limit"], "limits": limits,
            "interval": pd.Timedelta(limits["interval"])}


def time_sections(start: pd.Timestamp, end: pd.Timestamp, hours: float | None = None) -> list[tuple]:
    """Split the time span into consecutive rows of a figure, starting at midnight of the first day.

    Args:
        start: First timestamp to cover.
        end: Last timestamp to cover.
        hours: Length of each section; PLOT_SECTION_HOURS when None.

    Returns:
        List of (section start, section end) pairs.
    """
    step = pd.Timedelta(hours=C.PLOT_SECTION_HOURS if hours is None else hours)
    s, out = start.floor("D"), []
    while s <= end:
        out.append((s, s + step))
        s += step
    return out


def mask_to_spans(index: pd.DatetimeIndex, mask: np.ndarray, interval: pd.Timedelta) -> list[tuple]:
    """Turn a boolean row mask into time spans for shading.

    Args:
        index: Timestamps the mask refers to.
        mask: True for rows to shade (e.g. NOC or pre-flare rows).
        interval: Grid step; each span ends one step after its last row.

    Returns:
        List of (start, end) pairs, one per run of consecutive True rows.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    return [(index[a], index[b - 1] + interval) for a, b in zip(edges[::2], edges[1::2])]


def clip_spans(spans: list[tuple], s: pd.Timestamp, e: pd.Timestamp) -> list[tuple]:
    """Keep the spans that overlap [s, e), cut to that range.

    Args:
        spans: (start, end) pairs from mask_to_spans() or context_spans().
        s: Section start.
        e: Section end.

    Returns:
        The clipped spans.
    """
    return [(max(a, s), min(b, e)) for a, b in spans if b > s and a < e]


def context_spans(D: dict[str, Any]) -> dict[str, list[tuple]]:
    """Time spans of NOC rows, pre-flare windows and flare events, used to shade every time-series figure.

    Args:
        D: Plot inputs from load_plot_inputs().

    Returns:
        Dict with lists of (start, end) pairs under 'noc', 'preflare' and 'event'.
    """
    t2, iv = D["t2"], D["interval"]
    return {"noc": mask_to_spans(t2.index, t2["noc"].to_numpy(), iv),
            "preflare": mask_to_spans(t2.index, (t2["preflare_event_id"] > 0).to_numpy(), iv),
            "event": [(s, e + iv) for s, e in zip(D["events"]["start"], D["events"]["end"])]}


def find_t2_peaks(t2: pd.Series, limit: float, sections: list[tuple]) -> list[pd.Timestamp]:
    """Pick the T² peaks to label: highest values above the limit, PLOT_PEAK_MIN_SEP_MIN apart, a few per section.

    Args:
        t2: Overall T² per row, from t2_all_rows.parquet.
        limit: Overall T² limit.
        sections: Figure sections from time_sections().

    Returns:
        Peak timestamps in time order (at most PLOT_MAX_LABELS_PER_SECTION per section).
    """
    sep = pd.Timedelta(minutes=C.PLOT_PEAK_MIN_SEP_MIN)
    above = t2[t2 > limit]
    picked = []
    for s, e in sections:
        chosen = []
        for t in above[(above.index >= s) & (above.index < e)].sort_values(ascending=False).index:
            if all(abs(t - c) >= sep for c in chosen):
                chosen.append(t)
                if len(chosen) == C.PLOT_MAX_LABELS_PER_SECTION:
                    break
        picked += chosen
    return sorted(picked)


def top_tags_text(loadings: pd.DataFrame, pc: str) -> str:
    """Describe a PC by its PLOT_TOP_TAGS largest-loading plant tags.

    Args:
        loadings: PCA loadings from pca_loadings.csv (rows = PCs, columns = tags).
        pc: PC name, e.g. 'LOUP_PC3'.

    Returns:
        E.g. '99TI1234.PV 34%; 99PI5678.PV 21%' (share = squared loading), or '' for an unknown PC.
    """
    if pc not in loadings.index:
        return ""
    row = loadings.loc[pc].astype(np.float64)
    return "; ".join(f"{t} {100 * row[t] ** 2:.0f}%" for t in row.abs().nlargest(C.PLOT_TOP_TAGS).index)


def node_unit(node: str) -> str:
    """Unit of a graph node: the prefix of '<unit>_PC<k>', or 'flare' for flare_net.

    Args:
        node: Node name from the graph.

    Returns:
        Unit name.
    """
    return "flare" if node == C.FLARE_NET else node.rsplit("_PC", 1)[0]


def label_t2_peaks(D: dict[str, Any], sections: list[tuple]) -> pd.DataFrame:
    """Name each labelled T² peak by the PC with the largest |T_j·PA(j)| there, its unit, top tags and nearest event.

    Args:
        D: Plot inputs from load_plot_inputs().
        sections: Figure sections from time_sections().

    Returns:
        One row per peak: timestamp, T2, top_pc, unit, T_value, share_of_T2, significant (|T| above that PC's limit),
        top_tags, preflare_event_id, in_flare_event, next_event_id, minutes_to_next_onset, peaks_labelled_by_pc.
    """
    t2, terms, events = D["t2"]["T2"], D["terms"], D["events"]
    rows = []
    for t in find_t2_peaks(t2, D["t2_lim"], sections):
        tr = terms.loc[t].astype(np.float64)
        pc = str(tr.abs().idxmax())
        v = float(tr[pc])
        nxt = events[events["start"] >= t]
        rows.append({"timestamp": t, "T2": float(t2[t]), "top_pc": pc, "unit": node_unit(pc), "T_value": v,
                     "share_of_T2": v ** 2 / float((tr ** 2).sum()), "significant": abs(v) > float(D["term_lim"][pc]),
                     "top_tags": top_tags_text(D["loadings"], pc),
                     "preflare_event_id": int(D["t2"].at[t, "preflare_event_id"]),
                     "in_flare_event": bool(D["t2"].at[t, "flare_event"]),
                     "next_event_id": int(nxt["event_id"].iloc[0]) if len(nxt) else -1,
                     "minutes_to_next_onset": (nxt["start"].iloc[0] - t) / pd.Timedelta(minutes=1) if len(nxt) else np.nan})
    cols = ["timestamp", "T2", "top_pc", "unit", "T_value", "share_of_T2", "significant", "top_tags",
            "preflare_event_id", "in_flare_event", "next_event_id", "minutes_to_next_onset"]
    peaks = pd.DataFrame(rows, columns=cols)
    peaks["peaks_labelled_by_pc"] = peaks["top_pc"].map(peaks["top_pc"].value_counts())
    log(f"labelled T² peaks: {len(peaks)}; top PC above its own limit at {int(peaks['significant'].sum())}; "
        f"distinct PCs: {peaks['top_pc'].nunique()}")
    return peaks


# ---- PLOTTING ----
def plot_modules() -> tuple[Any, Any]:
    """Load matplotlib without a display (Agg) and set one style for every figure.

    Returns:
        A tuple (matplotlib.pyplot, matplotlib.dates).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 9})
    return plt, mdates


def format_time_axis(ax: Any, mdates: Any, s: pd.Timestamp, e: pd.Timestamp) -> None:
    """Set an axis to the time range [s, e) with ticks suited to its length (days, 6 h or 30 min).

    Args:
        ax: Matplotlib axis.
        mdates: matplotlib.dates from plot_modules().
        s: Range start.
        e: Range end.
    """
    ax.set_xlim(s, e)
    hours = (e - s) / pd.Timedelta(hours=1)
    if hours > 24 * 7:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2 if hours < 24 * 45 else 7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    elif hours > 6:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def shade(ax: Any, spans: list[tuple], color: str, alpha: float, label: str) -> None:
    """Shade time spans on an axis; only the first span carries the legend label.

    Args:
        ax: Matplotlib axis.
        spans: (start, end) pairs.
        color: Fill colour.
        alpha: Fill opacity.
        label: Legend text.
    """
    for k, (a, b) in enumerate(spans):
        ax.axvspan(a, b, color=color, alpha=alpha, lw=0, label=label if k == 0 else None)


def add_legend(ax: Any) -> None:
    """Show a legend only when the axis has labelled artists.

    Args:
        ax: Matplotlib axis.
    """
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right")


def save_figure(fig: Any, path: Path, plt: Any) -> None:
    """Save a figure at PLOT_DPI (tight bounding box) and close it.

    Args:
        fig: Matplotlib figure.
        path: PNG file to write.
        plt: matplotlib.pyplot from plot_modules().
    """
    fig.savefig(path, dpi=C.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {path.name}")


def peak_label(r: Any) -> str:
    """Two-line label of a T² peak: PC with its T value, then its top tag.

    Args:
        r: One row (namedtuple) of the peak table from label_t2_peaks().

    Returns:
        Label text.
    """
    return f"{r.top_pc} ({r.T_value:+.1f})\n{r.top_tags.split('; ')[0]}"


def plot_flare_overview(D: dict[str, Any], spans: dict[str, list[tuple]], sections: list[tuple], path: Path) -> None:
    """Figure 1: flare_net over the whole window, its three inputs, then flare_net per section with each event numbered.

    Args:
        D: Plot inputs from load_plot_inputs().
        spans: Shading spans from context_spans().
        sections: Figure sections from time_sections().
        path: PNG file to write.
    """
    plt, mdates = plot_modules()
    col, f, events, thr, iv = C.PLOT_COLORS, D["flare"], D["events"], D["thr"], D["interval"]
    net = f[C.FLARE_NET]
    n = len(sections) + 2
    fig, axes = plt.subplots(n, 1, figsize=(C.PLOT_WIDTH_IN, C.PLOT_ROW_HEIGHT_IN * n), squeeze=False,
                             gridspec_kw={"hspace": C.PLOT_ROW_GAP})
    axes = axes[:, 0]
    ax = axes[0]
    ax.plot(net.index, net.to_numpy(), color=col["flare"], lw=0.5)
    ax.axhline(thr, color=col["limit"], ls="--", lw=1, label=f"event threshold {thr:.2f}")
    for t in events["start"]:
        ax.axvline(t, color=col["event"], lw=0.8, alpha=0.7)
    ax.set_title(f"flare_net, whole window: {len(events)} flare events of >= {C.FLARE_MIN_DURATION_MIN} min "
                 "(orange lines = onsets)", loc="left")
    ax.set_ylabel("flare_net (t/h)")
    add_legend(ax)
    format_time_axis(ax, mdates, sections[0][0], sections[-1][1])
    ax = axes[1]
    for c, color in (("flare_1", "#1f77b4"), ("flare_2", "#9467bd"), ("recovery", "#2ca02c")):
        ax.plot(f.index, f[c].to_numpy(), lw=0.5, color=color, label=c)
    shade(ax, mask_to_spans(f.index, f["recovery"].isna().to_numpy(), iv), col["missing"], 0.25, "recovery missing")
    ax.set_title("flare_net inputs: flare_net = flare_1 + flare_2 - recovery", loc="left")
    ax.set_ylabel("t/h")
    add_legend(ax)
    format_time_axis(ax, mdates, sections[0][0], sections[-1][1])
    for k, (s, e) in enumerate(sections):
        ax = axes[k + 2]
        sel = net[(net.index >= s) & (net.index < e)]
        shade(ax, clip_spans(spans["preflare"], s, e), col["preflare"], 0.15, "30-min pre-flare window")
        shade(ax, clip_spans(spans["event"], s, e), col["event"], 0.3, "flare event")
        ax.plot(sel.index, sel.to_numpy(), color=col["flare"], lw=0.7)
        ax.axhline(thr, color=col["limit"], ls="--", lw=1)
        inside = events[(events["start"] >= s) & (events["start"] < e)]
        for ev in inside.itertuples(index=False):
            ax.text(ev.start, 0.97, f"E{ev.event_id}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, fontweight="bold", color=col["event"])
        ax.set_title(f"{s:%d %b %H:%M} to {e:%d %b %H:%M}: {len(inside)} event onsets", loc="left")
        ax.set_ylabel("flare_net (t/h)")
        add_legend(ax)
        format_time_axis(ax, mdates, s, e)
    save_figure(fig, path, plt)


def plot_t2_overview(D: dict[str, Any], spans: dict[str, list[tuple]], sections: list[tuple], peaks: pd.DataFrame,
                     path: Path) -> None:
    """Figure 2: overall T² (log scale) per section with its limit, context shading and each labelled peak named.

    Args:
        D: Plot inputs from load_plot_inputs().
        spans: Shading spans from context_spans().
        sections: Figure sections from time_sections().
        peaks: Peak table from label_t2_peaks().
        path: PNG file to write.
    """
    plt, mdates = plot_modules()
    col, lim = C.PLOT_COLORS, D["t2_lim"]
    t2 = D["t2"]["T2"].clip(lower=1e-3)
    n = len(sections) + 1
    fig, axes = plt.subplots(n, 1, figsize=(C.PLOT_WIDTH_IN, C.PLOT_ROW_HEIGHT_IN * n), squeeze=False,
                             gridspec_kw={"hspace": C.PLOT_ROW_GAP})
    axes = axes[:, 0]
    ax = axes[0]
    ax.plot(t2.index, t2.to_numpy(), color=col["t2"], lw=0.4)
    ax.axhline(lim, color=col["limit"], ls="--", lw=1, label=f"T² limit (Eq. 13) {lim:.1f}")
    for t in D["events"]["start"]:
        ax.axvline(t, color=col["event"], lw=0.8, alpha=0.7)
    ax.set_yscale("log")
    ax.set_title(f"Overall T² = Σ_j T²_j·PA(j) (PCs only), whole window: {100 * (t2 > lim).mean():.1f}% of rows above "
                 "the limit (orange lines = flare onsets)", loc="left")
    ax.set_ylabel("T²")
    add_legend(ax)
    format_time_axis(ax, mdates, sections[0][0], sections[-1][1])
    for k, (s, e) in enumerate(sections):
        ax = axes[k + 1]
        sel = t2[(t2.index >= s) & (t2.index < e)]
        shade(ax, clip_spans(spans["noc"], s, e), col["noc"], 0.08, "NOC (baseline)")
        shade(ax, clip_spans(spans["preflare"], s, e), col["preflare"], 0.2, "30-min pre-flare window")
        shade(ax, clip_spans(spans["event"], s, e), col["event"], 0.3, "flare event")
        ax.plot(sel.index, sel.to_numpy(), color=col["t2"], lw=0.6)
        ax.axhline(lim, color=col["limit"], ls="--", lw=1)
        ax.set_yscale("log")
        pk = peaks[(peaks["timestamp"] >= s) & (peaks["timestamp"] < e)]
        for i, r in enumerate(pk.itertuples(index=False)):
            ax.plot(r.timestamp, r.T2, "v", ms=6, color="black" if r.significant else col["missing"])
            ax.annotate(peak_label(r), xy=(r.timestamp, r.T2), xytext=(0, 8 + 16 * (i % 3)), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5,
                        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8})
        if len(sel):
            ax.set_ylim(top=float(sel.max()) * 30)
        ax.set_title(f"{s:%d %b %H:%M} to {e:%d %b %H:%M}: {len(pk)} labelled peaks "
                     "(black ▼ = that PC is above its own limit, grey ▼ = not)", loc="left")
        ax.set_ylabel("T²")
        add_legend(ax)
        format_time_axis(ax, mdates, s, e)
    save_figure(fig, path, plt)


def plot_event_zooms(D: dict[str, Any], path: Path) -> None:
    """Figure 3: per event, flare_net, T² and each PC's share of T² from 2 h before onset to 30 min after (one PNG).

    Args:
        D: Plot inputs from load_plot_inputs().
        path: PNG file to write.
    """
    plt, mdates = plot_modules()
    col, iv, ev = C.PLOT_COLORS, D["interval"], D["events"]
    if ev.empty:
        log("no flare events: Figure 3 skipped")
        return
    if len(ev) > C.PLOT_MAX_EVENTS:
        log(f"{len(ev)} events: Figure 3 shows the {C.PLOT_MAX_EVENTS} with the largest flare_net peak")
        ev = ev.nlargest(C.PLOT_MAX_EVENTS, "peak_flare_net").sort_values("start")
    ncol = 2 if len(ev) > 1 else 1
    nrow = int(np.ceil(len(ev) / ncol))
    fig = plt.figure(figsize=(C.PLOT_WIDTH_IN, C.PLOT_EVENT_HEIGHT_IN * nrow), constrained_layout=True)
    subs = np.atleast_1d(fig.subfigures(nrow, ncol, hspace=0.04, wspace=0.03)).ravel()
    before, after = pd.Timedelta(minutes=C.PLOT_ZOOM_BEFORE_MIN), pd.Timedelta(minutes=C.PLOT_ZOOM_AFTER_MIN)
    pre = pd.Timedelta(minutes=C.FLARE_PRE_WINDOW_MIN)
    net, t2, terms = D["flare"][C.FLARE_NET], D["t2"]["T2"], D["terms"]
    for sf, e in zip(subs, ev.itertuples(index=False)):
        a, b = e.start - before, e.start + after
        axs = sf.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.3]})
        f = net.loc[a:b]
        axs[0].plot(f.index, f.to_numpy(), color=col["flare"], lw=1)
        axs[0].axhline(D["thr"], color=col["limit"], ls="--", lw=1)
        axs[0].set_ylabel("flare_net (t/h)")
        t = t2.loc[a:b].clip(lower=1e-3)
        axs[1].plot(t.index, t.to_numpy(), color=col["t2"], lw=1)
        axs[1].axhline(D["t2_lim"], color=col["limit"], ls="--", lw=1)
        axs[1].set_yscale("log")
        axs[1].set_ylabel("T²")
        sq = terms.loc[a:b].astype(np.float64) ** 2
        if len(sq):
            share = sq.div(sq.sum(axis=1), axis=0)
            win = share.loc[e.start - pre:e.start - iv]
            top = list((win if len(win) else share).mean().nlargest(C.PLOT_TOP_PCS_ZOOM).index)
            other = (1 - share[top].sum(axis=1)).clip(lower=0)
            axs[2].stackplot(share.index, *[share[c].to_numpy() for c in top], other.to_numpy(),
                             labels=top + ["other PCs"], colors=list(plt.cm.tab10.colors[:len(top)]) + ["#d9d9d9"])
            axs[2].legend(loc="upper left", ncol=3, fontsize=8)
        axs[2].set_ylim(0, 1)
        axs[2].set_ylabel("share of T²")
        for ax in axs:
            ax.axvspan(e.start - pre, e.start, color=col["preflare"], alpha=0.12, lw=0)
            ax.axvspan(e.start, e.end + iv, color=col["event"], alpha=0.25, lw=0)
            ax.axvline(e.start, color=col["event"], lw=1.2)
        format_time_axis(axs[2], mdates, a, b)
        sf.suptitle(f"E{e.event_id}: onset {e.start:%d %b %H:%M}, {e.duration_min:.0f} min, peak flare_net "
                    f"{e.peak_flare_net:.2f} t/h (red band = 30-min pre-flare window)", x=0.01, ha="left", fontsize=12)
    fig.savefig(path, dpi=C.PLOT_DPI)
    plt.close(fig)
    log(f"saved {path.name}")


def plot_pc_terms(D: dict[str, Any], peaks: pd.DataFrame, spans: dict[str, list[tuple]], path: Path) -> None:
    """Figure 4: T_j·PA(j) over the whole window for each PC that labels a T² peak, with its own ± limit.

    Args:
        D: Plot inputs from load_plot_inputs().
        peaks: Peak table from label_t2_peaks().
        spans: Shading spans from context_spans().
        path: PNG file to write.
    """
    plt, mdates = plot_modules()
    if peaks.empty:
        log("no labelled T² peaks: Figure 4 skipped")
        return
    col, terms = C.PLOT_COLORS, D["terms"]
    pcs = list(peaks["top_pc"].value_counts().index[:C.PLOT_MAX_PCS])
    fig, axes = plt.subplots(len(pcs), 1, figsize=(C.PLOT_WIDTH_IN, C.PLOT_ROW_HEIGHT_IN * len(pcs)), squeeze=False,
                             sharex=True, gridspec_kw={"hspace": C.PLOT_ROW_GAP})
    for ax, pc in zip(axes[:, 0], pcs):
        z = float(D["term_lim"][pc])
        s = terms[pc].astype(np.float64)
        own = peaks[peaks["top_pc"] == pc]
        shade(ax, spans["event"], col["event"], 0.3, "flare event")
        ax.plot(s.index, s.to_numpy(), color=col["t2"], lw=0.4)
        ax.axhline(z, color=col["limit"], ls="--", lw=1, label=f"±√(T²_j limit, Eq. 13) = ±{z:.2f}")
        ax.axhline(-z, color=col["limit"], ls="--", lw=1)
        for t in peaks["timestamp"]:
            ax.axvline(t, color=col["missing"], lw=0.4, ls=":")
        ax.plot(own["timestamp"], own["T_value"], "o", ms=5, color=col["preflare"], label="T² peaks labelled with this PC")
        big = max(1.5 * z, float(np.nanpercentile(np.abs(s), 99.9)), 1.1 * float(own["T_value"].abs().max()))
        ax.set_ylim(-big, big)
        ax.set_title(f"{pc} (top tags: {own['top_tags'].iloc[0] or '-'}): labels {len(own)} T² peaks, "
                     f"|T| above its limit at {int(own['significant'].sum())} of them", loc="left")
        ax.set_ylabel("T_j·PA(j)")
        add_legend(ax)
    format_time_axis(axes[-1, 0], mdates, terms.index.min().floor("D"), terms.index.max().ceil("D"))
    save_figure(fig, path, plt)


def plot_individual_pcs(D: dict[str, Any], peaks: pd.DataFrame, spans: dict[str, list[tuple]], sections: list[tuple],
                        folder: Path) -> None:
    """One PNG per PC: T_j·PA(j) over the whole window, then per section, with its own ± limit and labelled peaks.

    Args:
        D: Plot inputs from load_plot_inputs().
        peaks: Peak table from label_t2_peaks().
        spans: Shading spans from context_spans().
        sections: Sections for these figures, from time_sections() with PLOT_PC_SECTION_HOURS.
        folder: Output folder ('individual PCs'); its old PNGs are deleted first.
    """
    plt, mdates = plot_modules()
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    col, terms, dag = C.PLOT_COLORS, D["terms"], D["dag"]
    windows = [(sections[0][0], sections[-1][1])] + list(sections)
    for i, pc in enumerate(terms.columns, start=1):
        z = float(D["term_lim"][pc])
        s = terms[pc].astype(np.float64)
        own = peaks[peaks["top_pc"] == pc]
        par = list(dag.loc[dag["target"] == pc, "source"])
        kids = [k for k in dag.loc[dag["source"] == pc, "target"] if k != C.FLARE_NET]
        big = max(1.5 * z, float(np.nanpercentile(np.abs(s), 99.9)),
                  1.1 * float(own["T_value"].abs().max()) if len(own) else 0.0)
        fig, axes = plt.subplots(len(windows), 1, figsize=(C.PLOT_WIDTH_IN, C.PLOT_ROW_HEIGHT_IN * len(windows)),
                                 squeeze=False, gridspec_kw={"hspace": C.PLOT_ROW_GAP})
        for k, (ax, (a, b)) in enumerate(zip(axes[:, 0], windows)):
            sel = s[(s.index >= a) & (s.index < b)]
            o = own[(own["timestamp"] >= a) & (own["timestamp"] < b)]
            shade(ax, clip_spans(spans["preflare"], a, b), col["preflare"], 0.2, "30-min pre-flare window")
            shade(ax, clip_spans(spans["event"], a, b), col["event"], 0.3, "flare event")
            ax.plot(sel.index, sel.to_numpy(), color=col["t2"], lw=0.4 if k == 0 else 0.7)
            ax.axhline(z, color=col["limit"], ls="--", lw=1, label=f"±√(T²_j limit, Eq. 13) = ±{z:.2f}")
            ax.axhline(-z, color=col["limit"], ls="--", lw=1)
            ax.plot(o["timestamp"], o["T_value"], "o", ms=5, color=col["preflare"], label="T² peaks labelled with this PC")
            ax.set_ylim(-big, big)
            ax.set_ylabel("T_j·PA(j)")
            if k == 0:
                ax.set_title(f"{pc}: whole window; |T| > {z:.2f} in {100 * (s.abs() > z).mean():.2f}% of rows; labels "
                             f"{len(own)} T² peaks\nparents ({len(par)}): {', '.join(par[:10]) or 'none (root node)'}"
                             f"{' ...' if len(par) > 10 else ''}; PC children: {len(kids)}; top tags: "
                             f"{top_tags_text(D['loadings'], pc) or '-'}", loc="left")
            else:
                ax.set_title(f"{a:%d %b %H:%M} to {b:%d %b %H:%M}: {len(o)} labelled peaks", loc="left")
            add_legend(ax)
            format_time_axis(ax, mdates, a, b)
        save_figure(fig, folder / f"{i:03d}_{pc}.png", plt)
    log(f"individual PC figures: {terms.shape[1]} PNGs in '{folder.name}'")


def plot_causal_graph(D: dict[str, Any], path: Path) -> None:
    """Figure 5: the final graph (PC algorithm edges among PCs + every PC -> flare_net) and a unit-level summary.

    Args:
        D: Plot inputs from load_plot_inputs().
        path: PNG file to write.
    """
    import networkx as nx
    from matplotlib.patches import Patch
    plt, _ = plot_modules()
    dag, cp, col = D["dag"], D["cpdag"], C.PLOT_COLORS
    pcs = list(D["terms"].columns)
    units = list(dict.fromkeys(node_unit(v) for v in pcs))
    ucol = {u: plt.cm.tab20(i % 20) for i, u in enumerate(units)}
    ucol["flare"] = col["flare"]
    G = nx.DiGraph()
    G.add_nodes_from(pcs + [C.FLARE_NET])
    G.add_edges_from(zip(dag["source"], dag["target"]))
    to_flare = [(s, t) for s, t in G.edges if t == C.FLARE_NET]
    same = [(s, t) for s, t in G.edges if t != C.FLARE_NET and node_unit(s) == node_unit(t)]
    cross = [(s, t) for s, t in G.edges if t != C.FLARE_NET and node_unit(s) != node_unit(t)]
    n_dir = int((cp["type"] == EDGE_DIRECTED).sum())
    fig, axes = plt.subplots(2, 1, figsize=(C.PLOT_WIDTH_IN, 40), gridspec_kw={"height_ratios": [24, 15]})

    ax = axes[0]
    ang = {v: 2 * np.pi * i / len(pcs) for i, v in enumerate(pcs)}
    pos = {v: (np.cos(a), np.sin(a)) for v, a in ang.items()}
    pos[C.FLARE_NET] = (0.0, 0.0)
    nx.draw_networkx_edges(G, pos, edgelist=to_flare, ax=ax, edge_color=col["flare"], width=0.5, alpha=0.25,
                           arrowsize=6, node_size=600)
    nx.draw_networkx_edges(G, pos, edgelist=cross, ax=ax, edge_color="#7f7f7f", width=0.4, alpha=0.3, arrowsize=6,
                           node_size=40)
    nx.draw_networkx_edges(G, pos, edgelist=same, ax=ax, edge_color="#1f5fa8", width=0.6, alpha=0.5, arrowsize=6,
                           node_size=40, connectionstyle="arc3,rad=0.3")
    nx.draw_networkx_nodes(G, pos, nodelist=pcs, ax=ax, node_size=40, node_color=[ucol[node_unit(v)] for v in pcs])
    nx.draw_networkx_nodes(G, pos, nodelist=[C.FLARE_NET], ax=ax, node_size=600, node_shape="s", node_color=col["flare"])
    ax.text(0, -0.07, C.FLARE_NET, ha="center", va="top", fontsize=12, fontweight="bold")
    for u in units:
        a = np.mean([ang[v] for v in pcs if node_unit(v) == u])
        ax.text(1.13 * np.cos(a), 1.13 * np.sin(a), u, ha="center", va="center", fontsize=12, color=ucol[u],
                fontweight="bold")
    ax.set_title(f"Final graph: {len(pcs)} PCs + flare_net. PC algorithm on PCs only: {len(same) + len(cross)} edges "
                 f"(cross-unit {len(cross)} grey, same-unit {len(same)} blue; PC output had {n_dir} directed and "
                 f"{len(cp) - n_dir} undirected). Every PC -> flare_net added afterwards ({len(to_flare)} red).",
                 loc="left")
    ax.legend(handles=[Patch(color=ucol[u], label=u) for u in units], loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    U = nx.DiGraph()
    U.add_nodes_from(units)
    counts = pd.Series([(node_unit(s), node_unit(t)) for s, t in G.edges if t != C.FLARE_NET]).value_counts()
    within = {u: int(counts.get((u, u), 0)) for u in units}
    for (a, b), c in counts.items():
        if a != b:
            U.add_edge(a, b, n=int(c))
    upos = {u: (np.cos(2 * np.pi * i / len(units)), np.sin(2 * np.pi * i / len(units))) for i, u in enumerate(units)}
    cmax = max([d["n"] for _, _, d in U.edges(data=True)] + [1])
    for a, b, d in U.edges(data=True):
        nx.draw_networkx_edges(U, upos, edgelist=[(a, b)], ax=ax, width=0.5 + 6 * d["n"] / cmax, alpha=0.6,
                               edge_color="#555555", arrowsize=14, node_size=2600, connectionstyle="arc3,rad=0.12")
    nx.draw_networkx_nodes(U, upos, ax=ax, node_size=2600, node_color=[ucol[u] for u in U.nodes])
    n_pc = pd.Series([node_unit(v) for v in pcs]).value_counts()
    nx.draw_networkx_labels(U, upos, ax=ax, font_size=9,
                            labels={u: f"{u}\n{n_pc[u]} PCs\n{within[u]} within" for u in U.nodes})
    top = [(a, b, d["n"]) for a, b, d in sorted(U.edges(data=True), key=lambda e: -e[2]["n"])][:12]
    ax.text(1.02, 0.98, "Most edges between units:\n" + "\n".join(f"{a} → {b}: {n}" for a, b, n in top),
            transform=ax.transAxes, va="top", fontsize=10, family="monospace")
    ax.set_title("Unit-level summary of the PC edges: arrow width = number of PC edges from unit to unit", loc="left")
    ax.set_aspect("equal")
    ax.axis("off")
    save_figure(fig, path, plt)


# ---- ROOT-CAUSE TABLE ----
def root_cause_table(D: dict[str, Any]) -> pd.DataFrame:
    """Per flare event: PCs above their own limit in the 30-min pre-flare window, ranked by minutes above, then earliest.

    Args:
        D: Plot inputs from load_plot_inputs().

    Returns:
        One row per (event, PC): rank, pc, unit, minutes_significant (minutes above its limit), window_rows,
        first_significant, minutes_before_onset, T_at_first_significant, max_abs_T, significant_at_first_ooc
        (above its limit at the first T² alarm), top_tags.
    """
    t2, terms, lim = D["t2"], D["terms"], D["t2_lim"]
    rows = []
    for ev in D["events"].itertuples(index=False):
        w = t2.index[t2["preflare_event_id"].to_numpy() == ev.event_id]
        if not len(w):
            continue
        T = terms.loc[w].astype(np.float64)
        sig = T.abs() > D["term_lim"].reindex(T.columns).to_numpy()
        above = t2.loc[w, "T2"].to_numpy() > lim
        first_ooc = w[int(np.argmax(above))] if above.any() else None
        for pc in sig.columns[sig.any().to_numpy()]:
            s = sig[pc].to_numpy()
            i0 = int(np.argmax(s))
            rows.append({"event_id": ev.event_id, "onset": ev.start, "pc": pc, "unit": node_unit(pc),
                         "minutes_significant": int(s.sum()), "window_rows": len(w), "first_significant": w[i0],
                         "minutes_before_onset": (ev.start - w[i0]) / pd.Timedelta(minutes=1),
                         "T_at_first_significant": float(T[pc].iloc[i0]), "max_abs_T": float(T[pc].abs().max()),
                         "significant_at_first_ooc": bool(first_ooc is not None and sig.at[first_ooc, pc]),
                         "top_tags": top_tags_text(D["loadings"], pc)})
    cols = ["event_id", "onset", "pc", "unit", "minutes_significant", "window_rows", "first_significant",
            "minutes_before_onset", "T_at_first_significant", "max_abs_T", "significant_at_first_ooc", "top_tags"]
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        out = out.sort_values(["event_id", "minutes_significant", "minutes_before_onset"],
                              ascending=[True, False, False])
        out.insert(2, "rank", out.groupby("event_id").cumcount() + 1)
        for r in out[out["rank"] == 1].itertuples(index=False):
            log(f"  E{r.event_id}: top candidate {r.pc}, above its limit {r.minutes_significant} of {r.window_rows} min, "
                f"first {r.minutes_before_onset:.0f} min before onset")
    log(f"root-cause table: {len(out)} rows for {out['event_id'].nunique()} events")
    return out


# ---- SUMMARY ----
def drop_reason(status: str) -> str:
    """Short drop reason from a tag-registry status, for counting drops by reason.

    Args:
        status: Status string from tag_registry.csv, e.g. 'dropped: 64.5% missing after ffill (> 30%)'.

    Returns:
        One of 'constant in NOC', 'too much missing', 'near-constant', 'duplicate column', 'constant', or the status.
    """
    for key, name in (("within NOC", "constant in NOC"), ("missing", "too much missing"),
                      ("near-constant", "near-constant"), ("duplicate", "duplicate column"), ("constant", "constant")):
        if key in str(status):
            return name
    return str(status)


def build_summary(out: Path) -> None:
    """Write outputs/SUMMARY.md: data and cleaning, PCA variance, graph and loops, T² limits, detection and root causes.

    Args:
        out: Output folder (OUT_DIR from .env) holding a finished main.py run.
    """
    D = load_plot_inputs(out)
    lim, t2, ev, dag, cp = D["limits"], D["t2"], D["events"], D["dag"], D["cpdag"]
    reg = pd.read_csv(out / "tag_registry.csv", dtype=str, keep_default_na=False)
    units = pd.read_csv(out / "tables" / "pcs_per_unit.csv")
    dropped_pc = pd.read_csv(out / "graph" / "pc_dropped_nodes.csv", dtype=str)
    stats = pd.read_csv(out / "graph" / "dag_stats.csv").iloc[0]
    terml = pd.read_csv(out / "t2" / "term_limits.csv")
    first = pd.read_csv(out / "t2" / "preflare_first_ooc.csv").set_index("event_id")
    short = pd.read_csv(out / "tables" / "short_flares.csv")
    rc = root_cause_table(D)
    pre = t2["preflare_event_id"] > 0

    drops = reg.loc[reg["status"].str.startswith("dropped"), "status"].map(drop_reason).value_counts()
    L = ["# SUMMARY: unit-wise PCA → PC algorithm → causation-based T²", "",
         f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} from `outputs/` by `summary.py`.", "",
         "## Data and cleaning", "| item | value |", "|---|---|",
         f"| time window | {window_text()} |", f"| sampling interval | {lim['interval']} |",
         f"| tags loaded | {int((reg['tag'] != C.FLARE_NET).sum())} ({int((reg['is_flare'] == 'True').sum()) - 1} flare tags) |",
         f"| process tags used in PCA | {int((reg['status'] == 'used_in_pca').sum())} |",
         f"| tags dropped | {int(drops.sum())}: " + (", ".join(f"{k} {v}" for k, v in drops.items()) or "none") + " |",
         f"| rows in matrix | {len(t2)} |", f"| NOC rows (baseline) | {int(t2['noc'].sum())} |",
         f"| flare-event rows / pre-flare rows | {int(t2['flare_event'].sum())} / {int(pre.sum())} |",
         f"| flare_net threshold | {D['thr']:.3f} t/h ({lim['flare_threshold_rule']}) |",
         f"| flare events (>= {C.FLARE_MIN_DURATION_MIN} min) | {len(ev)}, {ev['duration_min'].sum():.0f} event minutes |",
         f"| shorter runs above the threshold (not events, kept out of NOC) | {len(short)} |", ""]

    n_all = float(units["n_tags"].sum())
    overall = float((units["variance_kept"] * units["n_tags"]).sum()) / n_all if n_all else float("nan")
    L += ["## PCA variance", "| unit | tags | rank | PCs kept | variance captured |", "|---|---|---|---|---|"]
    L += [f"| {r.unit} | {r.n_tags} | {r.rank} | {r.n_pcs} | {100 * r.variance_kept:.1f}% |"
          for r in units.itertuples(index=False)]
    L += [f"| **all units** | {int(n_all)} | | {int(units['n_pcs'].sum())} | {100 * overall:.1f}% |", ""]

    to_flare = dag[dag["target"] == C.FLARE_NET]
    pc_edges = dag[dag["target"] != C.FLARE_NET]
    same = int((pc_edges["source"].map(node_unit) == pc_edges["target"].map(node_unit)).sum())
    types = cp["type"].value_counts()
    L += ["## Graph and loops",
          f"- PC algorithm input: {terml.shape[0] + len(dropped_pc)} PCs (flare_net not included); dropped before PC "
          f"(collinear / dependent): {len(dropped_pc)}"
          + (f" ({', '.join(dropped_pc['node'].head(10))})" if len(dropped_pc) else "") + ".",
          f"- PC output: {len(cp)} edges = {int(types.get(EDGE_DIRECTED, 0))} directed, "
          f"{int(types.get(EDGE_UNDIRECTED, 0))} undirected, {int(types.get(EDGE_BIDIRECTED, 0))} bidirected.",
          f"- DAG among PCs: {len(pc_edges)} edges = {same} same-unit, {len(pc_edges) - same} cross-unit. Then every PC "
          f"-> {C.FLARE_NET} added: {len(to_flare)} edges.",
          f"- Loops: {int(stats['n_reversed_edges'])} directed edges reversed to break cycles; "
          f"{int(stats['n_forced_steps'])} forced Dor-Tarsi steps (no consistent extension); final graph is acyclic "
          "(checked in the run).",
          f"- Parents per PC (among PCs): mean {terml['n_parents'].mean():.2f}, max {int(terml['n_parents'].max())}; "
          f"root PCs (no parents): {int((terml['n_parents'] == 0).sum())}.", ""]

    n_found = int(first["detected"].astype(str).eq("True").sum()) if "detected" in first else 0
    L += ["## T² detection and root causes",
          f"- Overall T² (PCs only, no flare data) limit {D['t2_lim']:.2f}: report Eq. 13 (corrected), m = "
          f"{int(lim['n_noc_rows'])} NOC rows, p = {int(lim['p'])} PCs, k = 0, α = {lim['alpha0']:g}. Above it: "
          f"{lim['pct_noc_above_t2_limit']:.2f}% of NOC rows (expected about {100 * lim['alpha0']:g}%), "
          f"{lim['pct_preflare_above_t2_limit']:.2f}% of pre-flare rows.",
          f"- Flare events found (T² above its limit at least once in the {C.FLARE_PRE_WINDOW_MIN} min before onset): "
          f"{n_found} of {len(ev)}.",
          f"- Per-PC limits on T²_j: Eq. 13 with p = 1, k = number of parents, α = {lim['alpha_term']:g}: "
          f"{terml['t2_limit'].min():.3f} to {terml['t2_limit'].max():.3f}. NOC rows above: mean "
          f"{lim['pct_noc_above_term_limit_mean']:.2f}% per PC (expected about {100 * lim['alpha_term']:g}%).",
          f"- Sum identity Σ_j T²_j·PA(j) = T²: max relative error {lim['sum_identity_max_rel_error']:.1e}. Median gap to "
          f"the sample-Σ T² on NOC: {lim['gap_sample_vs_bn_median_noc']:.3f}.", "",
          "| event | onset | min | peak t/h | found | first T² alarm (min before onset) | top root-cause PC | "
          "minutes above its limit |", "|---|---|---|---|---|---|---|---|"]
    top = rc[rc["rank"] == 1].set_index("event_id") if len(rc) else pd.DataFrame()
    for e in ev.head(40).itertuples(index=False):
        alarm = first["minutes_before_onset"].get(e.event_id, np.nan)
        found = "yes" if str(first["detected"].get(e.event_id, False)) == "True" else "no"
        if e.event_id in top.index:
            r = top.loc[e.event_id]
            cause = f"{r['pc']} | {int(r['minutes_significant'])} of {int(r['window_rows'])}"
        else:
            cause = "none | -"
        L.append(f"| E{e.event_id} | {e.start:%d %b %H:%M} | {e.duration_min:.0f} | {e.peak_flare_net:.2f} | {found} | "
                 f"{'-' if pd.isna(alarm) else f'{alarm:.0f}'} | {cause} |")
    if len(ev) > 40:
        L.append(f"\n{len(ev) - 40} more events in `tables/flare_events.csv` and `tables/root_cause_by_event.csv`.")
    start, end = pd.Timestamp(C.TA_CHECK_START), pd.Timestamp(C.TA_CHECK_END)
    ws, we = C.window_bounds()
    if (ws is not None and end <= ws) or (we is not None and start >= we):
        L += ["", f"- TA cross-check: not applicable, the TA's period [{C.TA_CHECK_START}, {C.TA_CHECK_END}) is outside "
              "the time window."]
    else:
        ta = ev[(ev["start"] >= start) & (ev["start"] < end)]
        L += ["", f"- TA cross-check: {len(ta)} events with onset in [{C.TA_CHECK_START}, {C.TA_CHECK_END}); TA said "
              f"{C.TA_FLARE_COUNT} flares of about {C.TA_FLARE_MINUTES} min."]
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"SUMMARY.md written ({len(L)} lines)")
