"""All logic of the pipeline: loading, cleaning, flare tags and events, unit-wise PCA, PC algorithm, DAG, T² and its decomposition."""
import importlib.util
import inspect
import re
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import chi2, norm

import config as C

EDGE_DIRECTED, EDGE_UNDIRECTED, EDGE_BIDIRECTED = "->", "--", "<->"
_T0 = time.time()
_LOG = {"file": None}


# ---- LOGGING AND CHECKS ----
def check_packages():
    """Stop with a pip command if a required package is missing."""
    need = {"pandas": "pandas", "pyarrow": "pyarrow", "numpy": "numpy", "scipy": "scipy", "networkx": "networkx",
            "causallearn": "causal-learn", "dotenv": "python-dotenv", "sklearn": "scikit-learn"}
    missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        sys.exit("Missing packages. Install with:  pip install " + " ".join(missing))


def make_output_dirs(out):
    """Create the output sub-folders and return them by name."""
    dirs = {name: out / name for name in ("tables", "graph", "t2", "logs")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def start_log(path):
    """Copy every log line to this file as well as the screen."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    _LOG["file"] = open(path, "w", encoding="utf-8")


def log(msg=""):
    """Print a time-stamped message and append it to the run log."""
    line = f"[{time.time() - _T0:8.1f}s] {msg}"
    print(line, flush=True)
    if _LOG["file"] is not None:
        _LOG["file"].write(line + "\n")
        _LOG["file"].flush()


def gate(ok, msg):
    """Log a check and stop the run with exit code 1 if it fails."""
    log(f"CHECK {'PASS' if ok else 'FAIL'}: {msg}")
    if not ok:
        log("CHECK FAILED - run stopped.")
        sys.exit(1)


# ---- LOADING ----
def norm_unit(name):
    """Unit-name key that ignores case and punctuation (SEU-1 == SEU1)."""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def unit_from_text(text, unit_folders):
    """Longest unit name contained in a path (ISOM_3 wins over ISOM), else None."""
    key = norm_unit(text)
    hits = [u for u in unit_folders if norm_unit(u) in key]
    return max(hits, key=len) if hits else None


def tag_key(name):
    """Deduplication key: tag name without spaces, case ignored."""
    return re.sub(r"\s+", "", str(name)).upper()


def in_window(times):
    """Boolean mask of timestamps inside [WINDOW_START, WINDOW_END)."""
    start, end = C.window_bounds()
    t = pd.DatetimeIndex(times)
    mask = np.ones(len(t), dtype=bool)
    if start is not None:
        mask &= t >= start
    if end is not None:
        mask &= t < end
    return mask


def window_text():
    """Readable description of the time window."""
    start, end = C.window_bounds()
    if start is None and end is None:
        return "all data (no window)"
    return f"{start or 'start of data'} to {end or 'end of data'} (end excluded)"


def parquet_files(folder):
    """All parquet files below a folder, sorted."""
    return sorted(p for p in Path(folder).rglob("*.parquet") if p.is_file())


def read_table(path):
    """Read one parquet file; a non-default index (e.g. a DatetimeIndex) becomes a column."""
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    df.columns = [str(c) for c in df.columns]
    return df


def to_datetime_or_none(s):
    """Parse a column as naive timestamps (tz-aware -> UTC) and its original time zone; None if < 95 % parse."""
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


def find_time_column(df):
    """Name, parsed timestamps and original time zone of the timestamp column, or (None, None, '')."""
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    hinted = [c for c in df.columns if c not in dt_cols and ("time" in c.lower() or "date" in c.lower())]
    for c in dt_cols + hinted:
        parsed, tz = to_datetime_or_none(df[c])
        if parsed is not None:
            return c, parsed, tz
    return None, None, ""


def numeric_part(df):
    """Numeric tag columns of a raw table; mostly-numeric text columns are coerced, other columns dropped."""
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
    """Median spacing of the unique timestamps."""
    ts = pd.DatetimeIndex(index).unique().sort_values()
    return ts.to_series().diff().median() if len(ts) > 1 else pd.NaT


def to_ns_index(values):
    """DatetimeIndex in nanoseconds, so every time difference in the pipeline has one unit."""
    idx = pd.DatetimeIndex(values, name="timestamp")
    return idx.as_unit("ns") if hasattr(idx, "as_unit") else idx


def load_file(path, data_root):
    """One parquet file -> numeric frame indexed by timestamp inside the window (None without a timestamp) and info."""
    df = read_table(path)
    tcol, t, tz = find_time_column(df)
    info = {"file": Path(path).relative_to(data_root).as_posix(), "rows": len(df), "time_column": tcol, "tz": tz}
    if tcol is None:
        return None, info
    num, dropped, coerced = numeric_part(df.drop(columns=[tcol]))
    num.index = to_ns_index(t.to_numpy())
    num = num[~num.index.isna() & in_window(num.index)]
    info.update({"rows_in_window": len(num), "median_interval": median_interval(num.index)})
    return num, info


def load_all_tags(P):
    """Load every tag of the flare, unit and optional folders and build the tag registry (first copy of a duplicate kept)."""
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


def join_and_resample(folder_frames, intervals, reg):
    """Set sentinels to NaN, resample every folder to the most common sampling interval and join on time."""
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
def build_flare_net(X, reg, P):
    """Replace all flare tags by one node flare_net = flare 1 + flare 2 - recovery and return its raw series."""
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
def detect_flare_events(net, interval):
    """Flag flare_net > median + K·MAD and merge flags at most EVENT_GAP samples apart into events (onset = first flag)."""
    med = float(net.median())
    mad = 1.4826 * float((net - med).abs().median())
    thr = med + C.FLARE_MAD_K * mad
    flagged = (net > thr).to_numpy()
    t = pd.DatetimeIndex(net.index)[flagged]
    v = net.to_numpy()[flagged]
    max_gap = (C.EVENT_GAP * interval).to_timedelta64()
    new = np.r_[True, np.diff(t.values) > max_gap] if len(t) else np.zeros(0, dtype=bool)
    g = pd.DataFrame({"t": t, "v": v, "eid": np.cumsum(new)}).groupby("eid")
    events = pd.DataFrame({"start": g["t"].min(), "end": g["t"].max(), "n_flagged": g.size(),
                           "peak_flare_net": g["v"].max()}).reset_index(drop=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    events["duration_min"] = (events["end"] - events["start"] + interval) / pd.Timedelta(minutes=1)
    log(f"flare threshold = median {med:.3f} + {C.FLARE_MAD_K:g} x MAD {mad:.3f} = {thr:.3f}; flagged rows "
        f"{100 * flagged.mean():.2f}%; flare events: {len(events)} (flags <= {C.EVENT_GAP} samples apart merged)")
    return events, thr


def crosscheck_flare_events_ta(events):
    """Print the events in the TA's month and compare them with the TA's account; never stops the run."""
    start, end = pd.Timestamp(C.TA_CHECK_START), pd.Timestamp(C.TA_CHECK_END)
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


def label_flare_rows(idx, events, interval, n_tags):
    """Flare-event rows, pre-flare window labels and the NOC rows (first TRAIN_FRAC of time minus flare windows ± buffer)."""
    flare = np.zeros(len(idx), dtype=bool)
    near_flare = np.zeros(len(idx), dtype=bool)
    buffer = pd.Timedelta(minutes=C.FLARE_BUFFER_MIN)
    for s, e in zip(events["start"], events["end"]):
        flare |= (idx > s - interval) & (idx <= e)
        near_flare |= (idx > s - interval - buffer) & (idx <= e + buffer)
    pre_eid = extract_preflare_window(idx, events, interval, flare)
    cut = idx.min() + C.TRAIN_FRAC * (idx.max() - idx.min())
    noc = (idx < cut) & ~near_flare
    rule = f"NOC = rows before {cut} (first {C.TRAIN_FRAC:.0%} of time) minus flare events ± {buffer}"
    min_noc = 5 * n_tags
    if noc.sum() < min_noc:
        noc = (idx < cut) & ~flare & (pre_eid == 0)
        rule = f"NOC = rows before {cut} minus flare-event rows and pre-flare windows (no buffer)"
        log(f"DEVIATION: NOC with the ± {buffer} buffer had < {min_noc} rows (5 x tags); fell back to: {rule}")
    log(f"{rule}; flare-event rows {100 * flare.mean():.2f}%, pre-flare rows {int((pre_eid > 0).sum())}, "
        f"NOC rows {int(noc.sum())}")
    gate(noc.sum() >= min_noc, f"NOC rows {int(noc.sum())} >= {min_noc} (5 x tags)")
    gate(not (noc & (flare | (pre_eid > 0))).any(), "no NOC row lies inside a flare event or a pre-flare window")
    return flare, pre_eid, noc


def preflare_window_length(interval):
    """Number of samples in FLARE_PRE_WINDOW_MIN minutes at the measured sampling interval."""
    n = int(round(pd.Timedelta(minutes=C.FLARE_PRE_WINDOW_MIN) / interval))
    gate(n >= 1, f"pre-flare window {C.FLARE_PRE_WINDOW_MIN} min = {n} samples of {interval}")
    return n


def extract_preflare_window(idx, events, interval, flare):
    """Label every row in [onset − 30 min, onset) with its event id (0 = none); rows inside any flare event are left out."""
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


def check_flare_source_agreement(flare_z, mu, sd, net, thr, flare, out_csv):
    """Assert that the flare_net column given to PC and the flare_net that defined the events agree on the same timestamps."""
    gate(flare_z.index.isin(net.index).all(), "every matrix timestamp exists on the flare_net grid used for the events")
    raw = net.reindex(flare_z.index).to_numpy(dtype=np.float64)
    back = flare_z.to_numpy(dtype=np.float64) * sd + mu
    tol = 1e-4 * (1.0 + np.abs(np.nan_to_num(raw)))
    filled = np.isnan(raw)
    value_bad = ~filled & (np.abs(back - raw) > tol)
    flag_bad = ~filled & (np.abs(raw - thr) > tol) & ((back > thr) != (raw > thr))
    label_bad = ~filled & (raw > thr) & ~flare
    filled_flagged = filled & (back > thr)
    bad = value_bad | flag_bad | label_bad | filled_flagged
    rows = pd.DataFrame({"timestamp": flare_z.index[bad], "flare_net_events": raw[bad], "flare_net_pc": back[bad],
                         "value_mismatch": value_bad[bad], "flag_mismatch": flag_bad[bad],
                         "flagged_outside_event": label_bad[bad], "filled_value_above_threshold": filled_flagged[bad]})
    rows.to_csv(out_csv, index=False)
    for r in rows.head(20).itertuples(index=False):
        log(f"  MISMATCH {r.timestamp}: events {r.flare_net_events}, PC input {r.flare_net_pc}")
    log(f"flare_net agreement: {int(filled.sum())} matrix rows filled by the short-gap forward fill (no raw value)")
    gate(not bad.any(), f"flare_net used by PC equals the flare_net used for the events ({int(bad.sum())} mismatching rows)")


# ---- CLEANING ----
def ffill_short_gaps(df, max_gap):
    """Forward-fill only NaN runs of length <= max_gap that follow a valid value (longer gaps stay NaN)."""
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


def clean_matrix(X, reg, status):
    """Short-gap fill, drop sparse / constant / near-constant / duplicate tags and incomplete rows; flare_net is never dropped."""
    is_flare = dict(zip(reg["tag"], reg["is_flare"]))

    def drop(frame, cols, reason):
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


def standardize_with_noc(X, reg, noc, status):
    """Drop tags constant in NOC, z-score every tag with NOC mean / std and set leftover NaN to 0 (= NOC mean)."""
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
        log(f"  {c} is constant within NOC: standardized with all rows (used for the PC graph only)")
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
def choose_n_pcs(evr, ev):
    """PCs kept for one unit: N_PCS_PER_UNIT or the VAR_EXPLAINED_THRESHOLD rule, capped by the numerical rank."""
    if C.VAR_EXPLAINED_THRESHOLD is None:
        k = C.N_PCS_PER_UNIT
    else:
        k = int(np.searchsorted(np.cumsum(evr), C.VAR_EXPLAINED_THRESHOLD) + 1)
    rank = int((ev > C.EIG_MIN * ev[0]).sum())
    return min(k, len(evr), rank), rank


def fit_unit_pca(Z, reg, unit_folders, noc):
    """Fit one PCA per unit on its NOC rows, keep x PCs per unit and return NOC-standardized scores for all rows."""
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


# ---- APPEND FLARE TAGS ----
def append_flare_tags(scores, flare_z, noc, pc_only):
    """Append the NOC-standardized flare_net column to the stacked PC scores to form the PC-algorithm input."""
    gate(scores.index.equals(flare_z.index), "PC scores and flare_net share the same timestamps")
    F = flare_z.astype(np.float64)
    for c in F.columns:
        if c in pc_only:
            continue
        m, s = float(F.loc[noc, c].mean()), float(F.loc[noc, c].std())
        gate(abs(m) < 1e-3 and abs(s - 1) < 1e-3, f"{c} is standardized like the PCs (NOC mean {m:.1e}, std {s:.4f})")
    N = pd.concat([scores, F], axis=1)
    log(f"node matrix for PC: {scores.shape[1]} PCs + {F.shape[1]} flare tag(s) = {N.shape[1]} nodes")
    return N


# ---- PC ALGORITHM ----
def drop_dependent_nodes(D, names, fl):
    """Drop constant, collinear (|r| >= COLLINEAR_R) and near-linearly-dependent nodes before Fisher-Z; flare tags kept."""
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
        victim, partner = (i, j) if (fl[j] and not fl[i]) else (j, i)
        keep[victim] = False
        drops.append({"node": names[victim], "reason": f"collinear (|r| >= {C.COLLINEAR_R})", "partner": names[partner],
                      "value": float(r[h])})
    while True:
        k = np.flatnonzero(keep)
        ev, V = np.linalg.eigh(R[np.ix_(k, k)])
        if ev[0] >= C.EIG_MIN:
            break
        load = np.abs(V[:, 0])
        load[fl[k]] = -1.0
        v = k[int(np.argmax(load))]
        keep[v] = False
        drops.append({"node": names[v], "reason": f"near-linear dependence (eigenvalue {ev[0]:.2e} < {C.EIG_MIN:g})",
                      "partner": "", "value": float(ev[0])})
    for d in drops:
        log(f"  drop {d['node']} before PC: {d['reason']} {d['partner']}")
    log(f"correlation matrix of the {int(keep.sum())} PC nodes: smallest eigenvalue {ev[0]:.3e}")
    return keep, D, R, drops


def choose_pc_depth(Rk, fl_k, n):
    """Largest depth <= PC_MAX_K whose CI-test upper bound from the marginal skeleton fits PC_TEST_BUDGET."""
    zc = norm.ppf(1 - C.PC_ALPHA / 2)
    zs = np.arctanh(np.clip(np.abs(Rk), 0, 1 - 1e-12)) * np.sqrt(n - 3)
    A0 = zs > zc
    np.fill_diagonal(A0, False)
    A0[np.ix_(fl_k, fl_k)] = False
    deg = A0.sum(axis=1)
    bound = [int(sum(int(d) * comb(max(int(d) - 1, 0), m) for d in deg)) for m in range(C.PC_MAX_K + 1)]
    cum = np.cumsum(bound)
    fits = [m for m in range(1, C.PC_MAX_K + 1) if cum[m] <= C.PC_TEST_BUDGET]
    max_k = max(fits) if fits else 1
    log(f"marginal skeleton: {int(A0.sum() // 2)} edges, max degree {int(deg.max())}; CI-test bound by depth "
        f"{dict(enumerate(bound))}; depth cap max_k = {max_k}")
    return max_k


def edge_list(G, names):
    """Edges of a causal-learn graph matrix: G[i,j] = -1 and G[j,i] = 1 means i -> j; -1/-1 undirected; 1/1 bidirected."""
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


def enforce_flare_sink(G, names_k, fl_k):
    """Orient any flare edge PC left undirected / bidirected / outgoing as X -> flare and remove flare-flare edges."""
    pos = {t: i for i, t in enumerate(names_k)}
    fixed = 0
    for s, t, ty in edge_list(G, names_k).itertuples(index=False):
        a, b = pos[s], pos[t]
        if fl_k[a] and fl_k[b]:
            G[a, b] = G[b, a] = 0
            fixed += 1
        elif fl_k[a]:
            G[b, a], G[a, b] = -1, 1
            fixed += 1
        elif fl_k[b] and ty != EDGE_DIRECTED:
            G[a, b], G[b, a] = -1, 1
            fixed += 1
    e = edge_list(G, names_k)
    fl_names = {n for n, f in zip(names_k, fl_k) if f}
    out_fl = int((e["source"].isin(fl_names) & (e["type"] == EDGE_DIRECTED)).sum())
    und_fl = int(((e["type"] != EDGE_DIRECTED) & (e["source"].isin(fl_names) | e["target"].isin(fl_names))).sum())
    log(f"flare edges re-oriented after PC: {fixed}")
    gate(out_fl == 0 and und_fl == 0, "flare tags are sinks: 0 outgoing and 0 undirected flare edges")
    return G, e


def run_pc_algorithm(N, flare_cols):
    """Run causal-learn PC (Fisher-Z, stable) on the node matrix; flare tags may not have outgoing edges."""
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    gate("max_k" in inspect.signature(pc).parameters, "installed causal-learn supports the depth cap max_k")
    names = list(N.columns)
    fl = np.array([c in flare_cols for c in names])
    D = N.iloc[::C.PC_ROW_STEP].to_numpy(dtype=np.float64)
    log(f"PC uses every {C.PC_ROW_STEP}th row: {len(D)} rows x {len(names)} nodes")
    keep, D, R, drops = drop_dependent_nodes(D, names, fl)
    k = np.flatnonzero(keep)
    names_k, fl_k = [names[i] for i in k], fl[k]
    gate(bool(fl_k.any()), "flare tag still present for PC")
    Dk = np.ascontiguousarray(D[:, k])
    max_k = choose_pc_depth(R[np.ix_(k, k)], fl_k, len(Dk))
    bk = BackgroundKnowledge()
    for i in range(len(names_k)):
        bk.add_node_to_tier(GraphNode(f"X{i + 1}"), 1 if fl_k[i] else 0)
    bk.forbid_within_tier(1)
    t0 = time.time()
    cg = pc(Dk, alpha=C.PC_ALPHA, indep_test="fisherz", stable=True, uc_rule=0, uc_priority=2, background_knowledge=bk,
            verbose=False, show_progress=False, max_k=max_k)
    G = np.array(cg.G.graph, dtype=int)
    log(f"PC (alpha {C.PC_ALPHA}, Fisher-Z, stable, flare in tier 1): {len(edge_list(G, names_k))} edges in "
        f"{time.time() - t0:.0f} s")
    G, edges = enforce_flare_sink(G, names_k, fl_k)
    return names_k, edges, pd.DataFrame(drops, columns=["node", "reason", "partner", "value"])


# ---- CPDAG TO DAG ----
def cpdag_to_dag(edges, names, flare_nodes):
    """Orient the CPDAG into a DAG with Dor-Tarsi (flare tags removed first as sinks) and return each node's parents."""
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

    def adj(x):
        return out[x] | inn[x] | und[x]

    def remove(x):
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

    def violations(x):
        ax = adj(x)
        return sum(1 for y in und[x] if not (ax - {y}) <= (adj(y) | {y}))

    flares = [v for v in names if v in flare_nodes]
    gate(all(not out[f] for f in flares), "no flare tag has an outgoing edge in the CPDAG")
    for f in flares:
        remove(f)
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
    gate(all(g.out_degree(f) == 0 for f in flares), "flare tags have no outgoing edge in the DAG")
    if counts["forced"] or counts["reversed"]:
        log(f"DEVIATION: CPDAG has no consistent extension: {counts['forced']} forced Dor-Tarsi steps, "
            f"{counts['reversed']} directed edges reversed to break cycles")
    parents = {v: sorted(g.predecessors(v), key=order.get) for v in names}
    dag_edges = pd.DataFrame(sorted(dag, key=lambda st: (order[st[1]], order[st[0]])), columns=["source", "target"])
    log(f"DAG: {len(dag)} edges, mean parents {np.mean([len(p) for p in parents.values()]):.2f}")
    return parents, dag_edges


def pc_subgraph_parents(parents, flare_nodes):
    """Parents of every PC node with the flare nodes removed; flare tags are sinks, so no PC loses a parent."""
    pcs = [v for v in parents if v not in flare_nodes]
    gate(all(q not in flare_nodes for v in pcs for q in parents[v]), "no PC has a flare tag as parent")
    return pcs, {v: list(parents[v]) for v in pcs}


# ---- T2 ----
def noc_correlation(X, noc):
    """Standardize the T² nodes with NOC mean / std and return the data and their NOC correlation matrix Σ."""
    Xn = X[noc]
    m, s = Xn.mean(axis=0), Xn.std(axis=0, ddof=1)
    gate(bool((s > 0).all()), "every T² node varies over NOC")
    Xs = (X - m) / s
    Xn = Xs[noc]
    S = Xn.T @ Xn / (len(Xn) - 1)
    S = (S + S.T) / 2
    np.fill_diagonal(S, 1.0)
    return Xs, S


def t2_control_limit(p):
    """Upper control limit chi²_p(1 − ALPHA0) of the T² chart (Eq. 6)."""
    return float(chi2.ppf(1 - C.ALPHA0, p))


def compute_t2(X, Sigma):
    """Hotelling T² = Zᵀ Σ⁻¹ Z for every row of standardized data X (Eq. 7); pseudo-inverse if Σ is singular."""
    try:
        L = np.linalg.cholesky(Sigma)
        return (solve_triangular(L, X.T, lower=True) ** 2).sum(axis=0), Sigma.shape[0]
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(Sigma)
        keep = vals > 1e-10 * vals.max()
        log(f"DEVIATION: Σ is singular; T² uses a pseudo-inverse on {int(keep.sum())} of {len(vals)} directions")
        return ((X @ (vecs[:, keep] / np.sqrt(vals[keep]))) ** 2).sum(axis=1), int(keep.sum())


# ---- T2 DECOMPOSITION ----
def fit_causal_regressions(S, nodes, parents):
    """Path coefficients β_j = Σ_PP⁻¹ Σ_Pj and conditional variances σ²_j = 1 − β_jᵀ Σ_Pj per node (p. 49, Eqs. 5, 11)."""
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


def decompose_t2_causal(X, B, sigma2):
    """Causal terms T_j·PA(j) = (Z_j − Σ_k β_k Z_pa_k) / σ_j·PA(j) for every row and node (Eqs. 9, 10)."""
    return (X - X @ B) / np.sqrt(sigma2)


def bn_implied_correlation(B, sigma2):
    """Covariance of Z implied by the linear Gaussian BN Z = BᵀZ + E: Σ_G = (I − Bᵀ)⁻¹ D (I − Bᵀ)⁻ᵀ (Eqs. 1-4)."""
    A = np.linalg.inv(np.eye(len(sigma2)) - B.T)
    Sg = A @ np.diag(sigma2) @ A.T
    return (Sg + Sg.T) / 2


def check_sum_identity(T, X, B, sigma2):
    """Assert Σ_j T²_j·PA(j) = Zᵀ Σ_G⁻¹ Z on every row (Definition 2, Eq. 13, Proposition 1) and return the sum."""
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
        return sum_t2
    direct = (solve_triangular(L, X.T, lower=True) ** 2).sum(axis=0)
    rel_g = float((np.abs(sum_t2 - direct) / np.maximum(direct, 1.0)).max())
    tol = max(C.IDENTITY_TOL, 1e3 * cond * np.finfo(float).eps)
    gate(rel_g < tol, f"same identity with Σ_G = (I-Bᵀ)⁻¹ D (I-Bᵀ)⁻ᵀ inverted numerically (condition number {cond:.1e}): "
                      f"max relative error {rel_g:.2e} < {tol:.1e}")
    return sum_t2


def check_noc_terms(T, noc, nodes, floored):
    """Assert every T_j·PA(j) has mean 0 and std 1 over NOC, as the N(0,1) reference distribution requires."""
    ok = np.array([t not in floored for t in nodes])
    Tn = T[noc][:, ok]
    m, s = float(np.abs(Tn.mean(axis=0)).max()), float(np.abs(Tn.std(axis=0, ddof=1) - 1).max())
    gate(m < 1e-3 and s < 1e-3, f"NOC T_j·PA(j): max |mean| {m:.1e}, max |std - 1| {s:.1e}")


# ---- PRE-FLARE EVALUATION ----
def evaluate_preflare_windows(idx, pre_eid, events, T, t2, t2_lim, z_lim, nodes):
    """Per pre-flare window: rows above the T² limit, the first one and its significant T_j·PA(j) (Fig. 5)."""
    rows = []
    for ev in events.itertuples(index=False):
        w = np.flatnonzero(pre_eid == ev.event_id)
        ooc = t2[w] > t2_lim
        row = {"event_id": ev.event_id, "onset": ev.start, "window_rows": len(w), "rows_t2_above_limit": int(ooc.sum()),
               "first_ooc": pd.NaT, "minutes_before_onset": np.nan, "t2_at_first_ooc": np.nan, "significant_pcs": ""}
        if ooc.any():
            i0 = w[int(np.argmax(ooc))]
            sig = np.flatnonzero(np.abs(T[i0]) > z_lim)
            sig = sig[np.argsort(-np.abs(T[i0, sig]))]
            row.update({"first_ooc": idx[i0], "minutes_before_onset": (ev.start - idx[i0]) / pd.Timedelta(minutes=1),
                        "t2_at_first_ooc": float(t2[i0]),
                        "significant_pcs": "; ".join(f"{nodes[j]} ({T[i0, j]:+.2f})" for j in sig)})
        rows.append(row)
    out = pd.DataFrame(rows)
    log(f"pre-flare windows: {int((out['window_rows'] > 0).sum())} of {len(out)} events have rows; "
        f"{int(out['first_ooc'].notna().sum())} reach T² > {t2_lim:.2f}")
    return out
