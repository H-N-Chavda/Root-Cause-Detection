"""Stage 0 - data audit: files, timestamps, sampling, per-unit quality, tag -> unit map, flare labels.

Writes outputs/00_data_audit.md, outputs/data/{audit.json, tag_unit_map.csv, flare_events.csv}
and the per-file / per-tag tables in outputs/tables/.
"""
import re
from datetime import date

import numpy as np
import pandas as pd

from common import (CONTROL_FOLDER, DATA, DATA_ROOT, EVENT_GAP, EXPECTED_N_PARAMS, FLAG_HINTS,
                    FLARE_FOLDER, FLARE_MAD_K, OUT, SENTINELS, TAB, TAG_FOLDER, UNITS,
                    find_time_column, flags_to_events, gate, load_file, load_folder, log, md_table,
                    median_interval, numeric_part, parquet_files, read_table, save_json, stage,
                    to_datetime_or_none)


def norm(name):
    """Unit-name key that ignores case and punctuation (SEU-1 == SEU1)."""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


NORM_UNITS = {norm(u): u for u in UNITS}


def unit_from_text(text):
    """Longest unit name contained in a path or label (so ISOM_3 wins over ISOM), else None."""
    key = norm(text)
    hits = [u for k, u in NORM_UNITS.items() if k in key]
    return max(hits, key=len) if hits else None


def per_tag_stats(df, unit):
    """Missing %, constant / near-constant, sentinels, zero floods and robust outliers per tag."""
    n_valid = df.notna().sum()
    n_unique = df.nunique()
    top_freq = df.apply(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan)
    med = df.median()
    mad = 1.4826 * (df - med).abs().median()
    robust_z = (df - med).abs() / mad.replace(0, np.nan)
    st = pd.DataFrame({
        "tag": df.columns, "unit": unit,
        "missing_pct": 100 * (1 - n_valid / len(df)),
        "std": df.std(), "n_unique": n_unique,
        "constant": n_unique <= 1,
        "near_constant": (top_freq > 0.99) & (n_unique > 1),
        "sentinel_count": df.isin(SENTINELS).sum(),
        "zero_pct": 100 * (df == 0).sum() / n_valid.replace(0, np.nan),
        "outliers_robust_z_gt_10": (robust_z > 10).sum(),
        "min": df.min(), "max": df.max(),
    })
    return st.reset_index(drop=True)


def parse_meta(meta, all_tags):
    """Find the tag / unit / description columns of the tag master list."""
    tag_col, best = None, 0
    for c in meta.columns:
        hits = len(set(meta[c].dropna().astype(str).str.strip()) & all_tags)
        if hits > best:
            tag_col, best = c, hits
    if tag_col is None or best < 0.5 * len(all_tags):
        return {}, {}, (f"not usable: no column matches >= 50 % of the unit tag names "
                        f"(best '{tag_col}', {best} of {len(all_tags)})")
    unit_col = None
    for c in meta.columns:
        v = meta[c].dropna().astype(str).map(norm)
        if c != tag_col and len(v) and v.isin(list(NORM_UNITS)).mean() >= 0.5:
            unit_col = c
            break
    desc_col = next((c for c in meta.columns if "desc" in c.lower() and c != tag_col), None)
    m = meta.dropna(subset=[tag_col]).drop_duplicates(subset=[tag_col])
    tags = m[tag_col].astype(str).str.strip()
    meta_unit = dict(zip(tags, m[unit_col].astype(str).map(norm).map(NORM_UNITS))) if unit_col else {}
    meta_desc = dict(zip(tags, m[desc_col].astype(str))) if desc_col else {}
    note = (f"usable: tag column '{tag_col}' matches {best} of {len(all_tags)} unit tags; "
            f"unit column '{unit_col}'; description column '{desc_col}'")
    return meta_unit, meta_desc, note


def audit_flare(files):
    """Describe flare_data and derive flare event windows.
    Priority: explicit start/end windows > binary flag column > continuous signal above median + 3*MAD."""
    folder = DATA_ROOT / FLARE_FOLDER
    empty = pd.DataFrame({"start": pd.Series(dtype="datetime64[ns]"), "end": pd.Series(dtype="datetime64[ns]")})
    if not folder.is_dir():
        return empty, ["folder not found"], "none", "no flare labels (folder missing)", None
    desc, windows, flags, signals = [], [], [], []
    for p in parquet_files(folder):
        df = read_table(p)
        rel = str(p.relative_to(DATA_ROOT))
        tcol, t = find_time_column(df)
        files.append({"folder": FLARE_FOLDER, "file": rel, "rows": len(df), "columns": df.shape[1], "time_column": tcol})
        shown = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns[:30])
        desc.append(f"`{rel}`: {len(df)} rows, {df.shape[1]} columns: {shown}{' ...' if df.shape[1] > 30 else ''}")
        starts = [s for s in (to_datetime_or_none(df[c]) for c in df.columns if "start" in c.lower()) if s is not None]
        ends = [e for e in (to_datetime_or_none(df[c]) for c in df.columns
                            if "end" in c.lower() or "stop" in c.lower()) if e is not None]
        if starts and ends:
            windows.append(pd.DataFrame({"start": starts[0].to_numpy(), "end": ends[0].to_numpy()}))
            continue
        if tcol is None:
            continue
        num, _, _ = numeric_part(df.drop(columns=[tcol]))
        num.index = pd.DatetimeIndex(t.to_numpy())
        num = num[~num.index.isna()]
        for c in num.columns:
            s = num[c].groupby(level=0).max().rename(f"{rel}:{c}")
            v = s.dropna()
            is_flag = len(v) > 0 and v.isin([0, 1]).all() and any(h in c.lower() for h in FLAG_HINTS)
            (flags if is_flag else signals).append(s)

    thresholds = None
    if windows:
        ev = pd.concat(windows).dropna()
        ev = ev[ev["end"] >= ev["start"]].sort_values("start").reset_index(drop=True)
        fmt_ = "event windows (start/end columns)"
        label = "flaring = inside a [start, end] window from flare_data (unambiguous)"
    elif flags:
        F = pd.concat(flags, axis=1).sort_index()
        ev = flags_to_events(F.index, (F == 1).any(axis=1).to_numpy(), EVENT_GAP * median_interval(F.index))
        fmt_ = f"binary flag column(s): {', '.join(F.columns)}"
        label = f"flaring = any flag == 1; flags <= {EVENT_GAP} native samples apart merged"
    elif signals:
        S = pd.concat(signals, axis=1).sort_index()
        med = S.median()
        mad = 1.4826 * (S - med).abs().median()
        thr = med + FLARE_MAD_K * mad
        thresholds = pd.DataFrame({"signal": S.columns, "median": med.values, "mad_scaled": mad.values,
                                   "threshold": thr.values, "pct_above": 100 * (S > thr).mean().values})
        ev = flags_to_events(S.index, (S > thr).any(axis=1).to_numpy(), EVENT_GAP * median_interval(S.index))
        fmt_ = f"continuous signal(s) (no start/end windows): {', '.join(S.columns)}"
        label = (f"flaring = any flare signal > median + {FLARE_MAD_K:g} * MAD (MAD scaled by 1.4826); "
                 f"runs <= {EVENT_GAP} native samples apart merged. Option chosen by the project owner; "
                 "alternatives were 'any non-zero flow' and 'top 5 %'")
    else:
        return empty, desc, "nothing usable (no windows, no timestamped numeric columns)", "no flare labels", None
    return ev[["start", "end"]], desc, fmt_, label, thresholds


# ============================================================================= main
stage("Stage 0 - data audit")
gate(str(DATA_ROOT) not in ("", ".") and DATA_ROOT.is_dir(),
     f"DATA_ROOT is an existing folder: '{DATA_ROOT}' (set it in pca_pc_t2/.env)")

folders = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
units_found = [u for u in UNITS if u in folders]
missing_units = [u for u in UNITS if u not in folders]
unexpected = [f for f in folders if f not in UNITS and f not in (TAG_FOLDER, FLARE_FOLDER, CONTROL_FOLDER)]
log(f"folders: {folders}")
gate(len(units_found) > 0, f"unit folders found: {len(units_found)} of {len(UNITS)}")

# ---- units
files, tag_stats, unit_rows, unit_tags = [], [], [], {}
for u in units_found:
    raw, infos = load_folder(DATA_ROOT / u)
    files += [dict(i, folder=u) for i in infos]
    if raw.empty:
        log(f"  {u}: no readable parquet file with a timestamp column")
        continue
    dup = int(raw.index.duplicated().sum())
    df = raw.groupby(level=0).mean()     # one row per timestamp (files can split rows or columns)
    del raw
    unit_tags[u] = list(df.columns)
    st = per_tag_stats(df, u)
    tag_stats.append(st)
    unit_rows.append({"unit": u, "files": len(infos), "parameters": df.shape[1], "timestamps": len(df),
                      "start": df.index.min(), "end": df.index.max(),
                      "median_interval": median_interval(df.index),
                      "missing_pct": st["missing_pct"].mean(), "constant": int(st["constant"].sum()),
                      "near_constant": int(st["near_constant"].sum()),
                      "tags_with_sentinels": int((st["sentinel_count"] > 0).sum()),
                      "zero_flood_tags_gt50pct": int((st["zero_pct"] > 50).sum()),
                      "tags_with_outliers": int((st["outliers_robust_z_gt_10"] > 0).sum()),
                      "duplicated_timestamps": dup})
    log(f"  {u}: {df.shape[1]} tags, {len(df)} timestamps, {len(infos)} files")
    del df
gate(len(unit_tags) > 0, "at least one unit has readable timestamped data")
units = list(unit_tags)
all_tags = set().union(*unit_tags.values())

# ---- control data (excluded unless it reconciles the count to 418)
ctrl_tags = {}
ctrl_dir = DATA_ROOT / CONTROL_FOLDER
if ctrl_dir.is_dir():
    for p in parquet_files(ctrl_dir):
        num, info = load_file(p)
        files.append(dict(info, folder=CONTROL_FOLDER))
        if num is not None:
            guess = unit_from_text(str(p.relative_to(ctrl_dir)))
            for c in num.columns:
                ctrl_tags.setdefault(c, guess)

# ---- tag master list
meta_unit, meta_desc, meta_note = {}, {}, f"folder '{TAG_FOLDER}' not found; folder membership used"
tag_dir = DATA_ROOT / TAG_FOLDER
if tag_dir.is_dir():
    metas, ts_notes = [], []
    for p in parquet_files(tag_dir):
        df = read_table(p)
        tcol, _ = find_time_column(df)
        files.append({"folder": TAG_FOLDER, "file": str(p.relative_to(DATA_ROOT)), "rows": len(df),
                      "columns": df.shape[1], "time_column": tcol})
        if tcol is None:
            metas.append(df)
        else:
            ts_notes.append(f"{p.name}: time series, {len(set(df.columns) & all_tags)} of its "
                            f"{df.shape[1]} columns are unit tags")
    if metas:
        meta_unit, meta_desc, meta_note = parse_meta(pd.concat(metas, ignore_index=True), all_tags)
    else:
        meta_note = "contains time-series tables, not tag metadata; folder membership used. " + "; ".join(ts_notes)
log(f"{TAG_FOLDER}: {meta_note}")

# ---- tag -> unit map
rows = []
for u in units:
    for t in unit_tags[u]:
        mu = meta_unit.get(t)
        from_meta = mu in unit_tags
        rows.append({"tag": t, "unit": mu if from_meta else u, "folder_unit": u, "source_folder": u,
                     "unit_source": TAG_FOLDER if from_meta else "folder", "description": meta_desc.get(t, "")})
n_unit_params = len(rows)
ctrl_unique = sorted(t for t in ctrl_tags if t not in all_tags)
ctrl_unit = {t: meta_unit.get(t) if meta_unit.get(t) in unit_tags else ctrl_tags[t] for t in ctrl_unique}
include_control = (bool(ctrl_unique) and n_unit_params != EXPECTED_N_PARAMS
                   and n_unit_params + len(ctrl_unique) == EXPECTED_N_PARAMS
                   and all(ctrl_unit[t] in unit_tags for t in ctrl_unique))
if include_control:
    rows += [{"tag": t, "unit": ctrl_unit[t], "folder_unit": ctrl_unit[t], "source_folder": CONTROL_FOLDER,
              "unit_source": "control folder path / metadata", "description": meta_desc.get(t, "")}
             for t in ctrl_unique]
tm = pd.DataFrame(rows)
dup_tag = tm.duplicated("tag", keep=False)
tm["column"] = np.where(dup_tag, tm["tag"] + "@" + tm["source_folder"], tm["tag"])
tm = tm[["column", "tag", "unit", "source_folder", "folder_unit", "unit_source", "description"]]
tm.to_csv(DATA / "tag_unit_map.csv", index=False)
conflicts = tm[(tm["unit_source"] == TAG_FOLDER) & (tm["unit"] != tm["folder_unit"])]
conflicts.to_csv(TAB / "tag_unit_conflicts.csv", index=False)
n_params = len(tm)

# ---- sampling interval
iv = pd.to_timedelta(pd.Series({r["unit"]: r["median_interval"] for r in unit_rows})).dropna().dt.round("1s")
counts = iv.value_counts()
gate(len(counts) > 0, "a sampling interval could be measured")
dominant = max(counts[counts == counts.max()].index)
gate(dominant > pd.Timedelta(0), f"dominant sampling interval = {dominant}")

# ---- flare data
flare_ev, flare_desc, flare_fmt, label_def, thresholds = audit_flare(files)
flare_ev = flare_ev.assign(event_id=np.arange(1, len(flare_ev) + 1))[["event_id", "start", "end"]]
flare_ev.to_csv(DATA / "flare_events.csv", index=False)
if thresholds is not None:
    thresholds.to_csv(TAB / "flare_thresholds.csv", index=False)
has_labels = len(flare_ev) > 0

# ---- tables
files_df = pd.DataFrame(files)
files_df.to_csv(TAB / "audit_files.csv", index=False)
tags_df = pd.concat(tag_stats, ignore_index=True)
tags_df.to_csv(TAB / "audit_tags.csv", index=False)
units_df = pd.DataFrame(unit_rows)
units_df.to_csv(TAB / "audit_units.csv", index=False)
t_start, t_end = units_df["start"].min(), units_df["end"].max()
unit_files = files_df[files_df["folder"].isin(units)]
time_cols = sorted({str(c) for c in unit_files["time_column"].dropna()})
time_types = sorted({str(c) for c in unit_files["time_dtype"].dropna()})
no_time = unit_files.loc[unit_files["time_column"].isna(), "file"].tolist()

save_json({"units": units, "missing_units": missing_units, "unexpected_folders": unexpected,
           "n_params": n_params, "n_unit_params": n_unit_params, "expected_n_params": EXPECTED_N_PARAMS,
           "include_control": include_control, "n_control_unique_tags": len(ctrl_unique),
           "interval_s": dominant.total_seconds(), "time_start": t_start, "time_end": t_end,
           "time_columns": time_cols, "has_labels": has_labels, "n_flare_events": len(flare_ev),
           "flare_format": flare_fmt, "label_definition": label_def, "tag_map_source": meta_note,
           "tag_name_collisions": int(dup_tag.sum())}, "audit.json")

# ---- markdown report
folder_rows = pd.DataFrame([{"folder": f,
                             "role": ("unit" if f in UNITS else "tag master list" if f == TAG_FOLDER else
                                      "flare data" if f == FLARE_FOLDER else
                                      "control data" if f == CONTROL_FOLDER else "UNEXPECTED"),
                             "parquet_files": len(parquet_files(DATA_ROOT / f))} for f in folders])
recon = "MATCH" if n_params == EXPECTED_N_PARAMS else f"DISCREPANCY ({n_params - EXPECTED_N_PARAMS:+d})"
md = ["# 00 Data audit", "",
      f"Generated by `s0_audit.py` on {date.today().isoformat()}. DATA_ROOT = `{DATA_ROOT}`. Raw data read-only.", "",
      "## Folders", md_table(folder_rows), "",
      f"- Units with data: {len(units)} of 13. Missing folders: {missing_units or 'none'}. "
      f"Unexpected folders: {unexpected or 'none'}.",
      "- Per-file listing (rows, columns, time column, range, interval): `outputs/tables/audit_files.csv`.", "",
      "## Parameters",
      f"- Parameter count used: **{n_params}** (expected {EXPECTED_N_PARAMS}) -> {recon}. Units: {len(units)} (expected 13).",
      f"- Tags in unit folders: {n_unit_params}. `{CONTROL_FOLDER}`: {len(ctrl_tags)} tags, "
      f"{len(ctrl_tags) - len(ctrl_unique)} also in unit folders, {len(ctrl_unique)} unique -> "
      f"{'INCLUDED (unit count + control-unique tags = 418 exactly, all mappable to a unit)' if include_control else 'EXCLUDED'}.",
      f"- Tag-name collisions across folders: {int(dup_tag.sum())} rows (renamed `tag@FOLDER`).",
      f"- `{TAG_FOLDER}`: {meta_note}.",
      f"- Metadata-vs-folder unit conflicts: {len(conflicts)} (`outputs/tables/tag_unit_conflicts.csv`; metadata wins).",
      "- Tag -> unit map saved: `outputs/data/tag_unit_map.csv`.", "",
      "## Time",
      f"- Timestamp column name(s) in unit files: {time_cols}; dtype(s): {time_types}; "
      f"consistent: {'yes' if len(time_cols) == 1 else 'NO - names differ, each file parsed on its own'}. "
      "tz-aware stamps converted to UTC (naive).",
      f"- Unit files without a usable timestamp (skipped): {no_time or 'none'}.",
      f"- Time range (all units): {t_start} to {t_end}.",
      f"- Dominant sampling interval (most common per-unit median): **{dominant}** -> Stage 1 resamples to it (bin mean).", "",
      "## Per-unit quality (before cleaning; per-tag detail in `outputs/tables/audit_tags.csv`)",
      md_table(units_df[["unit", "files", "parameters", "timestamps", "start", "end", "median_interval",
                         "missing_pct", "constant", "near_constant", "tags_with_sentinels",
                         "zero_flood_tags_gt50pct", "tags_with_outliers", "duplicated_timestamps"]]), "",
      "## Flare data", *[f"- {d}" for d in flare_desc],
      f"- Format detected: {flare_fmt}.", f"- Fault label: {label_def}.",
      f"- Flare events: {len(flare_ev)}"
      + (f"; first {flare_ev['start'].min()}, last {flare_ev['end'].max()}; "
         f"total {(flare_ev['end'] - flare_ev['start']).sum()}" if has_labels else ""), "",
      "## Assumptions",
      "- Unit folders = the 13 expected names; files found recursively (`*.parquet`).",
      "- Text values in numeric tags (e.g. 'Bad Input') are treated as missing.",
      f"- Sentinel values {SENTINELS} become NaN in Stage 1 (project-owner decision).",
      "- Near-constant = one value covers > 99 % of samples; zero flood = > 50 % exact zeros; "
      "outlier = robust z (median/MAD) > 10. Reported only; not removed here.",
      "- Duplicated timestamps counted after stacking a unit's files; files that split columns also produce them.", ""]
(OUT / "00_data_audit.md").write_text("\n".join(md), encoding="utf-8")

log(f"parameters: {n_params} (expected {EXPECTED_N_PARAMS}) -> {recon}; units {len(units)}/13")
log(f"time range {t_start} .. {t_end}; dominant interval {dominant}; flare events {len(flare_ev)} ({flare_fmt})")
gate((OUT / "00_data_audit.md").exists() and (DATA / "tag_unit_map.csv").exists(),
     "audit report states parameter count, time range, interval; tag_unit_map.csv saved")
