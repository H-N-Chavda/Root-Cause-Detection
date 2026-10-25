"""Stage 1 - load every tag, build tag_registry.csv, align on time, clean, standardize, save the combined matrix.

Reused from the earlier pipeline (log of what was reused is in outputs/LOG.md and SUMMARY.md):
- missing values: pca_pc_t2/src/s1_build_matrix.py as-is (sentinels -> NaN, ffill_short_gaps, drop tags > 30 %
  missing, drop rows < 90 % observed, z-score with reference statistics, leftover NaN -> 0 = reference mean);
- flare events: continuous-signal rule of pca_pc_t2/src/s0_audit.py audit_flare (signal > median + 3 * MAD);
- reference (normal operation): TRAIN rule of pca_pc_t2/src/s1_build_matrix.py (first 60 % of time minus
  flare windows +- 2 h).

Writes outputs/tag_registry.csv, outputs/data/{Z.parquet, meta.parquet, scaler.csv, flare_events.csv,
s1_summary.json} and outputs/tables/{duplicate_tags.csv, flare_thresholds.csv, gap_lengths.csv}.
"""
import re
import time

import numpy as np
import pandas as pd

from common import (ALL_TAGS_FOLDER, CONTROL_FOLDER, DATA, DATA_ROOT, EVENT_GAP, FFILL_MAX_GAP, FLARE_BUFFER,
                    FLARE_FOLDER, FLARE_MAD_K, MAX_MISSING, MIN_ROW_COMPLETE, NEAR_CONSTANT_SHARE, SENTINELS, TAB,
                    TRAIN_FRAC, UNIT_FOLDERS, assumption, clear_notes, deviation, done, ffill_short_gaps,
                    flags_to_events, gate, load_file, log, parquet_files, save_json, stage, write_registry)

S = "s1_load_clean"


def norm(name):
    """Unit-name key that ignores case and punctuation (SEU-1 == SEU1). Copied from pca_pc_t2 s0_audit."""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


NORM_UNITS = {norm(u): u for u in UNIT_FOLDERS}


def unit_from_text(text):
    """Longest unit name contained in a path (so ISOM_3 wins over ISOM), else None. Copied from s0_audit."""
    key = norm(text)
    hits = [u for k, u in NORM_UNITS.items() if k in key]
    return max(hits, key=len) if hits else None


def tag_key(name):
    """Deduplication key: tag name without spaces, case ignored."""
    return re.sub(r"\s+", "", str(name)).upper()


t0 = time.time()
stage("Stage 1 - load and clean")
clear_notes(S)
gate(DATA_ROOT.is_dir(), "DATA_ROOT in .env is an existing folder")
present = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
units = [u for u in UNIT_FOLDERS if u in present]
known = set(UNIT_FOLDERS) | {FLARE_FOLDER, ALL_TAGS_FOLDER, CONTROL_FOLDER}
log(f"folders under DATA_ROOT: {present}")
log(f"unit folders found: {len(units)} of {len(UNIT_FOLDERS)}; missing: "
    f"{[u for u in UNIT_FOLDERS if u not in present] or 'none'}; not named in .env (ignored): "
    f"{[f for f in present if f not in known] or 'none'}")
gate(len(units) > 0, "at least one unit folder exists")
gate(FLARE_FOLDER in present, f"flare folder '{FLARE_FOLDER}' exists (needed for the flare constraint)")

# ---- load all tags; a duplicate tag name keeps the copy from the first folder in this order
sources = [FLARE_FOLDER] + units + [f for f in (CONTROL_FOLDER, ALL_TAGS_FOLDER) if f in present]
assumption(S, "duplicate tag names (spaces and case ignored) keep the first copy in folder order "
              f"{' > '.join(sources)}; flare folder first so a flare tag is always marked is_flare")
reg_rows, dup_rows, intervals, folder_frames = [], [], [], {}
seen = {}                                   # tag key -> (folder, tag name kept)
for folder in sources:
    parts, files = {}, parquet_files(DATA_ROOT / folder)
    for p in files:
        num, info = load_file(p)
        if num is None:
            log(f"  skipped {info['file']}: no timestamp column")
            continue
        if pd.notna(info["median_interval"]):
            intervals.append(info["median_interval"])
        for c in num.columns:
            k = tag_key(c)
            if k in seen and seen[k][0] != folder:
                dup_rows.append({"tag": c, "duplicate_file": info["file"], "kept_folder": seen[k][0],
                                 "kept_tag": seen[k][1]})
                log(f"  duplicate: '{c}' in {info['file']} skipped, copy from {seen[k][0]} kept")
                continue
            if k not in seen:
                seen[k] = (folder, c)
                if folder == FLARE_FOLDER or folder in units:
                    unit = folder
                else:
                    unit = unit_from_text(info["file"]) or folder
                reg_rows.append({"tag": c, "source_folder": folder, "unit": unit,
                                 "is_flare": folder == FLARE_FOLDER, "file": info["file"]})
            parts.setdefault(seen[k][1], []).append(num[c].astype("float32"))   # same tag split over files
        del num
    if parts:
        cols = {}
        for c, ps in parts.items():
            s = pd.concat(ps).sort_index(kind="stable")
            cols[c] = s[~s.index.duplicated(keep="last")]
        folder_frames[folder] = pd.DataFrame(cols)
    log(f"  {folder}: {len(files)} files, {len(parts)} new tags")
pd.DataFrame(dup_rows, columns=["tag", "duplicate_file", "kept_folder", "kept_tag"]).to_csv(
    TAB / "duplicate_tags.csv", index=False)
log(f"tags loaded: {len(reg_rows)} unique, {len(dup_rows)} duplicate copies skipped (tables/duplicate_tags.csv)")
gate(len(reg_rows) > 0 and len(intervals) > 0, "at least one timestamped tag was loaded")

order = {f: i for i, f in enumerate(units + [CONTROL_FOLDER, ALL_TAGS_FOLDER, FLARE_FOLDER])}
reg = pd.DataFrame(reg_rows)
reg = reg.assign(_o=reg["source_folder"].map(order)).sort_values(["_o", "unit", "tag"], kind="stable")
reg = reg.drop(columns="_o").reset_index(drop=True)
is_flare = dict(zip(reg["tag"], reg["is_flare"]))
gate(reg["is_flare"].any(), "at least one flare tag was loaded")

# ---- sentinels -> NaN, resample every folder to the most common sampling interval, join on timestamp
iv = pd.to_timedelta(pd.Series(intervals)).dt.round("1s")
counts = iv[iv > pd.Timedelta(0)].value_counts()
gate(len(counts) > 0, "a positive sampling interval could be measured")
interval = max(counts[counts == counts.max()].index)
assumption(S, f"all tags resampled to the most common file sampling interval {interval} (bin mean), "
              "as in pca_pc_t2")
frames, n_sent, flare_raw = [], 0, None
for folder in list(folder_frames):
    f = folder_frames.pop(folder).replace([np.inf, -np.inf], np.nan)
    is_sent = f.isin(SENTINELS)
    n_sent += int(is_sent.to_numpy().sum())
    f = f.mask(is_sent).resample(interval).mean().astype("float32")
    if folder == FLARE_FOLDER:
        flare_raw = f.copy()
    frames.append(f)
    del is_sent
X = pd.concat(frames, axis=1).sort_index()
del frames
X = X[list(reg["tag"])]
X.index.name = "timestamp"
log(f"joined matrix {X.shape} on a {interval} grid; sentinel values set to NaN: {n_sent}")

# ---- flare events (pca_pc_t2 s0_audit rule for continuous flare signals)
med = flare_raw.median()
mad = 1.4826 * (flare_raw - med).abs().median()
thr = med + FLARE_MAD_K * mad
flagged = (flare_raw > thr).any(axis=1).to_numpy()
events = flags_to_events(flare_raw.index, flagged, EVENT_GAP * interval)
events = events.assign(event_id=np.arange(1, len(events) + 1))[["event_id", "start", "end", "n_flagged"]]
events.to_csv(DATA / "flare_events.csv", index=False)
pd.DataFrame({"flare_tag": flare_raw.columns, "median": med.values, "mad_scaled": mad.values, "threshold": thr.values,
              "pct_above": 100 * (flare_raw > thr).mean().values}).to_csv(TAB / "flare_thresholds.csv", index=False)
assumption(S, f"flaring = any flare tag > median + {FLARE_MAD_K:g} * MAD (MAD x 1.4826), flags <= {EVENT_GAP} samples "
              "apart merged into one event (pca_pc_t2 s0_audit rule)")
log(f"flare events: {len(events)}; flagged samples: {100 * flagged.mean():.1f}% of the flare grid")
del flare_raw

# ---- missing values and tag drops (pca_pc_t2 s1_build_matrix rules)
reg["missing_pct_raw"] = (100 * X.isna().mean()).reindex(reg["tag"]).round(3).to_numpy()
status = {t: "in_matrix" for t in reg["tag"]}


def drop(cols, reason):
    for c in cols:
        status[c] = f"dropped_s1: {reason(c)}"
        log(f"  drop {c}: {reason(c)}")
    return X.drop(columns=list(cols))


X, run_hist, n_filled = ffill_short_gaps(X, FFILL_MAX_GAP)
run_hist.rename_axis("nan_run_length").rename("count").to_csv(TAB / "gap_lengths.csv")
log(f"forward-filled {n_filled} values (NaN runs <= {FFILL_MAX_GAP} samples)")
miss = X.isna().mean()
X = drop(miss[miss > MAX_MISSING].index, lambda c: f"{100 * miss[c]:.1f}% missing after ffill (> {MAX_MISSING:.0%})")
nun = X.nunique()
X = drop(nun[nun <= 1].index, lambda c: "constant")
top_share = X.apply(lambda s: s.value_counts(normalize=True).iloc[0])
near = [c for c in X.columns if top_share[c] > NEAR_CONSTANT_SHARE]
flare_near = [c for c in near if is_flare[c]]
if flare_near:
    assumption(S, f"flare tags kept although near-constant (needed for the flare constraint): {flare_near}")
X = drop([c for c in near if not is_flare[c]],
         lambda c: f"near-constant ({100 * top_share[c]:.2f}% one value > {NEAR_CONSTANT_SHARE:.0%})")
rows_before = len(X)
X = X[X.notna().mean(axis=1) >= MIN_ROW_COMPLETE]
log(f"rows with < {MIN_ROW_COMPLETE:.0%} of tags observed dropped: {rows_before - len(X)} of {rows_before}")
gate(len(X) > 1000, f"{len(X)} rows remain after the row filter")

# ---- duplicate columns (identical values); the later one is dropped, a flare tag is never the one dropped
fp = pd.Series({c: int(pd.util.hash_pandas_object(X[c], index=False).sum()) for c in X.columns})
dups = {}
for _, cols in fp.groupby(fp).groups.items():
    cols = sorted(cols, key=lambda c: (not is_flare[c], list(X.columns).index(c)))
    for c in cols[1:]:
        if X[c].equals(X[cols[0]]):
            dups[c] = cols[0]
X = drop(list(dups), lambda c: f"duplicate column of {dups[c]}")

# ---- flare labels on the grid and the reference period
idx = X.index
flare = np.zeros(len(idx), dtype=bool)
near_flare = np.zeros(len(idx), dtype=bool)
for s, e in zip(events["start"], events["end"]):
    flare |= (idx > s - interval) & (idx <= e)
    near_flare |= (idx > s - interval - FLARE_BUFFER) & (idx <= e + FLARE_BUFFER)
cut = idx.min() + TRAIN_FRAC * (idx.max() - idx.min())
ref = (idx < cut) & ~near_flare
min_ref = 5 * X.shape[1]
ref_rule = (f"reference = rows before {cut} (first {TRAIN_FRAC:.0%} of time) minus flare windows +- {FLARE_BUFFER} "
            "(pca_pc_t2 TRAIN rule = the repo's normal-operation definition)")
if ref.sum() < min_ref:
    ref = idx < cut
    ref_rule = f"reference = rows before {cut} (first {TRAIN_FRAC:.0%} of time), flare windows NOT excluded"
    deviation(S, f"flare-free reference had < {min_ref} rows (5 x tags); fell back to: {ref_rule}")
assumption(S, ref_rule)
log(f"flare rows: {100 * flare.mean():.1f}% of the matrix; reference rows: {int(ref.sum())}")
gate(ref.sum() >= min_ref, f"reference rows {int(ref.sum())} >= {min_ref} (5 x tags)")

# ---- tags constant within the reference period cannot be standardized with it
nun_ref = X[ref].nunique()
const_ref = list(nun_ref[nun_ref <= 1].index)
pc_only = [c for c in const_ref if is_flare[c]]
X = drop([c for c in const_ref if not is_flare[c]], lambda c: "constant within the reference period")
if pc_only:
    assumption(S, f"flare tags constant within the reference period kept for PC only, standardized with all rows "
                  f"(a flare has no children, so leaving it out of T² changes no other term): {pc_only}")

# ---- z-score with reference statistics, leftover NaN -> 0 (= reference mean), column by column to save memory
mu, sd = X[ref].mean(), X[ref].std()
for c in pc_only:
    mu[c], sd[c] = X[c].mean(), X[c].std()
imputed_frac = X.isna().mean(axis=1)
n_imputed = int(X.isna().to_numpy().sum())
for c in X.columns:
    X[c] = ((X[c] - mu[c]) / sd[c]).fillna(0.0).astype("float32")
Z = X
del X
log(f"reference-mean imputed {n_imputed} values = {100 * n_imputed / Z.size:.2f}% of the matrix")
arr = Z.to_numpy()
gate(bool(np.isfinite(arr).all()), "combined matrix has no NaN / inf")
gate(bool((arr.std(axis=0) > 0).all()), "combined matrix has no constant column")
del arr

for c in pc_only:
    status[c] = "in_matrix: flare, PC only (constant in reference)"
reg["status"] = reg["tag"].map(status)
write_registry(reg)
n_in = int(reg["status"].str.startswith("in_matrix").sum())
gate(n_in == Z.shape[1], f"tag count in matrix {Z.shape[1]} == in_matrix rows of tag_registry.csv ({n_in})")

Z.to_parquet(DATA / "Z.parquet")
pd.DataFrame({"flare": flare, "near_flare": near_flare, "reference": ref,
              "imputed_frac": imputed_frac.to_numpy()}, index=idx).to_parquet(DATA / "meta.parquet")
pd.DataFrame({"mean": mu, "std": sd}).rename_axis("tag").to_csv(DATA / "scaler.csv")
kinds = ("missing", "near-constant", "reference", "duplicate", "constant")
drops = reg.loc[reg["status"].str.startswith("dropped_s1"), "status"].map(
    lambda s: next(k for k in kinds if k in s))
seconds = time.time() - t0
save_json({"interval": str(interval), "n_tags_loaded": len(reg), "n_duplicate_copies": len(dup_rows),
           "n_flare_tags_loaded": int(reg["is_flare"].sum()), "n_tags_matrix": Z.shape[1],
           "n_flare_tags_matrix": int(sum(is_flare[c] for c in Z.columns)), "flare_pc_only": pc_only,
           "n_dropped_s1": int(reg["status"].str.startswith("dropped_s1").sum()),
           "dropped_s1_by_reason": {k: int(v) for k, v in drops.value_counts().items()},
           "rows": len(Z), "rows_dropped_incomplete": rows_before - len(Z), "time_start": idx.min(),
           "time_end": idx.max(), "n_sentinels": n_sent, "n_filled": n_filled, "n_imputed": n_imputed,
           "pct_imputed": 100 * n_imputed / Z.size, "n_flare_events": len(events), "pct_flare_rows": 100 * flare.mean(),
           "n_reference_rows": int(ref.sum()), "reference_rule": ref_rule, "seconds": seconds}, "s1_summary.json")
log(f"combined matrix {Z.shape} saved; tags per unit: {reg[reg['status'].str.startswith('in_matrix')]['unit'].value_counts().to_dict()}")
done(S, seconds)
