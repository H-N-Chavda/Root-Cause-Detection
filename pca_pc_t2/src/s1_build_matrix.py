"""Stage 1 - master matrix: resample and join the units, clean, split by time, z-score with TRAIN stats.

Writes outputs/data/{X_train, X_test, labels}.parquet, scaler.csv, s1_summary.json and cleaning tables.
"""
import numpy as np
import pandas as pd

from common import (DATA, DATA_ROOT, EXPECTED_N_PARAMS, FFILL_MAX_GAP, FLARE_BUFFER, K, MAX_MISSING,
                    MIN_ROW_COMPLETE, SENTINELS, TAB, TRAIN_FRAC, UNITS, gate, load_folder, load_json, log,
                    read_flare_events, read_tag_map, save_json, stage)


def ffill_short_gaps(df, max_gap):
    """Forward-fill only NaN runs of length <= max_gap that follow a valid value (longer gaps stay NaN).
    Returns (filled frame, histogram of NaN-run lengths, number of filled values)."""
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


stage("Stage 1 - build master matrix")
audit = load_json("audit.json")
interval = pd.Timedelta(seconds=audit["interval_s"])
order = {u: i for i, u in enumerate(UNITS)}
tagmap = read_tag_map()
tagmap = tagmap.assign(_o=tagmap["unit"].map(order)).sort_values(["_o", "column"]).drop(columns="_o")
unit_of = dict(zip(tagmap["column"], tagmap["unit"]))
log(f"resampling every folder to the dominant interval {interval} (mean of the samples in each bin)")

# ---- load, replace sentinels, resample, outer-join on timestamp
frames, sentinel_rows = [], []
for folder, g in tagmap.groupby("source_folder", sort=False):
    raw, _ = load_folder(DATA_ROOT / folder)
    rename = dict(zip(g["tag"], g["column"]))
    raw = raw[[c for c in raw.columns if c in rename]].replace([np.inf, -np.inf], np.nan)
    is_sent = raw.isin(SENTINELS)
    n_sent = is_sent.sum()
    sentinel_rows += [{"column": rename[c], "unit": unit_of[rename[c]], "replaced": int(v)}
                      for c, v in n_sent[n_sent > 0].items()]
    raw = raw.mask(is_sent).rename(columns=rename)
    frames.append(raw.resample(interval).mean())
    log(f"  {folder}: {raw.shape[1]} tags, {len(raw)} raw rows -> {len(frames[-1])} bins")
    del raw, is_sent
X = pd.concat(frames, axis=1).sort_index()
del frames
X = X[[c for c in tagmap["column"] if c in X.columns]]
X.index.name = "timestamp"
not_loaded = sorted(set(tagmap["column"]) - set(X.columns))
pd.DataFrame(sentinel_rows, columns=["column", "unit", "replaced"]).to_csv(TAB / "sentinels_replaced.csv", index=False)
log(f"joined matrix {X.shape}; sentinel values replaced: {sum(r['replaced'] for r in sentinel_rows)}; "
    f"mapped tags not loaded: {len(not_loaded)}")
gate(X.shape[1] > 0 and len(X) > 0, "joined matrix is non-empty")
shape_joined = X.shape

unit_cols = {u: [c for c in X.columns if unit_of[c] == u] for u in UNITS if any(unit_of[c] == u for c in X.columns)}
coverage = pd.DataFrame({u: X[cols].notna().mean(axis=1) for u, cols in unit_cols.items()}).resample("1D").mean()
coverage.to_csv(TAB / "coverage_daily.csv")
miss_before = X.isna().mean()

# ---- short-gap forward fill, then column and row drops
X, run_hist, n_filled = ffill_short_gaps(X, FFILL_MAX_GAP)
run_hist.rename_axis("nan_run_length").rename("count").to_csv(TAB / "gap_lengths.csv")
lens, cnts = run_hist.index.to_numpy(), run_hist.to_numpy()
short = lens <= FFILL_MAX_GAP
pct_runs = 100 * cnts[short].sum() / max(cnts.sum(), 1)
pct_vals = 100 * (lens * cnts)[short].sum() / max((lens * cnts).sum(), 1)
log(f"ffill N = {FFILL_MAX_GAP} samples ({FFILL_MAX_GAP * interval}): fills {pct_runs:.1f}% of NaN runs "
    f"= {pct_vals:.1f}% of missing values; {n_filled} values filled")

dropped = []
miss_after = X.isna().mean()
too_missing = miss_after[miss_after > MAX_MISSING].index
dropped += [{"column": c, "unit": unit_of[c], "reason": f"> {MAX_MISSING:.0%} missing after ffill",
             "value": round(100 * miss_after[c], 2)} for c in too_missing]
X = X.drop(columns=too_missing)
constant = X.columns[X.nunique() <= 1]
dropped += [{"column": c, "unit": unit_of[c], "reason": "zero variance", "value": 0.0} for c in constant]
X = X.drop(columns=constant)
miss_kept = X.isna().mean()
rows_before = len(X)
X = X[X.notna().mean(axis=1) >= MIN_ROW_COMPLETE]       # was: X.dropna()
log(f"dropped {len(too_missing)} tags (>30% missing), {len(constant)} constant tags; "
    f"rows with < {MIN_ROW_COMPLETE:.0%} of tags observed dropped: {rows_before - len(X)} of {rows_before}")
gate(len(X) > 10 * K, f"{len(X)} rows remain (need > {10 * K})")
log(f"dropped {len(too_missing)} tags (>30% missing), {len(constant)} constant tags; "
    f"incomplete rows dropped: {rows_before - len(X)} of {rows_before}")
gate(len(X) > 10 * K, f"{len(X)} complete rows remain (need > {10 * K})")

# ---- flare labels on the grid (a bin [t, t + interval) is flaring if it overlaps a flare window)
events = read_flare_events()
has_labels = bool(audit["has_labels"]) and len(events) > 0
idx = X.index
flare = np.zeros(len(idx), dtype=bool)
near = np.zeros(len(idx), dtype=bool)
for s, e in zip(events["start"], events["end"]):
    flare |= (idx > s - interval) & (idx <= e)
    near |= (idx > s - interval - FLARE_BUFFER) & (idx <= e + FLARE_BUFFER)

# ---- split by time: TRAIN = first 60 % of the time span minus flare windows +- buffer; TEST = the rest
cut = idx.min() + TRAIN_FRAC * (idx.max() - idx.min())
train = (idx < cut) & ~near if has_labels else (idx < cut)
split_rule = (f"TRAIN = timestamps before {cut} (first {TRAIN_FRAC:.0%} of time) minus flare windows "
              f"+- {FLARE_BUFFER}; TEST = all other rows (incl. every flare window)" if has_labels else
              f"no reliable flare labels: TRAIN = first {TRAIN_FRAC:.0%} of time (before {cut}), TEST = rest")
log(split_rule)
gate(train.sum() > 10 * K and (~train).sum() > 0, f"TRAIN rows {train.sum()}, TEST rows {(~train).sum()}")

const_train = X.columns[X[train].nunique() <= 1]
dropped += [{"column": c, "unit": unit_of[c], "reason": "constant within TRAIN", "value": 0.0} for c in const_train]
X = X.drop(columns=const_train)
pd.DataFrame(dropped, columns=["column", "unit", "reason", "value"]).to_csv(TAB / "dropped_tags.csv", index=False)
gate(X.shape[1] >= K, f"{X.shape[1]} tags remain (need >= {K} for the PCA)")

# ---- z-score with TRAIN statistics only
mu = X[train].mean()                    # pandas skips NaN, so these use observed TRAIN values only
sd = X[train].std()
Z = (X - mu) / sd
imputed = Z.isna()
n_imputed = int(imputed.to_numpy().sum())
imputed_frac = imputed.mean(axis=1)     # per-row share of imputed tags, saved for the T² stage
Z = Z.fillna(0.0)                       # 0 after z-scoring = TRAIN mean
log(f"mean-imputed {n_imputed} values = {100 * n_imputed / Z.size:.2f}% of the matrix")
X_train, X_test = Z[train], Z[~train]
gate(np.isfinite(X_train.to_numpy()).all() and np.isfinite(X_test.to_numpy()).all(), "no NaN / inf in X_train, X_test")

X_train.to_parquet(DATA / "X_train.parquet")
X_test.to_parquet(DATA / "X_test.parquet")
pd.DataFrame({"mean": mu, "std": sd}).rename_axis("column").to_csv(DATA / "scaler.csv")
pd.DataFrame({"flare": flare.astype(int), "near_flare": near.astype(int),
              "set": np.where(train, "TRAIN", "TEST"),
              "imputed_frac": imputed_frac.to_numpy()}, index=idx).to_parquet(DATA / "labels.parquet")

kept = set(X.columns)
mrows = []
for u in UNITS:
    all_cols = list(tagmap.loc[tagmap["unit"] == u, "column"])
    if not all_cols:
        continue
    loaded = [c for c in all_cols if c in miss_before.index]
    clean = [c for c in loaded if c in kept]
    mrows.append({"unit": u, "n_params_raw": len(all_cols), "n_params_clean": len(clean),
                  "missing_before_pct": 100 * miss_before[loaded].mean() if loaded else np.nan,
                  "missing_after_pct": 100 * miss_kept[clean].mean() if clean else np.nan})
pd.DataFrame(mrows).to_csv(TAB / "missing_by_unit.csv", index=False)

save_json({"interval": str(interval), "shape_joined": shape_joined, "shape_final": X.shape,
           "n_train": int(train.sum()), "n_test": int((~train).sum()), "cut": cut, "split_rule": split_rule,
           "has_labels": has_labels, "flare_rows_test": int((flare & ~train).sum()),
           "flare_rows_train": int((flare & train).sum()), "ffill_max_gap": FFILL_MAX_GAP,
           "ffill_pct_runs": pct_runs, "ffill_pct_values": pct_vals, "n_filled": n_filled,
           "n_dropped_missing": len(too_missing), "n_dropped_constant": len(constant),
           "n_dropped_constant_train": len(const_train), "rows_dropped_incomplete": rows_before - len(X),
           "n_sentinels_replaced": sum(r["replaced"] for r in sentinel_rows), "n_not_loaded": len(not_loaded),
           "n_imputed": n_imputed, "pct_imputed": 100 * n_imputed / Z.size, "min_row_complete": MIN_ROW_COMPLETE,
           "time_start": idx.min(), "time_end": idx.max()}, "s1_summary.json")

log(f"X_train {X_train.shape}, X_test {X_test.shape}; columns {X.shape[1]} vs expected {EXPECTED_N_PARAMS}")
gate(True, "X_train / X_test saved, shapes printed, no NaN/inf, column count reported")
