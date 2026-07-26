#!/usr/bin/env python3
"""
backfill_multimodel_history.py — Fill mm_gfs_max, mm_ecmwf_max, mm_nbm_max,
mm_icon_max, mm_gem_max, mm_gem_hrdps_max into multiyear_atmospheric.csv from
Open-Meteo's historical-forecast-api.

Why this exists:
  v11 was anchored on mm_hrrr_vs_nws — but the NWS forecast log only goes back
  ~10 months. The multi-year atmospheric CSV has 4+ years of data, but only
  the dates that overlap with NWS log get a usable v11 signal (~144 rows).

  Solution: anchor v11's divergence on MODELS we have 4-year history for
  instead of NWS. mm_hrrr_max is already populated 4yr in the CSV. This
  script fills the other multi-model columns so we can compute:
    mm_hrrr_vs_gfs    — high-res mesoscale vs synoptic global (real signal)
    mm_hrrr_vs_ecmwf  — HRRR vs the #1 global deterministic model
    mm_hrrr_vs_nbm    — HRRR vs the #2-accuracy blend
  All three have 4+ years of populated history once this backfill runs.

Safety:
  - Backs up CSV to .bak-{timestamp} before write.
  - Fills only where existing is NaN.
  - Recomputes mm_mean / mm_std / mm_spread from all populated mm_*_max cols.
  - Polite 1s sleep between API calls.

Usage:
    python backfill_multimodel_history.py --city nyc
    python backfill_multimodel_history.py --city lax
    python backfill_multimodel_history.py --city nyc --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
# Previous Runs API — archives what each model PREDICTED at earlier lead times.
# temperature_2m_previous_day1 = the value the run from ~24h earlier forecast
# for that hour. Daily max of it = the day-before HRRR high forecast, i.e. the
# same lead the canonical night-before prediction anchors on at inference.
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Coordinates/timezone come from city_config — the single source of truth.
# (A previous version hardcoded downtown LA 34.05,-118.24 for "lax", ~15mi
# from KLAX airport and a different microclimate on marine-layer days.)
from city_config import CITIES as _CITY_CFG

CITIES = {
    key: {
        "lat": cfg["open_meteo_lat"], "lon": cfg["open_meteo_lon"],
        "tz": cfg["timezone"],
        "csv": f"{cfg['model_prefix']}multiyear_atmospheric.csv",
    }
    for key, cfg in _CITY_CFG.items()
}

# Map open-meteo model id -> multiyear CSV column name.
# NOTE: mm_hrrr_max is NOT here by default — live prediction-time captures
# must never be overwritten. Pass --include-hrrr to fill it where NaN only
# (needed for LAX, which launched Feb 2026 and has no 4yr HRRR history, so
# the v16 residual model has no anchor and skips training entirely).
HRRR_MODEL = {"ncep_hrrr_conus": "mm_hrrr_max"}
MODELS = {
    "gfs_seamless":          "mm_gfs_max",
    "ecmwf_ifs025":          "mm_ecmwf_max",
    "ncep_nbm_conus":        "mm_nbm_max",
    "icon_seamless":         "mm_icon_max",
    "gem_seamless":          "mm_gem_max",
    "gem_hrdps_continental": "mm_gem_hrdps_max",
}

CHUNK_DAYS = 90


def _get(url: str, retries: int = 3) -> dict:
    last_exc = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_exc = e
            if i < retries - 1:
                time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"Open-Meteo request failed after {retries}: {last_exc}")


def fetch_model_daily(lat: float, lon: float, start: str, end: str, tz: str,
                      model: str) -> pd.DataFrame:
    """Fetch daily temperature_2m_max for one model over a date range,
    chunking into CHUNK_DAYS windows."""
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    frames = []
    cursor = d0
    while cursor <= d1:
        window_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), d1)
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": cursor.isoformat(), "end_date": window_end.isoformat(),
            "daily": "temperature_2m_max",
            "models": model,
            "temperature_unit": "fahrenheit",
            "timezone": tz,
        }
        qs = urllib.parse.urlencode(params)
        try:
            data = _get(f"{HISTORICAL_FORECAST_URL}?{qs}")
            if "error" in data:
                print(f"     ⚠️ {model} {cursor}→{window_end}: "
                      f"{data.get('reason', 'error')}")
            else:
                daily = data.get("daily", {}) or {}
                times = daily.get("time", []) or []
                vals  = daily.get("temperature_2m_max", []) or []
                if times:
                    frames.append(pd.DataFrame({"date": times, "value": vals}))
        except Exception as e:
            print(f"     ⚠️ {model} {cursor}→{window_end} failed: {e}")
        cursor = window_end + timedelta(days=1)
        time.sleep(1.0)
    if not frames:
        return pd.DataFrame(columns=["date", "value"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"])
    return out


def fetch_hrrr_d1_daily(lat: float, lon: float, start: str, end: str,
                        tz: str) -> pd.DataFrame:
    """Daily max of the day-before HRRR run (previous-runs API, hourly →
    grouped by local calendar day). Coverage starts ~2024; earlier dates
    return all-null and are simply skipped."""
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    daily_max: dict[str, float] = {}
    cursor = d0
    while cursor <= d1:
        window_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), d1)
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": cursor.isoformat(), "end_date": window_end.isoformat(),
            "hourly": "temperature_2m_previous_day1",
            "models": "ncep_hrrr_conus",
            "temperature_unit": "fahrenheit",
            "timezone": tz,
        }
        qs = urllib.parse.urlencode(params)
        try:
            data = _get(f"{PREVIOUS_RUNS_URL}?{qs}")
            hourly = data.get("hourly", {}) or {}
            times = hourly.get("time", []) or []
            vals = hourly.get("temperature_2m_previous_day1", []) or []
            for t, v in zip(times, vals):
                if v is None:
                    continue
                day = t[:10]
                if day not in daily_max or v > daily_max[day]:
                    daily_max[day] = v
        except Exception as e:
            print(f"     ⚠️ hrrr_d1 {cursor}→{window_end} failed: {e}")
        cursor = window_end + timedelta(days=1)
        time.sleep(1.0)
    if not daily_max:
        return pd.DataFrame(columns=["date", "value"])
    return pd.DataFrame(
        {"date": sorted(daily_max), "value": [daily_max[d] for d in sorted(daily_max)]}
    )


def recompute_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute mm_mean / mm_std / mm_spread from all populated mm_*_max cols
    (including mm_hrrr_max which is already there). Only writes where the
    existing value is NaN — never overwrites prediction-time captures."""
    model_cols = [c for c in [
        "mm_hrrr_max", "mm_gfs_max", "mm_ecmwf_max", "mm_nbm_max",
        "mm_icon_max", "mm_gem_max", "mm_gem_hrdps_max",
    ] if c in df.columns]
    if not model_cols:
        return df
    arr = df[model_cols].to_numpy(dtype=float)
    # Per-row count of populated models.
    n_per_row = (~np.isnan(arr)).sum(axis=1)
    enough = n_per_row >= 2
    means = np.nanmean(arr, axis=1)
    stds  = np.nanstd(arr, axis=1)
    mins_  = np.nanmin(arr, axis=1)
    maxs_  = np.nanmax(arr, axis=1)
    spreads = maxs_ - mins_

    for col, vals in [("mm_mean", means), ("mm_std", stds), ("mm_spread", spreads)]:
        if col not in df.columns:
            df[col] = np.nan
        # Only fill where existing is NaN AND we have ≥2 models on that row.
        mask = df[col].isna() & enough
        df.loc[mask, col] = vals[mask]

    # mm_ecmwf_gfs_diff
    if "mm_ecmwf_max" in df.columns and "mm_gfs_max" in df.columns:
        if "mm_ecmwf_gfs_diff" not in df.columns:
            df["mm_ecmwf_gfs_diff"] = np.nan
        diff = df["mm_ecmwf_max"] - df["mm_gfs_max"]
        mask = df["mm_ecmwf_gfs_diff"].isna() & diff.notna()
        df.loc[mask, "mm_ecmwf_gfs_diff"] = diff[mask]

    return df


def run(city: str, dry_run: bool = False, start_override: str | None = None,
        include_hrrr: bool = False) -> int:
    cfg = CITIES[city]
    csv_path = Path(cfg["csv"])
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return 1

    models = dict(MODELS)
    if include_hrrr:
        # Fill-only-NaN semantics below make this safe: live prediction-time
        # HRRR captures are never overwritten, only historical gaps filled.
        models = {**HRRR_MODEL, **models}
        print("⚓ --include-hrrr: filling mm_hrrr_max gaps from archived HRRR runs")

    df = pd.read_csv(csv_path)
    df["target_date"] = df["target_date"].astype(str).str[:10]
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Ensure all target columns exist.
    for col in models.values():
        if col not in df.columns:
            df[col] = np.nan
        n = df[col].notna().sum()
        print(f"  before: {col}: {n}/{len(df)}")

    # ── INCREMENTAL FETCH (2026-05-04) ───────────────────────────────────
    # Fetch only since latest already-populated date, with 7-day overlap.
    # Without this, every run re-pulled 4yr × 6 models × 90-day chunks =
    # ~96 API calls × ~8s each = ~13 min per city. Two cities = 26 min
    # just on this backfill alone.
    end = df["target_date"].max()
    if start_override:
        start = start_override
    else:
        populated_dates = []
        for col in models.values():
            if col in df.columns and df[col].notna().any():
                d = df.loc[df[col].notna(), "target_date"].max()
                populated_dates.append(d)
        if populated_dates:
            existing_max = min(populated_dates)
            existing_max_dt = datetime.strptime(existing_max, "%Y-%m-%d").date()
            incremental_start_dt = existing_max_dt - timedelta(days=7)
            start = max("2022-03-23", incremental_start_dt.isoformat())
        else:
            start = df["target_date"].min()
    if start < "2022-03-23":  # historical-forecast-api coverage limit
        start = "2022-03-23"
    print(f"\nIncremental fetch {cfg['lat']},{cfg['lon']} from {start} → {end} "
          f"(was {df['target_date'].min()} → {end} pre-incremental)")
    print(f"Models: {list(models.keys())}")

    fill_summary = {}
    for model_id, col_name in models.items():
        existing = df[col_name].notna().sum()
        if existing == len(df):
            print(f"\n[{col_name}] already 100% populated — skipping.")
            continue
        print(f"\n[{col_name}] fetching {model_id}…")
        daily = fetch_model_daily(cfg["lat"], cfg["lon"], start, end, cfg["tz"], model_id)
        if daily.empty:
            print(f"  ⚠️ no data for {model_id} — skipping.")
            continue
        print(f"  Retrieved {len(daily)} daily values for {model_id}")
        merged = df.merge(daily, left_on="target_date", right_on="date",
                          how="left", suffixes=("", "_fetch"))
        before = df[col_name].notna().sum()
        mask = df[col_name].isna() & merged["value"].notna()
        df.loc[mask, col_name] = merged.loc[mask, "value"]
        after = df[col_name].notna().sum()
        fill_summary[col_name] = after - before
        print(f"  filled {after - before} ({after}/{len(df)} total)")

    if include_hrrr:
        col = "mm_hrrr_max_d1"
        if col not in df.columns:
            df[col] = np.nan
        if df[col].notna().sum() < len(df):
            # Incremental like the models above: resume from last populated
            # d1 date (with 7-day overlap), full range on first run.
            if df[col].notna().any():
                d1_start_dt = datetime.strptime(
                    df.loc[df[col].notna(), "target_date"].max(), "%Y-%m-%d"
                ).date() - timedelta(days=7)
                d1_start = max("2022-03-23", d1_start_dt.isoformat())
            else:
                d1_start = max("2022-03-23", df["target_date"].min())
            print(f"\n[{col}] fetching day-before-lead HRRR (previous-runs API) "
                  f"{d1_start} → {end}…")
            daily = fetch_hrrr_d1_daily(cfg["lat"], cfg["lon"], d1_start, end, cfg["tz"])
            if daily.empty:
                print(f"  ⚠️ no d1 data — skipping.")
            else:
                merged = df.merge(daily, left_on="target_date", right_on="date",
                                  how="left", suffixes=("", "_fetch"))
                before = df[col].notna().sum()
                mask = df[col].isna() & merged["value"].notna()
                df.loc[mask, col] = merged.loc[mask, "value"]
                print(f"  filled {df[col].notna().sum() - before} "
                      f"({df[col].notna().sum()}/{len(df)} total)")

    print("\n📊 Recomputing consensus features (mm_mean / mm_std / mm_spread / mm_ecmwf_gfs_diff)…")
    df = recompute_consensus(df)

    print("\n📊 Final fill results:")
    for col in list(models.values()) + ["mm_mean", "mm_std", "mm_spread", "mm_ecmwf_gfs_diff", "mm_hrrr_max"]:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"  {col}: {n}/{len(df)}")

    if dry_run:
        print("\n🏜️  DRY RUN — no CSV write.")
        return 0

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    backup = csv_path.with_suffix(f".csv.bak-{ts}")
    csv_path.rename(backup)
    print(f"\n🛟 Backup: {backup}")
    df.to_csv(csv_path, index=False)
    print(f"✅ Wrote {csv_path} ({len(df)} rows, {len(df.columns)} cols)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="nyc", choices=list(CITIES.keys()))
    ap.add_argument("--start", default=None,
                    help="Override start date (YYYY-MM-DD); useful to refresh recent rows.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-hrrr", action="store_true",
                    help="Also fill mm_hrrr_max where NaN from archived HRRR runs "
                         "(never overwrites live prediction-time captures).")
    args = ap.parse_args()
    sys.exit(run(args.city, dry_run=args.dry_run, start_override=args.start,
                 include_hrrr=args.include_hrrr))
