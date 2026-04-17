#!/usr/bin/env python3
"""
backfill_new_columns.py — Backfill HRRR/925mb/850mb/solar/BL-height into multiyear_atmospheric.csv.

Uses the Open-Meteo Historical Forecast API (NOT archive) which definitively has:
  - ECMWF 0.25°: 850mb temp, 925mb temp, shortwave radiation, BL height (from 2017)
  - HRRR: daily max temperature (from 2014)
  - GFS: daily max temperature (from 2015)

Our training data starts 2022-01-01, so all models cover the full range.

Adds/patches these columns in multiyear_atmospheric.csv:
  atmospheric (used in ATMOSPHERIC_COLS → go into model):
    atm_850mb_temp_max, atm_850mb_temp_mean
    atm_925mb_temp_max, atm_925mb_temp_mean
    atm_solar_radiation_peak, atm_solar_radiation_mean
    atm_bl_height_max, atm_bl_height_mean

  multimodel (patched into CSV, training code updated to use them):
    mm_hrrr_max, mm_hrrr_ecmwf_diff, mm_hrrr_gfs_diff
    mm_spread, mm_mean, mm_std, mm_ecmwf_gfs_diff

Usage:
    python backfill_new_columns.py            # NYC only, patch missing rows
    python backfill_new_columns.py --force    # re-fetch all rows
    python backfill_new_columns.py --dry-run  # show what would be fetched
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

CITY_CONFIGS = {
    "nyc": {
        "lat": 40.7128,
        "lon": -74.0060,
        "timezone": "America/New_York",
        "csv": "multiyear_atmospheric.csv",
    },
    "lax": {
        "lat": 34.0522,
        "lon": -118.2437,
        "timezone": "America/Los_Angeles",
        "csv": "lax_multiyear_atmospheric.csv",
    },
}

# New columns to backfill (will be NaN if not successfully fetched)
NEW_ATM_COLS = [
    "atm_850mb_temp_max", "atm_850mb_temp_mean",
    "atm_925mb_temp_max", "atm_925mb_temp_mean",
    "atm_solar_radiation_peak", "atm_solar_radiation_mean",
    "atm_bl_height_max", "atm_bl_height_mean",
]
NEW_MM_COLS = [
    "mm_hrrr_max",
    "mm_hrrr_ecmwf_diff",
    "mm_hrrr_gfs_diff",
    "mm_spread",
    "mm_mean",
    "mm_std",
    "mm_ecmwf_gfs_diff",
]
ALL_NEW_COLS = NEW_ATM_COLS + NEW_MM_COLS


def _get_json(url: str, retries: int = 3, delay: float = 2.0) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if attempt < retries - 1:
                print(f"  ⚠️  Attempt {attempt+1} failed: {e}. Retrying in {delay*(attempt+1):.0f}s...")
                time.sleep(delay * (attempt + 1))
            else:
                raise RuntimeError(f"Request failed after {retries} attempts: {e}") from e


def fetch_ecmwf_hourly(lat: float, lon: float, start: str, end: str, tz: str) -> pd.DataFrame:
    """
    Fetch ECMWF 0.25° hourly data from Open-Meteo Historical Forecast API.
    Returns DataFrame with columns: time, temperature_850hPa, temperature_925hPa,
                                    shortwave_radiation, boundary_layer_height.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join([
            "temperature_2m",
            "shortwave_radiation",
            "boundary_layer_height",
            "temperature_850hPa",
            "temperature_925hPa",
        ]),
        "models": "ecmwf_ifs025",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": tz,
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{HISTORICAL_FORECAST_URL}?{qs}"

    data = _get_json(url)
    if "error" in data:
        raise RuntimeError(f"ECMWF API error: {data.get('reason', data['error'])}")

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({"time": pd.to_datetime(times)})
    for var in ["temperature_2m", "shortwave_radiation", "boundary_layer_height",
                "temperature_850hPa", "temperature_925hPa"]:
        raw = hourly.get(var, [None] * len(times))
        df[var] = pd.to_numeric(raw, errors="coerce")

    return df


def fetch_model_daily_max(
    lat: float, lon: float, start: str, end: str,
    tz: str, model: str
) -> dict[str, float]:
    """
    Fetch daily max temperature for a single model from the Historical Forecast API.
    Returns {date_str: max_temp_F}.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max",
        "models": model,
        "temperature_unit": "fahrenheit",
        "timezone": tz,
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{HISTORICAL_FORECAST_URL}?{qs}"

    try:
        data = _get_json(url)
        if "error" in data:
            print(f"    ⚠️  {model}: {data.get('reason', 'error')}")
            return {}

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        maxes = daily.get("temperature_2m_max", [])
        return {d: float(t) for d, t in zip(dates, maxes) if t is not None}
    except Exception as e:
        print(f"    ⚠️  {model} failed: {e}")
        return {}


def extract_atm_features_for_date(hourly_df: pd.DataFrame, target_date: str) -> dict:
    """
    Extract 850mb, 925mb, solar radiation, and BL height features for one date.
    Uses daytime window (10am-4pm local) for peak/mean calculations.
    """
    target = pd.Timestamp(target_date)
    day_mask = hourly_df["time"].dt.date == target.date()
    day = hourly_df[day_mask]

    if day.empty:
        return {col: np.nan for col in NEW_ATM_COLS}

    # Daytime window for pressure-level and solar features (10am-4pm)
    daytime_mask = (day["time"].dt.hour >= 10) & (day["time"].dt.hour <= 16)
    daytime = day[daytime_mask]

    features = {}

    # 850mb temperature (warm air advection aloft)
    if "temperature_850hPa" in day.columns:
        vals = daytime["temperature_850hPa"].dropna() if not daytime.empty else pd.Series(dtype=float)
        features["atm_850mb_temp_max"] = float(vals.max()) if len(vals) > 0 else np.nan
        features["atm_850mb_temp_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
    else:
        features["atm_850mb_temp_max"] = np.nan
        features["atm_850mb_temp_mean"] = np.nan

    # 925mb temperature (near-surface boundary layer)
    if "temperature_925hPa" in day.columns:
        vals = daytime["temperature_925hPa"].dropna() if not daytime.empty else pd.Series(dtype=float)
        features["atm_925mb_temp_max"] = float(vals.max()) if len(vals) > 0 else np.nan
        features["atm_925mb_temp_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
    else:
        features["atm_925mb_temp_max"] = np.nan
        features["atm_925mb_temp_mean"] = np.nan

    # Solar radiation — midday (10am-4pm)
    if "shortwave_radiation" in day.columns:
        vals = daytime["shortwave_radiation"].dropna() if not daytime.empty else pd.Series(dtype=float)
        features["atm_solar_radiation_peak"] = float(vals.max()) if len(vals) > 0 else np.nan
        features["atm_solar_radiation_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
    else:
        features["atm_solar_radiation_peak"] = np.nan
        features["atm_solar_radiation_mean"] = np.nan

    # Boundary layer height — peak heating window (10am-4pm)
    if "boundary_layer_height" in day.columns:
        vals = daytime["boundary_layer_height"].dropna() if not daytime.empty else pd.Series(dtype=float)
        features["atm_bl_height_max"] = float(vals.max()) if len(vals) > 0 else np.nan
        features["atm_bl_height_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
    else:
        features["atm_bl_height_max"] = np.nan
        features["atm_bl_height_mean"] = np.nan

    return features


def backfill_city(city_key: str, force: bool = False, dry_run: bool = False) -> None:
    cfg = CITY_CONFIGS.get(city_key)
    if cfg is None:
        print(f"Unknown city: {city_key}. Available: {list(CITY_CONFIGS.keys())}")
        return

    csv_path = cfg["csv"]
    lat, lon, tz = cfg["lat"], cfg["lon"], cfg["timezone"]

    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return

    print(f"\n{'='*60}")
    print(f"Backfilling new columns for {city_key.upper()} — {csv_path}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows. Date range: {df['target_date'].min()} to {df['target_date'].max()}")

    # Ensure new columns exist
    for col in ALL_NEW_COLS:
        if col not in df.columns:
            df[col] = np.nan
            print(f"  + Added column: {col}")

    # Identify rows that need backfilling
    if force:
        needs_fill_mask = pd.Series([True] * len(df))
    else:
        # Fill rows where ANY of the new columns is NaN
        needs_fill_mask = df[ALL_NEW_COLS].isna().any(axis=1)

    dates_to_fill = sorted(df.loc[needs_fill_mask, "target_date"].astype(str).tolist())
    print(f"Rows needing backfill: {len(dates_to_fill)} / {len(df)}")

    if not dates_to_fill:
        print("✅ All rows already have new columns populated.")
        return

    if dry_run:
        print(f"\n[DRY RUN] Would fetch {len(dates_to_fill)} dates:")
        print(f"  First: {dates_to_fill[0]}, Last: {dates_to_fill[-1]}")
        return

    # ── Chunk into 90-day windows ─────────────────────────────────────────
    CHUNK = 90
    dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in dates_to_fill]
    min_date = min(dates_dt)
    max_date = max(dates_dt)

    # Collect all results keyed by date
    atm_results: dict[str, dict] = {}      # date → atm features
    hrrr_results: dict[str, float] = {}    # date → HRRR daily max
    ecmwf_results: dict[str, float] = {}   # date → ECMWF daily max
    gfs_results: dict[str, float] = {}     # date → GFS daily max

    current = min_date
    chunk_num = 0
    total_chunks = int(np.ceil((max_date - min_date).days / CHUNK)) + 1

    while current <= max_date:
        chunk_end = min(current + timedelta(days=CHUNK - 1), max_date)
        chunk_num += 1
        start_str = current.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")

        # Dates in this chunk that need filling
        chunk_dates = [d for d in dates_to_fill if start_str <= d <= end_str]
        if not chunk_dates:
            current = chunk_end + timedelta(days=1)
            continue

        print(f"\n[{chunk_num}/{total_chunks}] {start_str} → {end_str} ({len(chunk_dates)} dates)")

        # ── ECMWF hourly: 850mb, 925mb, solar, BL ────────────────────────
        print(f"  Fetching ECMWF hourly (850mb, 925mb, solar, BL)...")
        try:
            ecmwf_hourly = fetch_ecmwf_hourly(lat, lon, start_str, end_str, tz)
            non_null_check = ecmwf_hourly.get("temperature_925hPa", pd.Series()).notna().sum() \
                if hasattr(ecmwf_hourly, "get") else 0

            print(f"    Got {len(ecmwf_hourly)} hourly rows")

            for target_date in chunk_dates:
                features = extract_atm_features_for_date(ecmwf_hourly, target_date)
                atm_results[target_date] = features
                # Quick quality check
                if pd.notna(features.get("atm_925mb_temp_max")):
                    pass  # good
                else:
                    print(f"    ⚠️  {target_date}: 925mb NaN (ECMWF may not cover this date)")

        except Exception as e:
            print(f"  ❌ ECMWF hourly failed: {e}")
            for target_date in chunk_dates:
                atm_results[target_date] = {col: np.nan for col in NEW_ATM_COLS}

        time.sleep(1.0)

        # ── HRRR daily max ────────────────────────────────────────────────
        print(f"  Fetching HRRR daily max...")
        try:
            chunk_hrrr = fetch_model_daily_max(lat, lon, start_str, end_str, tz, "ncep_hrrr_conus")
            hrrr_results.update(chunk_hrrr)
            print(f"    Got {len(chunk_hrrr)} HRRR dates")
        except Exception as e:
            print(f"  ❌ HRRR failed: {e}")

        time.sleep(0.8)

        # ── ECMWF daily max ───────────────────────────────────────────────
        print(f"  Fetching ECMWF daily max...")
        try:
            chunk_ecmwf = fetch_model_daily_max(lat, lon, start_str, end_str, tz, "ecmwf_ifs025")
            ecmwf_results.update(chunk_ecmwf)
            print(f"    Got {len(chunk_ecmwf)} ECMWF dates")
        except Exception as e:
            print(f"  ❌ ECMWF daily failed: {e}")

        time.sleep(0.8)

        # ── GFS daily max ─────────────────────────────────────────────────
        print(f"  Fetching GFS daily max...")
        try:
            chunk_gfs = fetch_model_daily_max(lat, lon, start_str, end_str, tz, "gfs_seamless")
            gfs_results.update(chunk_gfs)
            print(f"    Got {len(chunk_gfs)} GFS dates")
        except Exception as e:
            print(f"  ❌ GFS failed: {e}")

        time.sleep(1.0)
        current = chunk_end + timedelta(days=1)

    # ── Apply results back into DataFrame ─────────────────────────────────
    print(f"\nApplying results to DataFrame...")

    updated_atm = 0
    updated_mm = 0

    for idx, row in df.iterrows():
        date_str = str(row["target_date"])
        if date_str not in dates_to_fill:
            continue

        # Apply atmospheric features (only overwrite NaN — don't clobber good data)
        if date_str in atm_results:
            features = atm_results[date_str]
            for col in NEW_ATM_COLS:
                current_val = df.at[idx, col]
                new_val = features.get(col, np.nan)
                if (pd.isna(current_val) or force) and pd.notna(new_val):
                    df.at[idx, col] = new_val
                    updated_atm += 1

        # Compute multimodel features from daily max results
        hrrr = hrrr_results.get(date_str)
        ecmwf = ecmwf_results.get(date_str)
        gfs = gfs_results.get(date_str)

        if hrrr is not None:
            df.at[idx, "mm_hrrr_max"] = float(hrrr)
        if hrrr is not None and ecmwf is not None:
            df.at[idx, "mm_hrrr_ecmwf_diff"] = float(hrrr - ecmwf)
        if hrrr is not None and gfs is not None:
            df.at[idx, "mm_hrrr_gfs_diff"] = float(hrrr - gfs)
        if ecmwf is not None and gfs is not None:
            df.at[idx, "mm_ecmwf_gfs_diff"] = float(ecmwf - gfs)

        # Multi-model spread/mean/std (using HRRR, ECMWF, GFS)
        all_models = [v for v in [hrrr, ecmwf, gfs] if v is not None]
        if len(all_models) >= 2:
            df.at[idx, "mm_spread"] = float(max(all_models) - min(all_models))
            df.at[idx, "mm_mean"] = float(np.mean(all_models))
            df.at[idx, "mm_std"] = float(np.std(all_models))
            updated_mm += 1

    print(f"  Updated {updated_atm} atmospheric values, {updated_mm} multimodel rows")

    # ── Save ──────────────────────────────────────────────────────────────
    df = df.sort_values("target_date").reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved {len(df)} rows to {csv_path}")

    # ── Summary stats ─────────────────────────────────────────────────────
    print("\nColumn coverage after backfill:")
    for col in ALL_NEW_COLS:
        if col in df.columns:
            n_filled = df[col].notna().sum()
            print(f"  {col}: {n_filled}/{len(df)} ({n_filled/len(df)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Backfill new atmospheric/multimodel columns")
    parser.add_argument("--city", default="nyc", choices=list(CITY_CONFIGS.keys()),
                        help="City to backfill (default: nyc)")
    parser.add_argument("--all", action="store_true", help="Backfill all cities")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch all dates even if already populated")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched without making API calls")
    args = parser.parse_args()

    if args.all:
        for city in CITY_CONFIGS:
            backfill_city(city, force=args.force, dry_run=args.dry_run)
    else:
        backfill_city(args.city, force=args.force, dry_run=args.dry_run)

    print("\nDone! Run train_models.py to retrain with the new features.")


if __name__ == "__main__":
    main()
