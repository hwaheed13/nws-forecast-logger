# backfill_multiyear.py — Fetch 3+ years of historical atmospheric features
# from the Open-Meteo Archive API to expand ML training data from ~231 days
# to 1500+ days.
#
# For these historical dates we DON'T have NWS/AccuWeather forecast data,
# but HistGradientBoosting handles NaN natively — the model learns
# atmospheric → bucket relationships from this expanded data.
#
# Usage:
#   python backfill_multiyear.py --city nyc
#   python backfill_multiyear.py --city nyc --start 2020-01-01 --end 2025-07-01
#   python backfill_multiyear.py --all

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from city_config import get_city_config, CITIES
from open_meteo_client import (
    fetch_historical_hourly,
    extract_daily_atmospheric,
)


# Default date range: 3+ years before existing data starts (July 2025)
DEFAULT_START = "2022-01-01"
DEFAULT_END = "2025-07-01"  # Existing atmospheric_data.csv covers July 2025+


def backfill_multiyear(
    city_key: str,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    force: bool = False,
) -> str:
    """
    Backfill multi-year atmospheric data + actual daily highs for a city.

    For each day in the date range, extracts:
      - actual_high: from Open-Meteo's temperature_2m_max
      - 15 atmospheric features (wind, humidity, pressure, cloud, etc.)

    Returns path to the output CSV file.
    """
    cfg = get_city_config(city_key)
    lat = cfg["open_meteo_lat"]
    lon = cfg["open_meteo_lon"]
    tz = cfg["timezone"]
    prefix = cfg.get("model_prefix", "")
    output_csv = f"{prefix}multiyear_atmospheric.csv"

    print(f"\n{'='*60}")
    print(f"Multi-year backfill for {cfg['label']}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Coordinates: {lat}, {lon}")
    print(f"  Output: {output_csv}")
    print(f"{'='*60}")

    # Check existing data
    existing_dates = set()
    if os.path.exists(output_csv) and not force:
        existing_df = pd.read_csv(output_csv)
        existing_dates = set(existing_df["target_date"].astype(str).tolist())
        print(f"Already have {len(existing_dates)} dates in {output_csv}")

    # Generate all dates in the range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    all_dates = []
    current = start_dt
    while current <= end_dt:
        d = current.strftime("%Y-%m-%d")
        if d not in existing_dates:
            all_dates.append(d)
        current += timedelta(days=1)

    if not all_dates:
        print("All dates already backfilled. Use --force to re-fetch.")
        return output_csv

    print(f"Need to fetch {len(all_dates)} new dates")

    # Fetch in 90-day chunks (Open-Meteo handles ranges efficiently)
    CHUNK_DAYS = 90
    all_features = []
    chunk_num = 0
    i = 0

    while i < len(all_dates):
        chunk_start = all_dates[i]
        chunk_end_idx = min(i + CHUNK_DAYS - 1, len(all_dates) - 1)
        chunk_end = all_dates[chunk_end_idx]
        chunk_dates = all_dates[i : chunk_end_idx + 1]
        chunk_num += 1

        print(f"\nChunk {chunk_num}: {chunk_start} to {chunk_end} ({len(chunk_dates)} dates)")

        try:
            hourly_df = fetch_historical_hourly(lat, lon, chunk_start, chunk_end, tz)
            print(f"  Got {len(hourly_df)} hourly rows")

            # Extract the daily max from the hourly data for actual_high
            # Also available in the daily data (temperature_2m_max)
            for target_date in chunk_dates:
                features = extract_daily_atmospheric(hourly_df, target_date)
                if not features:
                    continue

                # Get actual daily high from the data
                target = pd.Timestamp(target_date)
                day_mask = hourly_df["time"].dt.date == target.date()
                day_data = hourly_df[day_mask]

                actual_high = None
                # Prefer the daily max column if available
                if "temperature_2m_max" in day_data.columns:
                    max_vals = day_data["temperature_2m_max"].dropna()
                    if len(max_vals) > 0:
                        actual_high = float(max_vals.iloc[0])

                # Fallback: compute max from hourly temperature_2m
                if actual_high is None:
                    temps = day_data["temperature_2m"].dropna()
                    if len(temps) > 0:
                        actual_high = float(temps.max())

                if actual_high is None:
                    continue

                features["target_date"] = target_date
                features["actual_high"] = actual_high
                features["city"] = city_key
                features["data_source"] = "open_meteo_archive"
                all_features.append(features)

            print(f"  Extracted {sum(1 for f in all_features if f['target_date'] in chunk_dates)} days")

        except Exception as e:
            print(f"  ERROR: Chunk failed: {e}")
            # Fall back to individual date fetching
            for target_date in chunk_dates:
                try:
                    single_df = fetch_historical_hourly(lat, lon, target_date, target_date, tz)
                    features = extract_daily_atmospheric(single_df, target_date)
                    if features:
                        # Get actual high
                        day_mask = single_df["time"].dt.date == pd.Timestamp(target_date).date()
                        day_data = single_df[day_mask]
                        actual_high = None
                        if "temperature_2m_max" in day_data.columns:
                            max_vals = day_data["temperature_2m_max"].dropna()
                            if len(max_vals) > 0:
                                actual_high = float(max_vals.iloc[0])
                        if actual_high is None:
                            temps = day_data["temperature_2m"].dropna()
                            if len(temps) > 0:
                                actual_high = float(temps.max())
                        if actual_high is not None:
                            features["target_date"] = target_date
                            features["actual_high"] = actual_high
                            features["city"] = city_key
                            features["data_source"] = "open_meteo_archive"
                            all_features.append(features)
                    time.sleep(0.5)
                except Exception as e2:
                    print(f"    {target_date} failed: {e2}")

        # Be nice to the API
        time.sleep(1.0)
        i = chunk_end_idx + 1

    if not all_features:
        print("No features extracted!")
        return output_csv

    # Build DataFrame
    new_df = pd.DataFrame(all_features)

    # Merge with existing data if any
    if existing_dates and os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["target_date"], keep="last")
        combined = combined.sort_values("target_date").reset_index(drop=True)
    else:
        combined = new_df.sort_values("target_date").reset_index(drop=True)

    combined.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"DONE: {len(combined)} total dates saved to {output_csv}")
    print(f"  Date range: {combined['target_date'].min()} to {combined['target_date'].max()}")
    print(f"  Columns: {list(combined.columns)}")

    # Summary stats
    if "actual_high" in combined.columns:
        ah = combined["actual_high"].dropna()
        print(f"\n  actual_high: mean={ah.mean():.1f}, std={ah.std():.1f}, "
              f"min={ah.min():.0f}, max={ah.max():.0f}")

    # Show seasonal distribution
    combined["_month"] = pd.to_datetime(combined["target_date"]).dt.month
    monthly = combined.groupby("_month").size()
    print(f"\n  Monthly distribution:")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m, count in monthly.items():
        print(f"    {month_names[int(m)-1]}: {count} days")

    return output_csv


def patch_live_columns(city_key: str, past_days: int = 90) -> None:
    """
    Patch recent rows in the multiyear CSV with live-API features that the archive
    API doesn't provide: HRRR, 925mb temp, solar irradiance, 850mb temp.

    Uses Open-Meteo forecast API with past_days=N (max 92) to get analysis data
    for the most recent ~90 days and fills in NaN cells for those columns.
    """
    from open_meteo_client import fetch_multimodel_forecast, extract_multimodel_features
    import urllib.request, json as _json
    import numpy as np

    cfg = get_city_config(city_key)
    lat = cfg["open_meteo_lat"]
    lon = cfg["open_meteo_lon"]
    tz = cfg["timezone"]
    prefix = cfg.get("model_prefix", "")
    output_csv = f"{prefix}multiyear_atmospheric.csv"

    if not os.path.exists(output_csv):
        print(f"No CSV found: {output_csv}")
        return

    df = pd.read_csv(output_csv)
    print(f"\n{'='*60}")
    print(f"Patching live-API columns for {cfg['label']} ({output_csv})")
    print(f"  Rows before patch: {len(df)}")

    # Determine which dates need patching
    cutoff = (datetime.utcnow() - timedelta(days=past_days)).strftime("%Y-%m-%d")
    recent_mask = df["target_date"] >= cutoff
    print(f"  Patching {recent_mask.sum()} rows from {cutoff} onwards")

    if recent_mask.sum() == 0:
        print("  Nothing to patch.")
        return

    # Ensure new columns exist
    new_cols = ["atm_925mb_temp_max", "atm_925mb_temp_mean",
                "atm_solar_radiation_peak", "atm_solar_radiation_mean",
                "mm_hrrr_max", "mm_hrrr_ecmwf_diff", "mm_hrrr_gfs_diff"]
    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Fetch HRRR + ECMWF + GFS via forecast API (past_days mode)
    print("  Fetching HRRR/multi-model data from forecast API...")
    try:
        tz_enc = tz.replace("/", "%2F")
        url = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               f"&daily=temperature_2m_max"
               f"&models=ncep_hrrr_conus,ecmwf_ifs025,gfs_seamless"
               f"&temperature_unit=fahrenheit"
               f"&timezone={tz_enc}"
               f"&past_days={past_days}&forecast_days=1")
        req = urllib.request.Request(url, headers={"User-Agent": "nws-forecast-logger/1.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            mm_data_raw = _json.loads(r.read())
        daily = mm_data_raw.get("daily", {})
        dates = daily.get("time", [])
        hrrr_vals = daily.get("temperature_2m_max_ncep_hrrr_conus", [])
        ecmwf_vals = daily.get("temperature_2m_max_ecmwf_ifs025", [])
        gfs_vals = daily.get("temperature_2m_max_gfs_seamless", [])
        hrrr_by_date = {d: v for d, v in zip(dates, hrrr_vals) if v is not None}
        ecmwf_by_date = {d: v for d, v in zip(dates, ecmwf_vals) if v is not None}
        gfs_by_date = {d: v for d, v in zip(dates, gfs_vals) if v is not None}
        print(f"    Got HRRR for {len(hrrr_by_date)} dates, ECMWF for {len(ecmwf_by_date)}")
    except Exception as e:
        print(f"  ⚠️ Multi-model fetch failed: {e}")
        hrrr_by_date, ecmwf_by_date, gfs_by_date = {}, {}, {}

    # Fetch 925hPa + solar via forecast API hourly (past_days)
    print("  Fetching 925hPa + solar data from forecast API...")
    try:
        url2 = (f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&hourly=temperature_925hPa,shortwave_radiation"
                f"&temperature_unit=fahrenheit"
                f"&timezone={tz_enc}"
                f"&past_days={past_days}&forecast_days=1")
        req2 = urllib.request.Request(url2, headers={"User-Agent": "nws-forecast-logger/1.0", "Accept": "application/json"})
        with urllib.request.urlopen(req2, timeout=20) as r2:
            hourly_data = _json.loads(r2.read())
        hourly = hourly_data.get("hourly", {})
        htimes = pd.to_datetime(hourly.get("time", []))
        t925 = hourly.get("temperature_925hPa", [None]*len(htimes))
        solar = hourly.get("shortwave_radiation", [None]*len(htimes))
        hourly_df = pd.DataFrame({"time": htimes, "temperature_925hPa": t925, "shortwave_radiation": solar})
        print(f"    Got {len(hourly_df)} hourly rows")
    except Exception as e:
        print(f"  ⚠️ Hourly 925/solar fetch failed: {e}")
        hourly_df = pd.DataFrame()

    # Patch each recent row
    patched = 0
    for idx in df[recent_mask].index:
        td = df.loc[idx, "target_date"]

        # HRRR / multi-model
        hrrr = hrrr_by_date.get(td)
        ecmwf = ecmwf_by_date.get(td)
        gfs = gfs_by_date.get(td)
        if hrrr is not None:
            df.loc[idx, "mm_hrrr_max"] = float(hrrr)
        if hrrr is not None and ecmwf is not None:
            df.loc[idx, "mm_hrrr_ecmwf_diff"] = float(hrrr - ecmwf)
        if hrrr is not None and gfs is not None:
            df.loc[idx, "mm_hrrr_gfs_diff"] = float(hrrr - gfs)

        # 925hPa + solar
        if not hourly_df.empty:
            day_mask = hourly_df["time"].dt.strftime("%Y-%m-%d") == td
            day_h = hourly_df[day_mask]
            # Daytime hours 10am-6pm for 925mb
            daytime = day_h[(day_h["time"].dt.hour >= 10) & (day_h["time"].dt.hour <= 18)]
            t925_dt = daytime["temperature_925hPa"].dropna()
            if len(t925_dt) > 0:
                df.loc[idx, "atm_925mb_temp_max"] = float(t925_dt.max())
                df.loc[idx, "atm_925mb_temp_mean"] = float(t925_dt.mean())
            # Midday solar 10am-2pm
            midday = day_h[(day_h["time"].dt.hour >= 10) & (day_h["time"].dt.hour <= 14)]
            sol_dt = midday["shortwave_radiation"].dropna()
            if len(sol_dt) > 0:
                df.loc[idx, "atm_solar_radiation_peak"] = float(sol_dt.max())
                df.loc[idx, "atm_solar_radiation_mean"] = float(sol_dt.mean())

        patched += 1

    df.to_csv(output_csv, index=False)
    print(f"  Patched {patched} rows. Saved to {output_csv}")

    # Show sample
    sample = df[recent_mask][["target_date", "mm_hrrr_max", "mm_hrrr_ecmwf_diff",
                               "atm_925mb_temp_max", "atm_solar_radiation_peak"]].tail(7)
    print(f"\n  Recent patched values:\n{sample.to_string(index=False)}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill multi-year atmospheric data from Open-Meteo Archive"
    )
    parser.add_argument("--city", default="nyc", help="City key (nyc, lax)")
    parser.add_argument("--all", action="store_true", help="Backfill all cities")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if data exists")
    parser.add_argument("--patch-live-cols", action="store_true",
                        help="Patch recent rows with HRRR/925mb/solar (forecast API past_days=90)")
    args = parser.parse_args()

    if getattr(args, "patch_live_cols", False):
        cities = list(CITIES.keys()) if args.all else [args.city]
        for city_key in cities:
            patch_live_columns(city_key)
        return

    if args.all:
        for city_key in CITIES:
            backfill_multiyear(city_key, args.start, args.end, args.force)
    else:
        backfill_multiyear(args.city, args.start, args.end, args.force)


if __name__ == "__main__":
    main()
