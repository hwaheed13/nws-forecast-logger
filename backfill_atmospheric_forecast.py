# backfill_atmospheric_forecast.py — Build a forecast-sourced multiyear CSV
# by pulling Open-Meteo's historical-forecast-api instead of the ERA5 archive.
#
# Why this exists:
#   The existing multiyear_atmospheric.csv trains v3 (atm_predictor) on ERA5
#   analysis data. Live inference uses forecast-API atmospheric features. The
#   distribution shift between training (analysis) and inference (forecast)
#   is a likely source of v3 error. This script produces a parallel CSV whose
#   training distribution matches live-inference distribution.
#
#   Output is consumed by `train_models.py --v3-forecast-shadow`, which saves
#   a SECOND atm_predictor (atm_predictor_forecast.pkl). Production continues
#   using atm_predictor.pkl until live shadow logs prove the new one wins.
#
# Usage:
#   python backfill_atmospheric_forecast.py --city nyc
#   python backfill_atmospheric_forecast.py --city lax
#   python backfill_atmospheric_forecast.py --all

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from city_config import get_city_config, CITIES
from open_meteo_client import (
    HOURLY_VARS,
    HOURLY_PRESSURE_LEVEL_VARS,
    DAILY_VARS,
    _get_json,
    extract_daily_atmospheric,
)

HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
# historical-forecast-api coverage starts here; older dates silently return empty
HF_MIN_DATE = "2022-03-23"


def fetch_historical_forecast_hourly(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """
    Drop-in replacement for fetch_historical_hourly() that hits
    historical-forecast-api (GFS/ECMWF forecast archives) instead of
    archive-api (ERA5). Returns the same DataFrame shape so downstream
    extract_daily_atmospheric() works unchanged.
    """
    all_hourly_vars = HOURLY_VARS + HOURLY_PRESSURE_LEVEL_VARS
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(all_hourly_vars),
        "daily": ",".join(DAILY_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": timezone,
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{HISTORICAL_FORECAST_URL}?{qs}"
    data = _get_json(url)

    if "error" in data:
        raise RuntimeError(
            f"Open-Meteo historical-forecast error: {data.get('reason', data['error'])}"
        )

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({"time": pd.to_datetime(times)})
    for var in all_hourly_vars:
        df[var] = hourly.get(var, [None] * len(times))

    daily = data.get("daily", {})
    daily_times = daily.get("time", [])
    if daily_times:
        daily_df = pd.DataFrame({"date": pd.to_datetime(daily_times).date})
        for var in DAILY_VARS:
            daily_df[var] = daily.get(var, [None] * len(daily_times))
        df["_date"] = df["time"].dt.date
        df = df.merge(daily_df, left_on="_date", right_on="date", how="left")
        df.drop(columns=["_date", "date"], inplace=True, errors="ignore")

    return df


def get_target_dates(city_key: str) -> list[str]:
    """
    Use the same dates as multiyear_atmospheric.csv for this city, clipped to
    the historical-forecast-api coverage window. Keeps the two CSVs aligned
    row-for-row where coverage overlaps.
    """
    prefix = get_city_config(city_key).get("model_prefix", "")
    src = f"{prefix}multiyear_atmospheric.csv"
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"{src} not found — run backfill_multiyear.py first so dates align."
        )
    df = pd.read_csv(src, usecols=["target_date"])
    df["target_date"] = df["target_date"].astype(str).str[:10]
    dates = sorted(d for d in df["target_date"].unique() if d >= HF_MIN_DATE)
    return dates


def backfill_city(city_key: str, force: bool = False) -> str:
    cfg = get_city_config(city_key)
    lat = cfg["open_meteo_lat"]
    lon = cfg["open_meteo_lon"]
    tz = cfg["timezone"]
    prefix = cfg.get("model_prefix", "")
    output_csv = f"{prefix}multiyear_atmospheric_forecast.csv"

    print(f"\n{'='*60}")
    print(f"Backfilling FORECAST-sourced atmospheric data for {cfg['label']}")
    print(f"{'='*60}")

    dates = get_target_dates(city_key)
    print(f"Found {len(dates)} dates (historical-forecast-api window: "
          f"{dates[0]} → {dates[-1]})")

    existing_dates: set[str] = set()
    if os.path.exists(output_csv) and not force:
        existing_df = pd.read_csv(output_csv)
        existing_dates = set(existing_df["target_date"].astype(str).str[:10].tolist())
        print(f"Already have {len(existing_dates)} dates in {output_csv}")
        dates = [d for d in dates if d not in existing_dates]
        if not dates:
            print("All dates already backfilled. Use --force to re-fetch.")
            return output_csv
        print(f"Need to fetch {len(dates)} new dates")

    CHUNK_DAYS = 90
    all_features: list[dict] = []

    dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    min_date = min(dates_dt)
    max_date = max(dates_dt)

    current_start = min_date
    chunk_num = 0

    while current_start <= max_date:
        chunk_end = min(current_start + timedelta(days=CHUNK_DAYS - 1), max_date)
        chunk_num += 1

        start_str = current_start.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")

        chunk_dates = [d for d in dates if start_str <= d <= end_str]
        if not chunk_dates:
            current_start = chunk_end + timedelta(days=1)
            continue

        print(f"\nChunk {chunk_num}: {start_str} → {end_str} "
              f"({len(chunk_dates)} dates)")

        try:
            hourly_df = fetch_historical_forecast_hourly(
                lat, lon, start_str, end_str, tz
            )
            print(f"  Got {len(hourly_df)} hourly rows")

            for target_date in chunk_dates:
                features = extract_daily_atmospheric(hourly_df, target_date)
                if features:
                    features["target_date"] = target_date
                    features["city"] = city_key
                    all_features.append(features)
                else:
                    print(f"  ⚠️ No data for {target_date}")
        except Exception as e:
            print(f"  ❌ Chunk failed: {e}")
            for target_date in chunk_dates:
                try:
                    single_df = fetch_historical_forecast_hourly(
                        lat, lon, target_date, target_date, tz
                    )
                    features = extract_daily_atmospheric(single_df, target_date)
                    if features:
                        features["target_date"] = target_date
                        features["city"] = city_key
                        all_features.append(features)
                    time.sleep(0.5)
                except Exception as e2:
                    print(f"  ❌ {target_date} failed: {e2}")

        time.sleep(1.0)
        current_start = chunk_end + timedelta(days=1)

    if not all_features:
        print("No features extracted!")
        return output_csv

    new_df = pd.DataFrame(all_features)

    if existing_dates and os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["target_date"], keep="last")
        combined = combined.sort_values("target_date").reset_index(drop=True)
    else:
        combined = new_df.sort_values("target_date").reset_index(drop=True)

    combined.to_csv(output_csv, index=False)

    print(f"\n✅ Saved {len(combined)} dates to {output_csv}")
    print(f"   Columns: {len(combined.columns)}")

    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n   Feature summary (sample of columns):")
        for col in list(numeric_cols)[:5]:
            vals = combined[col].dropna()
            if len(vals) > 0:
                print(
                    f"     {col}: mean={vals.mean():.1f}, std={vals.std():.1f}, "
                    f"min={vals.min():.1f}, max={vals.max():.1f}"
                )

    return output_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill forecast-sourced atmospheric data from Open-Meteo "
                    "historical-forecast-api."
    )
    parser.add_argument("--city", default="nyc", help="City key (nyc, lax)")
    parser.add_argument("--all", action="store_true", help="Backfill all cities")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if data exists")
    args = parser.parse_args()

    if args.all:
        for city_key in CITIES:
            backfill_city(city_key, force=args.force)
    else:
        backfill_city(args.city, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
