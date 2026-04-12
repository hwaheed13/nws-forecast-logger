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


def main():
    parser = argparse.ArgumentParser(
        description="Backfill multi-year atmospheric data from Open-Meteo Archive"
    )
    parser.add_argument("--city", default="nyc", help="City key (nyc, lax)")
    parser.add_argument("--all", action="store_true", help="Backfill all cities")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if data exists")
    args = parser.parse_args()

    if args.all:
        for city_key in CITIES:
            backfill_multiyear(city_key, args.start, args.end, args.force)
    else:
        backfill_multiyear(args.city, args.start, args.end, args.force)


if __name__ == "__main__":
    main()
