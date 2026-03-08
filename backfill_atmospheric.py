# backfill_atmospheric.py — One-time script to backfill atmospheric features
# for all historical training dates using the Open-Meteo archive API.
#
# Usage:
#   python backfill_atmospheric.py --city nyc
#   python backfill_atmospheric.py --city lax
#   python backfill_atmospheric.py --all

from __future__ import annotations

import argparse
import json
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


def get_dates_with_actuals(nws_csv: str) -> list[str]:
    """Extract sorted list of dates that have actual high temps recorded."""
    df = pd.read_csv(nws_csv)

    actual_rows = df[
        (df["forecast_or_actual"] == "actual")
        & df["actual_high"].notna()
        & (df["actual_high"] != "")
    ]

    dates = set()
    for _, row in actual_rows.iterrows():
        d = row.get("cli_date") or row.get("target_date")
        if d and str(d) not in ("", "<NA>", "nan", "None"):
            try:
                pd.to_datetime(str(d))
                dates.add(str(d).strip())
            except Exception:
                pass

    return sorted(dates)


def backfill_city(city_key: str, force: bool = False) -> str:
    """
    Backfill atmospheric features for all training dates of a city.
    Returns path to the output CSV file.
    """
    cfg = get_city_config(city_key)
    lat = cfg["open_meteo_lat"]
    lon = cfg["open_meteo_lon"]
    tz = cfg["timezone"]
    nws_csv = cfg["nws_csv"]
    prefix = cfg.get("model_prefix", "")
    output_csv = f"{prefix}atmospheric_data.csv"

    print(f"\n{'='*60}")
    print(f"Backfilling atmospheric data for {cfg['label']}")
    print(f"{'='*60}")

    # Get dates with actuals
    dates = get_dates_with_actuals(nws_csv)
    print(f"Found {len(dates)} dates with actual data: {dates[0]} to {dates[-1]}")

    # Check if output exists and load already-processed dates
    existing_dates = set()
    if os.path.exists(output_csv) and not force:
        existing_df = pd.read_csv(output_csv)
        existing_dates = set(existing_df["target_date"].astype(str).tolist())
        print(f"Already have {len(existing_dates)} dates in {output_csv}")
        dates = [d for d in dates if d not in existing_dates]
        if not dates:
            print("All dates already backfilled. Use --force to re-fetch.")
            return output_csv
        print(f"Need to fetch {len(dates)} new dates")

    # Batch-fetch: Open-Meteo archive API supports date ranges.
    # Chunk into 90-day windows to stay within API limits.
    CHUNK_DAYS = 90
    all_features = []

    # Sort dates and create chunks
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

        # Dates in this chunk that we need
        chunk_dates = [d for d in dates
                       if start_str <= d <= end_str]

        if not chunk_dates:
            current_start = chunk_end + timedelta(days=1)
            continue

        print(f"\nChunk {chunk_num}: {start_str} to {end_str} ({len(chunk_dates)} dates)")

        try:
            hourly_df = fetch_historical_hourly(lat, lon, start_str, end_str, tz)
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
            # Fall back to individual date fetching for this chunk
            for target_date in chunk_dates:
                try:
                    single_df = fetch_historical_hourly(lat, lon, target_date, target_date, tz)
                    features = extract_daily_atmospheric(single_df, target_date)
                    if features:
                        features["target_date"] = target_date
                        features["city"] = city_key
                        all_features.append(features)
                    time.sleep(0.5)
                except Exception as e2:
                    print(f"  ❌ {target_date} failed: {e2}")

        # Brief pause between chunks to be nice to the API
        time.sleep(1.0)
        current_start = chunk_end + timedelta(days=1)

    if not all_features:
        print("No features extracted!")
        return output_csv

    # Build DataFrame and save
    new_df = pd.DataFrame(all_features)

    # Merge with existing data if any
    if existing_dates and os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        # Dedupe by target_date (keep latest)
        combined = combined.drop_duplicates(subset=["target_date"], keep="last")
        combined = combined.sort_values("target_date").reset_index(drop=True)
    else:
        combined = new_df.sort_values("target_date").reset_index(drop=True)

    combined.to_csv(output_csv, index=False)

    print(f"\n✅ Saved {len(combined)} dates to {output_csv}")
    print(f"   Columns: {list(combined.columns)}")

    # Show summary stats
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n   Feature summary (sample of columns):")
        for col in list(numeric_cols)[:5]:
            vals = combined[col].dropna()
            if len(vals) > 0:
                print(f"     {col}: mean={vals.mean():.1f}, std={vals.std():.1f}, "
                      f"min={vals.min():.1f}, max={vals.max():.1f}")

    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Backfill atmospheric data from Open-Meteo")
    parser.add_argument("--city", default="nyc", help="City key (nyc, lax)")
    parser.add_argument("--all", action="store_true", help="Backfill all cities")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if data exists")
    args = parser.parse_args()

    if args.all:
        for city_key in CITIES:
            backfill_city(city_key, force=args.force)
    else:
        backfill_city(args.city, force=args.force)


if __name__ == "__main__":
    main()
