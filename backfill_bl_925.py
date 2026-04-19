#!/usr/bin/env python3
"""
backfill_bl_925.py — Fill atm_bl_height_max/mean and atm_925mb_temp_max/mean
into multiyear_atmospheric.csv from Open-Meteo's historical-forecast-api.

Why this exists:
  The existing multiyear CSV has 925mb coverage starting 2024-02 and BL-height
  only from 2026-01 — both depend on pressure-level / forecast variables that
  the plain ERA5 archive-api does not serve. Hitting historical-forecast-api
  (which backs historical GFS/ECMWF forecasts) fills the full 2022-01 → today
  window for both fields.

  Downstream impact: atm_925mb_temp_mean gates v13's entrainment_temp_diff,
  and atm_bl_height_max gates marine_containment. Previous retrain had
  entrainment_temp_diff populated on 805/1568 rows and marine_containment on
  79/1568 — this backfill lifts both toward 1500+/1568.

Safety:
  - Backs up the CSV to .bak-{timestamp} before writing.
  - Only fills rows where the target column is NaN (never overwrites existing
    values that may have come from higher-quality sources like live forecast
    snapshots).
  - Chunks requests into 90-day windows with 1s sleep between — polite to the
    API, avoids 429 cascades like we saw on IEM.

Usage:
    python backfill_bl_925.py --city nyc
    python backfill_bl_925.py --city lax
    python backfill_bl_925.py --city nyc --dry-run
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

CITIES = {
    "nyc": {
        "lat": 40.7834, "lon": -73.965, "tz": "America/New_York",
        "csv": "multiyear_atmospheric.csv",
    },
    "lax": {
        "lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles",
        "csv": "lax_multiyear_atmospheric.csv",
    },
}

HOURLY_VARS = ["boundary_layer_height", "temperature_925hPa"]
CHUNK_DAYS = 90


def _get(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if i < retries - 1:
                time.sleep(1.0 * (i + 1))
            else:
                raise RuntimeError(f"Open-Meteo request failed: {e}") from e


def fetch_chunk(lat: float, lon: float, start: str, end: str, tz: str) -> pd.DataFrame:
    """Fetch one date window from historical-forecast-api."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "temperature_unit": "fahrenheit",
        "timezone": tz,
    }
    qs = urllib.parse.urlencode(params)
    data = _get(f"{HISTORICAL_FORECAST_URL}?{qs}")
    if "error" in data:
        raise RuntimeError(f"API error: {data.get('reason', data['error'])}")
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame()
    df = pd.DataFrame({"time": pd.to_datetime(times)})
    for v in HOURLY_VARS:
        df[v] = hourly.get(v, [None] * len(times))
    return df


def fetch_range(lat: float, lon: float, start: str, end: str, tz: str) -> pd.DataFrame:
    """Fetch a multi-year range in CHUNK_DAYS windows."""
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    frames = []
    cursor = d0
    while cursor <= d1:
        window_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), d1)
        print(f"  📡 {cursor} → {window_end}", flush=True)
        try:
            chunk = fetch_chunk(lat, lon, cursor.isoformat(), window_end.isoformat(), tz)
            if not chunk.empty:
                frames.append(chunk)
        except Exception as e:
            print(f"     ⚠️ chunk failed, skipping: {e}")
        cursor = window_end + timedelta(days=1)
        time.sleep(1.0)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"])


def daily_aggregates(hourly: pd.DataFrame) -> pd.DataFrame:
    """Compute per-day aggregates matching extract_daily_atmospheric's logic."""
    if hourly.empty:
        return pd.DataFrame()
    hourly = hourly.copy()
    hourly["date"] = hourly["time"].dt.date
    hourly["hour"] = hourly["time"].dt.hour
    # 925mb: 10am–6pm window (matches open_meteo_client)
    m_925 = (hourly["hour"] >= 10) & (hourly["hour"] <= 18)
    t925 = hourly[m_925].groupby("date")["temperature_925hPa"].agg(["max", "mean"])
    t925.columns = ["atm_925mb_temp_max", "atm_925mb_temp_mean"]
    # BL height: 10am–4pm window
    m_bl = (hourly["hour"] >= 10) & (hourly["hour"] <= 16)
    bl = hourly[m_bl].groupby("date")["boundary_layer_height"].agg(["max", "mean"])
    bl.columns = ["atm_bl_height_max", "atm_bl_height_mean"]
    out = t925.join(bl, how="outer").reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


def run(city: str, dry_run: bool = False) -> int:
    cfg = CITIES[city]
    csv_path = Path(cfg["csv"])
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    df["target_date"] = df["target_date"].astype(str).str[:10]
    print(f"Loaded {len(df)} rows from {csv_path}")

    cols = ["atm_925mb_temp_max", "atm_925mb_temp_mean",
            "atm_bl_height_max", "atm_bl_height_mean"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        n = df[c].notna().sum()
        print(f"  before: {c}: {n}/{len(df)}")

    start = df["target_date"].min()
    end = df["target_date"].max()
    # historical-forecast-api coverage begins 2022-03-23
    if start < "2022-03-23":
        start = "2022-03-23"
    print(f"\nFetching {cfg['lat']},{cfg['lon']} from {start} → {end}")

    hourly = fetch_range(cfg["lat"], cfg["lon"], start, end, cfg["tz"])
    if hourly.empty:
        print("❌ No hourly data returned")
        return 1
    print(f"\n  Retrieved {len(hourly)} hourly rows")

    daily = daily_aggregates(hourly)
    print(f"  Computed {len(daily)} daily aggregates")

    # Fill-only merge: write fetched value only where existing is NaN.
    merged = df.merge(daily, left_on="target_date", right_on="date",
                      how="left", suffixes=("", "_new"))
    filled_counts = {}
    for c in cols:
        new_col = f"{c}_new"
        if new_col not in merged.columns:
            continue
        before = merged[c].notna().sum()
        mask = merged[c].isna() & merged[new_col].notna()
        merged.loc[mask, c] = merged.loc[mask, new_col]
        after = merged[c].notna().sum()
        filled_counts[c] = after - before
    merged = merged.drop(columns=[c for c in merged.columns
                                    if c.endswith("_new") or c == "date"])

    print("\n📊 Fill results:")
    for c in cols:
        n = merged[c].notna().sum()
        print(f"  {c}: {n}/{len(merged)} (+{filled_counts.get(c, 0)} filled)")

    if dry_run:
        print("\n🏜️  DRY RUN — no CSV write.")
        return 0

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    backup = csv_path.with_suffix(f".csv.bak-{ts}")
    csv_path.rename(backup)
    print(f"\n🛟 Backup: {backup}")
    merged.to_csv(csv_path, index=False)
    print(f"✅ Wrote {csv_path} ({len(merged)} rows, {len(merged.columns)} cols)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="nyc", choices=list(CITIES.keys()))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.exit(run(args.city, dry_run=args.dry_run))
