#!/usr/bin/env python3
"""
backfill_iem_asos.py — Backfill 4 years of daily peak obs from IEM ASOS
archive into multiyear_atmospheric.csv so v13 BL-safeguard features can
actually learn on historical data instead of the ~28 days in prediction_logs.

Problem this solves:
  model_metadata_v13.json reports n_entrainment_rows=0, n_marine_rows=0,
  n_inland_rows=0 because multiyear_atmospheric.csv has no obs_* columns.
  The v13 features (entrainment_temp_diff, marine_containment, inland_strength)
  depend on station obs — so without obs, they're NaN across 1,567 rows and
  v13 is numerically identical to v11.

What this script does:
  1. For each station in the city panel, request the full 2022-01-01 → today
     temperature timeseries from IEM ASOS archive in ONE call per station.
  2. Group by local date, take the daily peak (max tmpf).
  3. Write peak temps as new obs_{station}_temp columns into
     multiyear_atmospheric.csv (keyed on target_date).
  4. Compute the 3 BL-safeguard features from source cols.
  5. Back up the original CSV before overwriting.

Usage:
  python backfill_iem_asos.py --dry-run          # 7-day sample, no writes
  python backfill_iem_asos.py --city nyc         # NYC full backfill
  python backfill_iem_asos.py --city lax         # LAX full backfill
  python backfill_iem_asos.py --start 2022-01-01 --end 2026-04-18

Safety:
  - --dry-run flag prints sample rows and exits without writing.
  - Original CSV is copied to multiyear_atmospheric.csv.bak-{timestamp}
    before any write.
  - Only ADDS columns — never removes or renames existing ones.
  - Merge is keyed on target_date, so existing atm_* cols are preserved.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import shutil
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

import pandas as pd

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Per-city station panels. Keys are the STID without K prefix (IEM convention
# drops the leading K for CONUS ASOS: KJFK → JFK). Values are the column name
# we want in multiyear_atmospheric.csv.
CITY_STATIONS = {
    "nyc": {
        "NYC":  "obs_knyc_temp",   # Central Park (primary)
        "JFK":  "obs_kjfk_temp",
        "LGA":  "obs_klga_temp",
        "EWR":  "obs_kewr_temp",
        "TEB":  "obs_kteb_temp",
        "CDW":  "obs_kcdw_temp",
        "SMQ":  "obs_ksmq_temp",
    },
    "lax": {
        "LAX":  "obs_klax_temp",
        "BUR":  "obs_kbur_temp",
        "VNY":  "obs_kvny_temp",
        "SMO":  "obs_ksmo_temp",
        "CQT":  "obs_kcqt_temp",
    },
}

CITY_TZ = {"nyc": "America/New_York", "lax": "America/Los_Angeles"}

CITY_CSV = {
    "nyc": "multiyear_atmospheric.csv",
    "lax": "lax_multiyear_atmospheric.csv",
}


def fetch_iem_asos(station: str, start: date, end: date, tz: str) -> pd.DataFrame:
    """Fetch temperature timeseries from IEM ASOS archive. One HTTP call.

    Returns DataFrame with columns [valid, tmpf]. valid is a naive datetime
    in the station's local timezone.
    """
    params = [
        ("station", station),
        ("data", "tmpf"),
        ("year1", str(start.year)),
        ("month1", str(start.month)),
        ("day1", str(start.day)),
        ("year2", str(end.year)),
        ("month2", str(end.month)),
        ("day2", str(end.day)),
        ("tz", tz),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "no"),
        ("report_type", "3"),  # 3 = routine hourly observations
        ("report_type", "4"),  # 4 = specials (gusts, etc) — also has tmpf
    ]
    query = "&".join(f"{k}={v}" for k, v in params)
    url = f"{IEM_URL}?{query}"

    # Retry with backoff for transient IEM hiccups.
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nws-forecast-logger/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"IEM fetch failed for {station}: {last_err}")

    if not raw.strip():
        return pd.DataFrame(columns=["valid", "tmpf"])

    try:
        df = pd.read_csv(io.StringIO(raw))
    except Exception:
        return pd.DataFrame(columns=["valid", "tmpf"])

    if "tmpf" not in df.columns or "valid" not in df.columns:
        return pd.DataFrame(columns=["valid", "tmpf"])

    # tmpf comes as string "M" for missing, "T" for trace — coerce.
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
    df = df.dropna(subset=["valid", "tmpf"])
    # Filter obviously bogus values (temperature well outside Earth range).
    df = df[(df["tmpf"] > -60) & (df["tmpf"] < 140)]
    return df[["valid", "tmpf"]].reset_index(drop=True)


def daily_peak(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse minute/hourly obs to daily peak temp, keyed on local date.

    Returns DataFrame indexed by date with single 'peak' column.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "peak"])
    df = df.copy()
    df["date"] = df["valid"].dt.date
    out = df.groupby("date", as_index=False)["tmpf"].max().rename(columns={"tmpf": "peak"})
    return out


def backup_csv(csv_path: str) -> str:
    """Copy CSV to a timestamped backup before any write."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    bak = f"{csv_path}.bak-{ts}"
    shutil.copy2(csv_path, bak)
    return bak


def compute_bl_features(df: pd.DataFrame, city_key: str) -> pd.DataFrame:
    """Compute the 3 BL-safeguard features from source cols.

    Only NYC has all the inputs; LAX lacks the inland trio and JFK-KNYC pair.
    For LAX we leave the cols NaN and return — v13 is NYC-focused.
    """
    if city_key != "nyc":
        return df

    # entrainment_temp_diff = atm_925mb_temp_mean - obs_latest_temp
    # (obs_latest_temp proxy: daily peak KNYC temp)
    if "atm_925mb_temp_mean" in df.columns and "obs_knyc_temp" in df.columns:
        df["obs_latest_temp"] = df["obs_knyc_temp"]
        df["entrainment_temp_diff"] = (
            pd.to_numeric(df["atm_925mb_temp_mean"], errors="coerce")
            - pd.to_numeric(df["obs_knyc_temp"], errors="coerce")
        ).round(1)

    # marine_containment = obs_kjfk_vs_knyc / atm_bl_height_max
    if "obs_kjfk_temp" in df.columns and "obs_knyc_temp" in df.columns:
        df["obs_kjfk_vs_knyc"] = (
            pd.to_numeric(df["obs_kjfk_temp"], errors="coerce")
            - pd.to_numeric(df["obs_knyc_temp"], errors="coerce")
        ).round(1)
    if "obs_kjfk_vs_knyc" in df.columns and "atm_bl_height_max" in df.columns:
        bl = pd.to_numeric(df["atm_bl_height_max"], errors="coerce")
        df["marine_containment"] = (
            pd.to_numeric(df["obs_kjfk_vs_knyc"], errors="coerce") / bl.where(bl > 0)
        ).round(6)

    # inland_strength = mean(kteb, kcdw, ksmq) - mm_mean
    inland_cols = [c for c in ("obs_kteb_temp", "obs_kcdw_temp", "obs_ksmq_temp") if c in df.columns]
    if inland_cols and "mm_mean" in df.columns:
        inland_mean = df[inland_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        df["inland_strength"] = (inland_mean - pd.to_numeric(df["mm_mean"], errors="coerce")).round(1)

    return df


def run(
    city_key: str,
    start: date,
    end: date,
    dry_run: bool = False,
) -> bool:
    if city_key not in CITY_STATIONS:
        print(f"❌ Unknown city: {city_key}")
        return False

    csv_path = CITY_CSV[city_key]
    stations = CITY_STATIONS[city_key]
    tz = CITY_TZ[city_key]

    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} not found. Run from repo root.")
        return False

    print(f"\n{'='*60}")
    print(f"IEM ASOS backfill — {city_key.upper()}")
    print(f"Range: {start} → {end}")
    print(f"Stations: {', '.join(stations.keys())}")
    print(f"Target CSV: {csv_path}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")

    # Load existing multiyear CSV
    base = pd.read_csv(csv_path)
    if "target_date" not in base.columns:
        print(f"❌ {csv_path} has no target_date column")
        return False
    base["target_date"] = pd.to_datetime(base["target_date"], errors="coerce").dt.date
    before_rows = len(base)
    before_cols = list(base.columns)
    print(f"Loaded {before_rows} rows, {len(before_cols)} columns")

    # Pull each station
    peaks_by_station: Dict[str, pd.DataFrame] = {}
    for station, col in stations.items():
        print(f"  📡 IEM: fetching {station} 2022 → {end} ...")
        try:
            raw = fetch_iem_asos(station, start, end, tz)
        except Exception as e:
            print(f"     ⚠️ {station} failed: {e} — skipping")
            continue
        peaks = daily_peak(raw)
        print(f"     → {len(peaks)} daily peaks from {len(raw)} obs")
        peaks_by_station[col] = peaks
        time.sleep(1.0)  # be polite to IEM

    if not peaks_by_station:
        print("❌ No stations returned data. Aborting.")
        return False

    # Merge each station's peaks into the base df
    merged = base.copy()
    for col, peaks in peaks_by_station.items():
        if peaks.empty:
            continue
        peaks = peaks.rename(columns={"peak": col})
        merged = merged.merge(peaks, left_on="target_date", right_on="date", how="left")
        if "date" in merged.columns:
            merged = merged.drop(columns=["date"])

    # Compute BL-safeguard derived features (NYC only; LAX is a no-op)
    merged = compute_bl_features(merged, city_key)

    # Diff summary
    new_cols = [c for c in merged.columns if c not in before_cols]
    print(f"\n📊 New columns added: {new_cols}")
    for c in new_cols:
        n_valid = merged[c].notna().sum()
        print(f"    {c}: {n_valid}/{len(merged)} rows populated")

    # Test-case spot check
    print("\n📋 Sample rows (2024-04-12, 2025-04-15 if present):")
    for sample_date in ["2024-04-12", "2025-04-15", "2026-04-15"]:
        sd = pd.to_datetime(sample_date).date()
        row = merged[merged["target_date"] == sd]
        if not row.empty:
            print(f"  {sample_date}:")
            for c in new_cols:
                val = row.iloc[0].get(c)
                print(f"    {c} = {val}")

    if dry_run:
        print("\n🏜️  DRY RUN — no CSV write. Re-run without --dry-run to commit.")
        return True

    # Safety: back up original CSV
    bak = backup_csv(csv_path)
    print(f"\n🛟 Backup: {bak}")

    # Write back, preserving row count
    if len(merged) != before_rows:
        print(f"❌ Row count changed ({before_rows} → {len(merged)}). Aborting to protect data.")
        return False

    merged.to_csv(csv_path, index=False)
    print(f"✅ Wrote {csv_path} ({len(merged)} rows, {len(merged.columns)} columns)")
    print(f"   Added cols: {new_cols}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill IEM ASOS obs into multiyear CSV for v13 BL-safeguard training")
    parser.add_argument("--city", default="nyc", choices=["nyc", "lax"])
    parser.add_argument("--start", default="2022-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="7-day sample only, no writes")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()

    if args.dry_run:
        # Dry-run uses a 7-day window ending at `end` to keep IEM responses tiny.
        start = end - timedelta(days=7)
        print(f"(dry-run: narrowing window to {start} → {end})")

    ok = run(args.city, start, end, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)
