#!/usr/bin/env python3
"""
backfill_official_actuals.py — Overwrite actual_high in the multiyear CSVs with
the OFFICIAL NWS CLI daily high from the IEM CLI archive.

Why this exists (2026-07-25 audit):
  multiyear_atmospheric.csv's actual_high came from the Open-Meteo archive —
  a model GRID temperature, not the station the market settles on. Audit found
  it disagrees with official KNYC CLI highs by a MEAN of 2.12°F (70% of days
  ≥1°F, worst 16.7°F on 2026-02-28). Every model trained on that column was
  learning to predict the wrong number, and the v16 "improvement_vs_hrrr_alone"
  moat metric was measured against the wrong truth.

  The IEM CLI archive (https://mesonet.agron.iastate.edu/json/cli.py) serves
  the official CLI product highs for KNYC/KLAX back beyond 2022 — the exact
  value Kalshi settles on.

What it does:
  - Fetches CLI daily highs per year for the city's obs_station.
  - Preserves the old grid value in a new column `grid_actual_high`
    (only written the first time, so re-runs stay idempotent).
  - Overwrites actual_high with the official CLI high wherever available.
  - Backs up the CSV to .bak-{timestamp} before writing.

Usage:
    python backfill_official_actuals.py --city nyc
    python backfill_official_actuals.py --city lax
    python backfill_official_actuals.py --city nyc --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from city_config import get_city_config

IEM_CLI_URL = "https://mesonet.agron.iastate.edu/json/cli.py?station={station}&year={year}"


def _get(url: str, retries: int = 3) -> dict:
    last_exc = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_exc = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}") from last_exc


def fetch_official_highs(station: str, years: list[int]) -> dict[str, float]:
    """Return {date_iso: official_high_F} from the IEM CLI archive."""
    highs: dict[str, float] = {}
    for year in years:
        data = _get(IEM_CLI_URL.format(station=station, year=year))
        rows = data.get("results", [])
        n = 0
        for r in rows:
            d, h = r.get("valid"), r.get("high")
            if d and h is not None and h != "M":
                try:
                    highs[str(d)] = float(h)
                    n += 1
                except (ValueError, TypeError):
                    pass
        print(f"  {station} {year}: {n} official CLI highs")
        time.sleep(1)  # be polite
    return highs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="nyc", choices=["nyc", "lax"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = get_city_config(args.city)
    station = cfg["obs_station"]
    csv_path = Path(f"{cfg['model_prefix']}multiyear_atmospheric.csv")
    if not csv_path.exists():
        print(f"❌ {csv_path} not found")
        return 1

    df = pd.read_csv(csv_path)
    if "target_date" not in df.columns or "actual_high" not in df.columns:
        print("❌ CSV missing target_date/actual_high columns")
        return 1

    dates = pd.to_datetime(df["target_date"], errors="coerce")
    years = sorted({int(y) for y in dates.dt.year.dropna().unique()})
    print(f"Backfilling official {station} CLI highs for years {years[0]}–{years[-1]}")
    official = fetch_official_highs(station, years)
    if not official:
        print("❌ No official highs fetched — aborting, CSV untouched")
        return 1

    # Preserve the original grid value once; idempotent on re-runs.
    if "grid_actual_high" not in df.columns:
        df["grid_actual_high"] = df["actual_high"]
    else:
        unset = df["grid_actual_high"].isna()
        df.loc[unset, "grid_actual_high"] = df.loc[unset, "actual_high"]

    off_series = df["target_date"].astype(str).map(official)
    have_mask = off_series.notna()
    old = pd.to_numeric(df["actual_high"], errors="coerce")
    delta = (off_series - old)[have_mask]
    changed = int((delta.abs() > 0.05).sum())

    print(f"\n  Rows in CSV:              {len(df)}")
    print(f"  Rows with official high:  {int(have_mask.sum())}")
    print(f"  Rows changed (>0.05°F):   {changed}")
    if len(delta):
        print(f"  Mean |official − grid|:   {delta.abs().mean():.2f}°F")
        print(f"  Max  |official − grid|:   {delta.abs().max():.2f}°F")
    print(f"  Rows with NO official high (grid value kept): {int((~have_mask).sum())}")

    if args.dry_run:
        print("\n(dry-run: CSV untouched)")
        return 0

    backup = csv_path.with_suffix(csv_path.suffix + f".bak-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    csv_path.rename(backup)
    df.loc[have_mask, "actual_high"] = off_series[have_mask]
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Wrote {csv_path} (backup: {backup.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
