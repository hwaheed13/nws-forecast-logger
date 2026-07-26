#!/usr/bin/env python3
"""
export_predictions_ledger.py — Export production ML predictions from Supabase
prediction_logs into a git-tracked CSV so real-world performance is auditable
from the repo alone.

Why this exists (2026-07-25 audit):
  The local snapshot logger died 2025-12-18. From the repo alone there was NO
  recoverable record of what the production model actually predicted for any
  later date (and none ever for LAX) — the "moat" was only ever a CV number.
  This ledger makes production performance measurable: one row per
  (city, target_date), appended nightly by the lightweight retrain workflow.

  On the last measurable local sample (NYC Oct–Dec 2025, 67 days), the ML
  night-before LOST to both NWS and AccuWeather day-before by ~0.18°F MAE.
  Whether that's still true post-v16 is exactly what this file answers —
  see production_report.py.

Usage:
    python export_predictions_ledger.py --city nyc
    python export_predictions_ledger.py --city lax
Requires SUPABASE_URL + SUPABASE_SERVICE_ROLE env vars.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import requests

from city_config import get_city_config

# Curated, stable column order. Anything Supabase doesn't have stays blank —
# never crash the export on schema drift.
LEDGER_COLS = [
    "city", "target_date", "is_canonical",
    "ml_f", "ml_bucket", "ml_confidence", "ml_version",
    "ml_f_canonical", "ml_bucket_canonical",
    "ml_result", "ml_result_canonical", "bucket_rank_hit",
    "ml_bucket_2", "ml_bucket_2_prob",
    "ml_actual_high", "nws_last", "accu_last",
    "created_at", "updated_at",
]

PAGE = 1000


def fetch_all_rows(city: str) -> list[dict]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("⚠️ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE — cannot export ledger")
        return []
    rows: list[dict] = []
    offset = 0
    while True:
        resp = requests.get(
            f"{url}/rest/v1/prediction_logs"
            f"?city=eq.{city}&select=*&order=target_date.asc",
            headers={
                "apikey": key, "Authorization": f"Bearer {key}",
                "Range-Unit": "items", "Range": f"{offset}-{offset + PAGE - 1}",
            },
            timeout=30,
        )
        if resp.status_code not in (200, 206):
            print(f"⚠️ Supabase fetch failed ({resp.status_code}): {resp.text[:200]}")
            return rows
        batch = resp.json() or []
        rows.extend(batch)
        if len(batch) < PAGE:
            return rows
        offset += PAGE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="nyc", choices=["nyc", "lax"])
    args = ap.parse_args()

    cfg = get_city_config(args.city)
    out_path = Path(f"{cfg['model_prefix']}predictions_ledger.csv")

    rows = fetch_all_rows(args.city)
    if not rows:
        print(f"No rows exported for {args.city} — leaving {out_path} untouched")
        # Missing creds/network must not delete history already committed.
        return 0

    # One ledger row per (target_date, is_canonical); last write wins so the
    # nightly export naturally picks up scored results patched onto old rows.
    dedup: dict[tuple, dict] = {}
    for r in rows:
        d = str(r.get("target_date") or "")
        # Skip bookkeeping sentinels (e.g. ensemble_weights rows at 9999-12-31)
        # — the ledger is strictly one row per real settled/pending day.
        if not d.startswith("20"):
            continue
        k = (d, str(r.get("is_canonical")))
        dedup[k] = r

    missing = [c for c in LEDGER_COLS if rows and c not in rows[-1]]
    if missing:
        print(f"  (columns absent in prediction_logs, left blank: {missing})")

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS, extrasaction="ignore")
        w.writeheader()
        for k in sorted(dedup):
            row = {c: dedup[k].get(c, "") for c in LEDGER_COLS}
            row["city"] = args.city
            w.writerow({c: ("" if v is None else v) for c, v in row.items()})

    n_scored = sum(1 for r in dedup.values() if r.get("ml_actual_high") is not None)
    print(f"✅ Wrote {out_path}: {len(dedup)} rows ({n_scored} scored) "
          f"from {min(k[0] for k in dedup)} → {max(k[0] for k in dedup)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
