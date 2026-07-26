#!/usr/bin/env python3
"""
production_report.py — THE moat measurement.

Compares the production ML model's logged predictions (predictions_ledger.csv,
exported nightly from Supabase) against the public benchmarks it must beat:
NWS day-before, AccuWeather day-before. All errors are measured against the
OFFICIAL CLI actual highs in the NWS forecast log — the number the market
settles on.

CV metrics in model_metadata_v16.json estimate the moat; this report measures
it. If ml_mae is not clearly below nws_mae here, the moat does not exist in
production, whatever CV says.

Usage:
    python production_report.py            # both cities, 30/90/all-day windows
    python production_report.py --city nyc
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

from city_config import CITIES, get_city_config
from nwslogger.data.truth import load_official_actuals


def load_day_before_forecasts(csv_path: str, date_col: str, ts_col: str,
                              value_col: str, type_col: str | None = None) -> dict[str, float]:
    """Latest forecast whose timestamp's calendar date == target_date - 1."""
    best: dict[str, tuple[str, float]] = {}
    p = Path(csv_path)
    if not p.exists():
        return {}
    with p.open() as f:
        for r in csv.DictReader(f):
            if type_col and r.get(type_col) != "forecast":
                continue
            d, ts, v = r.get(date_col), r.get(ts_col, ""), r.get(value_col)
            if not d or not ts or not v:
                continue
            try:
                target = datetime.strptime(str(d)[:10], "%Y-%m-%d")
                ts_day = str(ts)[:10]
                if datetime.strptime(ts_day, "%Y-%m-%d") != target - timedelta(days=1):
                    continue
                fv = float(v)
            except (ValueError, TypeError):
                continue
            if d not in best or ts > best[d][0]:
                best[d] = (ts, fv)
    return {d: v for d, (_, v) in best.items()}


def load_ledger(prefix: str) -> dict[str, dict]:
    """Night-before (canonical) production predictions keyed by target_date."""
    p = Path(f"{prefix}predictions_ledger.csv")
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    with p.open() as f:
        for r in csv.DictReader(f):
            d = r.get("target_date")
            if not d:
                continue
            # Prefer the canonical (first-of-day) prediction — the honest
            # night-before call, comparable to a day-before public forecast.
            row = out.setdefault(d, {})
            mlf = r.get("ml_f_canonical") or r.get("ml_f")
            res = r.get("ml_result_canonical") or r.get("ml_result")
            if mlf:
                row.setdefault("ml_f", mlf)
            if res:
                row.setdefault("ml_result", res)
    return out


def mae(pairs: list[tuple[float, float]]) -> float | None:
    return round(sum(abs(a - b) for a, b in pairs) / len(pairs), 2) if pairs else None


def report_city(city: str, windows=(30, 90, None)) -> None:
    cfg = get_city_config(city)
    prefix = cfg["model_prefix"]
    actuals = load_official_actuals(cfg["nws_csv"])
    nws_db = load_day_before_forecasts(cfg["nws_csv"], "target_date", "timestamp",
                                       "predicted_high", type_col="forecast_or_actual")
    accu_db = load_day_before_forecasts(cfg["accu_csv"], "target_date", "timestamp",
                                        "predicted_high")
    ledger = load_ledger(prefix)

    today = datetime.now()
    print(f"\n{'═' * 62}\n{cfg['label']} — production accuracy vs OFFICIAL actuals\n{'═' * 62}")
    if not ledger:
        print(f"  ⚠️ No {prefix}predictions_ledger.csv — run export_predictions_ledger.py "
              f"(needs Supabase creds). Public benchmarks shown anyway.")

    hdr = f"  {'window':<10}{'n(ml)':>6}{'ML':>7}{'ML win%':>9}{'n':>6}{'NWS d-1':>9}{'n':>6}{'Accu d-1':>9}"
    print(hdr)
    for w in windows:
        lo = (today - timedelta(days=w)).strftime("%Y-%m-%d") if w else "0000"
        dates = [d for d in actuals if d >= lo]
        ml_pairs, wins, scored = [], 0, 0
        for d in dates:
            row = ledger.get(d)
            if row and row.get("ml_f"):
                try:
                    ml_pairs.append((float(row["ml_f"]), actuals[d]))
                except (ValueError, TypeError):
                    pass
                res = (row.get("ml_result") or "").upper()
                if res in ("WIN", "MISS"):
                    scored += 1
                    wins += res == "WIN"
        nws_pairs = [(nws_db[d], actuals[d]) for d in dates if d in nws_db]
        accu_pairs = [(accu_db[d], actuals[d]) for d in dates if d in accu_db]
        win_pct = f"{100 * wins / scored:.0f}%" if scored else "—"
        print(f"  {('last ' + str(w) + 'd') if w else 'all':<10}"
              f"{len(ml_pairs):>6}{str(mae(ml_pairs) or '—'):>7}{win_pct:>9}"
              f"{len(nws_pairs):>6}{str(mae(nws_pairs) or '—'):>9}"
              f"{len(accu_pairs):>6}{str(mae(accu_pairs) or '—'):>9}")

    print("  (ML = canonical night-before production prediction; win% = scored "
          "Kalshi bucket hits; NWS/Accu = latest day-before forecast)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=None, choices=list(CITIES.keys()))
    args = ap.parse_args()
    for c in ([args.city] if args.city else list(CITIES.keys())):
        report_city(c)
