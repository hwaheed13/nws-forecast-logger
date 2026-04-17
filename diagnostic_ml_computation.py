#!/usr/bin/env python3
"""
Diagnostic: Check if ML prediction can be computed for April 17
"""
import sys
sys.path.insert(0, '/sessions/inspiring-keen-gauss/mnt/nws-forecast-logger')

from nws_auto_logger import _read_all_rows, set_city
from datetime import date

set_city('nyc')

# Read all forecast data from CSV
rows, _ = _read_all_rows(include_accu=True)

target_date = '2026-04-17'

print(f"Total rows in CSV: {len(rows)}")
print(f"\n" + "="*80)
print(f"Checking forecasts for {target_date}:")
print("="*80)

# Count NWS forecasts for April 17
nws_fc = []
for r in rows:
    if r.get("forecast_or_actual") != "forecast":
        continue
    if r.get("target_date") != target_date:
        continue
    src = (r.get("source") or "").lower()
    if src == "accuweather":
        continue
    try:
        ph = float(r.get("predicted_high", ""))
    except (ValueError, TypeError):
        continue
    
    ts = r.get("timestamp") or r.get("forecast_time") or ""
    nws_fc.append({
        "timestamp": ts,
        "predicted_high": ph,
        "text": r.get("description", "")[:50]
    })

print(f"\n📊 NWS Forecasts found for {target_date}: {len(nws_fc)}")

if nws_fc:
    print("\nEarliest to latest:")
    for i, fc in enumerate(sorted(nws_fc, key=lambda x: x["timestamp"])[:10]):
        print(f"  {i+1}. {fc['timestamp']}: {fc['predicted_high']}°F")
else:
    print("  ❌ NO NWS FORECASTS FOUND!")
    print(f"\nThis explains why _compute_ml_prediction() returns None!")
    print(f"Check _read_all_rows() is reading the correct CSV files.")

# Also check AccuWeather
accu_fc = []
for r in rows:
    if r.get("forecast_or_actual") != "forecast":
        continue
    if r.get("target_date") != target_date:
        continue
    src = (r.get("source") or "").lower()
    if src != "accuweather":
        continue
    try:
        ph = float(r.get("predicted_high", ""))
    except (ValueError, TypeError):
        continue
    accu_fc.append(ph)

print(f"\n🌤️  AccuWeather forecasts: {len(accu_fc)}")

print(f"\n" + "="*80)
print(f"Dates with any forecast data:")
print("="*80)
dates = set(r.get("target_date") for r in rows if r.get("forecast_or_actual") == "forecast")
for d in sorted(dates)[-10:]:
    count = len([r for r in rows if r.get("target_date") == d and r.get("forecast_or_actual") == "forecast"])
    print(f"  {d}: {count} forecasts")

