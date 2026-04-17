#!/usr/bin/env python3
"""
Analyze historical heating data to determine optimal heating windows.

This script examines:
1. When heating actually occurs by time of day
2. Heating patterns by season and location
3. Confidence levels for each heating window
"""

import os
import json
from datetime import datetime
from supabase import create_client
import statistics

# ─────────────────────────────────────────────────────────────────────
# Initialize Supabase
# ─────────────────────────────────────────────────────────────────────
sb_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
sb_key = os.environ.get("SUPABASE_SERVICE_ROLE", "")

if not sb_url or not sb_key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE environment variables")

# Try using SQL directly since REST API has RLS issues
import psycopg2
from psycopg2.extras import RealDictCursor

# Extract PostgreSQL connection details from Supabase URL
# Format: https://[project].supabase.co
project_id = sb_url.split("//")[1].split(".")[0]
print(f"Project ID: {project_id}")

# ─────────────────────────────────────────────────────────────────────
# Helper: Get season from date
# ─────────────────────────────────────────────────────────────────────
def get_season(date_obj):
    """Return season name for a given date."""
    month = date_obj.month
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:  # 9, 10, 11
        return "fall"


# ─────────────────────────────────────────────────────────────────────
# Query heating data from Supabase SQL
# ─────────────────────────────────────────────────────────────────────
def fetch_heating_data():
    """
    Fetch heating data from prediction_logs.
    Returns list of dicts with: target_date, city, heating_rate
    """
    try:
        # Use direct HTTP API with proper headers
        import httpx

        postgrest_url = f"{sb_url}/rest/v1"
        headers = {
            "apikey": sb_key,
            "Authorization": f"Bearer {sb_key}",
            "Content-Type": "application/json",
        }

        # Fetch all rows with heating_rate populated
        url = f"{postgrest_url}/prediction_logs?select=target_date,city,atm_snapshot&atm_snapshot=not.is.null&limit=5000"

        print("Fetching heating data from Supabase...")
        response = httpx.get(url, headers=headers, timeout=30.0)

        if response.status_code != 200:
            print(f"Warning: API returned {response.status_code}")
            # Fall back to SQL query
            return fetch_heating_data_sql()

        data = response.json()

        # Parse the atm_snapshot JSON
        results = []
        for row in data:
            try:
                snapshot = json.loads(row['atm_snapshot']) if isinstance(row['atm_snapshot'], str) else row['atm_snapshot']
                heating_rate = snapshot.get('obs_snap_heating_rate')
                if heating_rate is not None and heating_rate != 'null':
                    results.append({
                        'target_date': row['target_date'],
                        'city': row['city'],
                        'heating_rate': float(heating_rate) if heating_rate else None
                    })
            except Exception as e:
                pass

        return results
    except Exception as e:
        print(f"API fetch failed: {e}, falling back to SQL")
        return fetch_heating_data_sql()


def fetch_heating_data_sql():
    """
    Fallback: Fetch using raw SQL query (works in Supabase SQL Editor).
    Returns list of dicts with heating data.
    """
    print("\nTo complete the analysis, run this SQL query in Supabase and save as CSV:")
    print("=" * 70)
    print("""
SELECT
  target_date,
  city,
  atm_snapshot->>'obs_snap_heating_rate' as heating_rate
FROM prediction_logs
WHERE atm_snapshot->>'obs_snap_heating_rate' IS NOT NULL
  AND atm_snapshot->>'obs_snap_heating_rate' != 'null'
ORDER BY target_date, city;
""")
    print("=" * 70)
    print("\nThen save the results as 'heating_data.csv' and we'll analyze it.\n")
    return []


# ─────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────
def analyze_heating_by_season_city(data):
    """Analyze heating rates grouped by season and city."""

    analysis = {}

    for city in ['nyc', 'lax']:
        city_data = [d for d in data if d['city'] == city]
        if not city_data:
            continue

        analysis[city] = {}

        for season in ['winter', 'spring', 'summer', 'fall']:
            season_data = []
            for row in city_data:
                date_obj = datetime.fromisoformat(row['target_date'])
                if get_season(date_obj) == season:
                    heating_rate = row['heating_rate']
                    if heating_rate is not None:
                        season_data.append(heating_rate)

            if season_data:
                analysis[city][season] = {
                    'count': len(season_data),
                    'mean': statistics.mean(season_data),
                    'median': statistics.median(season_data),
                    'stdev': statistics.stdev(season_data) if len(season_data) > 1 else 0,
                    'min': min(season_data),
                    'max': max(season_data),
                    'positive_count': sum(1 for h in season_data if h > 0),
                    'positive_pct': 100 * sum(1 for h in season_data if h > 0) / len(season_data)
                }

    return analysis


def print_analysis(analysis):
    """Pretty-print the analysis results."""

    print("\n" + "=" * 80)
    print("HEATING WINDOW ANALYSIS")
    print("=" * 80)

    for city in ['nyc', 'lax']:
        if city not in analysis:
            continue

        print(f"\n{city.upper()}:")
        print("-" * 80)

        for season in ['winter', 'spring', 'summer', 'fall']:
            if season not in analysis[city]:
                continue

            stats = analysis[city][season]
            print(f"\n  {season.upper()}:")
            print(f"    Observations: {stats['count']}")
            print(f"    Mean heating rate: {stats['mean']:.2f} °F/hr")
            print(f"    Median heating rate: {stats['median']:.2f} °F/hr")
            print(f"    Std dev: {stats['stdev']:.2f}")
            print(f"    Range: {stats['min']:.2f} to {stats['max']:.2f} °F/hr")
            print(f"    Positive heating: {stats['positive_count']}/{stats['count']} ({stats['positive_pct']:.1f}%)")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("HEATING WINDOW ANALYSIS - Step 1")
    print("=" * 80)

    # Fetch data
    data = fetch_heating_data()

    if not data:
        print("\n⚠️  No data fetched. The REST API is blocked by RLS policies.")
        print("\nPlease run this SQL query in Supabase SQL Editor to export the data:")
        print("-" * 80)
        print("""
SELECT
  target_date,
  city,
  atm_snapshot->>'obs_snap_heating_rate' as heating_rate
FROM prediction_logs
WHERE atm_snapshot->>'obs_snap_heating_rate' IS NOT NULL
  AND atm_snapshot->>'obs_snap_heating_rate' != 'null'
ORDER BY target_date, city;
        """)
        print("-" * 80)
        print("\nThen we can continue the analysis with the exported data.\n")
    else:
        print(f"\n✓ Fetched {len(data)} heating observations")

        # Analyze
        analysis = analyze_heating_by_season_city(data)

        # Print results
        print_analysis(analysis)

        # Save analysis to JSON
        import json
        with open('heating_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n✓ Analysis saved to heating_analysis.json")
