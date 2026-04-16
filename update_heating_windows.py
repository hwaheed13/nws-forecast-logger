#!/usr/bin/env python3
"""
Daily heating window updater.
Analyzes the latest observation data and updates heating_windows.py.
Runs nightly as part of the GitHub Actions workflow.

This script:
1. Fetches the latest obs_heating_rate data from prediction_logs
2. Analyzes heating patterns by season and city
3. Updates the HEATING_WINDOWS config if patterns have changed significantly
4. Commits changes to git if needed

Requires: DATABASE_URL environment variable (PostgreSQL connection string)
"""

import os
import json
import statistics
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────
# Database connection
# ─────────────────────────────────────────────────────────────────────
def get_db_connection():
    """Create PostgreSQL connection from DATABASE_URL env var."""
    try:
        import psycopg2
    except ImportError:
        print("⚠️  psycopg2 not installed. Installing...")
        os.system("pip install psycopg2-binary")
        import psycopg2

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL environment variable not set")
        return None

    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return None


def fetch_latest_heating_data() -> Optional[List[Dict]]:
    """Fetch latest heating rate data from PostgreSQL."""
    print("Connecting to database...")
    conn = get_db_connection()
    if not conn:
        return None

    try:
        cur = conn.cursor()

        # Query heating data by city and season
        query = """
        SELECT
          city,
          CASE
            WHEN EXTRACT(MONTH FROM target_date::date) IN (12, 1, 2) THEN 'winter'
            WHEN EXTRACT(MONTH FROM target_date::date) IN (3, 4, 5) THEN 'spring'
            WHEN EXTRACT(MONTH FROM target_date::date) IN (6, 7, 8) THEN 'summer'
            ELSE 'fall'
          END as season,
          COUNT(*) as observation_count,
          AVG(CAST(atm_snapshot->>'obs_snap_heating_rate' AS FLOAT)) as mean_heating_rate,
          SUM(CASE WHEN CAST(atm_snapshot->>'obs_snap_heating_rate' AS FLOAT) > 0 THEN 1 ELSE 0 END) as positive_count
        FROM prediction_logs
        WHERE atm_snapshot->>'obs_snap_heating_rate' IS NOT NULL
          AND atm_snapshot->>'obs_snap_heating_rate' != 'null'
        GROUP BY city, season
        ORDER BY city,
          CASE WHEN season = 'winter' THEN 1
               WHEN season = 'spring' THEN 2
               WHEN season = 'summer' THEN 3
               ELSE 4 END
        """

        cur.execute(query)
        rows = cur.fetchall()

        if not rows:
            print("⚠️  No heating data found in database")
            return None

        # Convert to list of dicts
        data = []
        for row in rows:
            data.append({
                'city': row[0],
                'season': row[1],
                'observation_count': row[2],
                'mean_heating_rate': row[3],
                'positive_count': row[4],
            })

        cur.close()
        conn.close()

        print(f"✓ Fetched heating data: {len(data)} season/city combinations")
        return data

    except Exception as e:
        print(f"❌ Query failed: {e}")
        return None


def analyze_heating_data(data: List[Dict]) -> Optional[Dict]:
    """
    Analyze heating data and determine optimal windows.

    Returns a dict like:
    {
        'nyc': {'winter': (10, 16), 'spring': (10, 16), ...},
        'lax': {'winter': (10, 16), 'spring': (10, 13), ...},
    }
    """
    if not data:
        return None

    analysis = {}

    for city in ['nyc', 'lax']:
        analysis[city] = {}
        city_data = [d for d in data if d['city'].lower() == city]

        if not city_data:
            print(f"⚠️  No data for {city.upper()}")
            continue

        for season in ['winter', 'spring', 'summer', 'fall']:
            season_data = [d for d in city_data if d['season'] == season]

            if not season_data:
                # Fall back to default window if no data for this season
                analysis[city][season] = (10, 16) if city == 'nyc' else (10, 13)
                continue

            # Calculate statistics
            mean_rate = season_data[0]['mean_heating_rate']
            observation_count = season_data[0]['observation_count']
            positive_count = season_data[0]['positive_count']
            positive_pct = 100 * positive_count / observation_count if observation_count > 0 else 0

            # Determine window based on heating characteristics
            if city == 'nyc':
                # NYC: Always show 10 AM - 4 PM (consistent positive heating)
                window = (10, 16)
            else:  # LAX
                # LAX: Adjust based on marine layer effect
                if mean_rate > 0 and positive_pct > 70:
                    window = (10, 16)  # Winter: solid heating
                elif mean_rate > -0.5 and positive_pct > 30:
                    window = (10, 13)  # Spring/Fall: slight cooling, narrow window
                else:
                    window = (10, 12)  # Summer: strong marine cap, very narrow window

            analysis[city][season] = window

            print(f"  {city.upper()} {season.upper()}: {mean_rate:.2f}°F/hr, {positive_pct:.0f}% positive → Window {window[0]:02d}:00-{window[1]:02d}:00")

    return analysis


def update_heating_windows_file(new_windows: Dict) -> bool:
    """
    Update heating_windows.py with new window configuration.
    Returns True if file was modified.
    """
    # Read current file
    try:
        with open('heating_windows.py', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ heating_windows.py not found")
        return False

    # Build new HEATING_WINDOWS dict string
    new_windows_str = "HEATING_WINDOWS = {\n"
    for city in ['nyc', 'lax']:
        new_windows_str += f'    "{city}": {{\n'
        for season in ['winter', 'spring', 'summer', 'fall']:
            window = new_windows[city][season]
            new_windows_str += f'        "{season}": {window},\n'
        new_windows_str += "    },\n"
    new_windows_str += "}"

    # Replace in content using regex
    pattern = r'HEATING_WINDOWS = \{[^}]*(?:\{[^}]*\}[^}]*)*\}'
    updated_content = re.sub(pattern, new_windows_str, content, flags=re.DOTALL)

    # Check if anything changed
    if updated_content == content:
        print("✓ No changes needed — heating windows are up to date")
        return False

    # Write updated file
    try:
        with open('heating_windows.py', 'w') as f:
            f.write(updated_content)
        print("✓ Updated heating_windows.py with new windows")
        return True
    except Exception as e:
        print(f"❌ Failed to write heating_windows.py: {e}")
        return False


def main():
    print("=" * 80)
    print("DAILY HEATING WINDOW UPDATER")
    print("=" * 80)
    print(f"Run time: {datetime.now().isoformat()}\n")

    # Fetch data
    print("Step 1: Fetching latest heating data from database...")
    data = fetch_latest_heating_data()

    if not data:
        print("\n❌ Failed to fetch data. Exiting.")
        return False

    # Analyze
    print("\nStep 2: Analyzing heating patterns...")
    new_windows = analyze_heating_data(data)

    if not new_windows:
        print("\n❌ Analysis failed. Exiting.")
        return False

    # Update file
    print("\nStep 3: Updating heating_windows.py...")
    updated = update_heating_windows_file(new_windows)

    print("\n" + "=" * 80)
    if updated:
        print("✅ SUCCESS: heating_windows.py has been updated")
        print("Changes will be committed by GitHub Actions workflow")
    else:
        print("✅ SUCCESS: heating windows analysis complete (no changes needed)")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
