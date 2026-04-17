#!/usr/bin/env python3
"""
Run migration: Add snapshot_hour column to prediction_logs
Usage: SUPABASE_SERVICE_ROLE=your_key python3 run_migration.py
"""
import os
import urllib.request
import json
import sys

SUPABASE_URL = "https://ztjtuhkjkqchsiuuvmzs.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE")

if not SUPABASE_KEY:
    print("ERROR: SUPABASE_SERVICE_ROLE environment variable not set")
    print("Usage: SUPABASE_SERVICE_ROLE=your_key python3 run_migration.py")
    sys.exit(1)

sql = """
ALTER TABLE prediction_logs
ADD COLUMN snapshot_hour INTEGER DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_prediction_logs_snapshot_hour
ON prediction_logs(target_date, snapshot_hour)
WHERE snapshot_hour IS NOT NULL;

COMMENT ON COLUMN prediction_logs.snapshot_hour IS
'3-hour bucket for snapshot archival (0, 3, 6, 9, 12, 15, 18, 21).
Enables training pipeline to see intraday atmospheric progression.
Values: 0=midnight-3am, 3=3-6am, 6=6-9am, 9=9am-noon, 12=noon-3pm, 15=3-6pm, 18=6-9pm, 21=9pm-midnight';
"""

print("🔄 Running migration: Add snapshot_hour column...")
print(f"📍 Target: Supabase project ztjtuhkjkqchsiuuvmzs")
print(f"📊 Table: prediction_logs")
print(f"📋 Column: snapshot_hour (INTEGER)")

# Use Supabase SQL endpoint
url = f"{SUPABASE_URL}/rest/v1/sql"

payload = {"query": sql}
data = json.dumps(payload).encode('utf-8')

req = urllib.request.Request(
    url,
    data=data,
    method="POST",
    headers={
        "Content-Type": "application/json",
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
)

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = resp.read().decode()
        print("\n✅ Migration successful!")
        print("📝 SQL executed:")
        print(sql)
except urllib.error.HTTPError as e:
    error_body = e.read().decode()
    print(f"\n❌ Migration failed with HTTP {e.code}")
    print(f"Response: {error_body}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Migration failed: {e}")
    sys.exit(1)
