#!/usr/bin/env python3
"""Test Supabase connection and RLS policy issues."""

import os
from supabase import create_client

sb_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
sb_key = os.environ.get("SUPABASE_SERVICE_ROLE", "")

if not sb_url or not sb_key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE environment variables")

print(f"Connecting to: {sb_url}")
print(f"Using service role key: {sb_key[:30]}...")

sb = create_client(sb_url, sb_key)
print("✓ Supabase client created")

# Test 1: Try to count rows in prediction_logs
print("\n--- Test 1: Count prediction_logs rows ---")
try:
    result = sb.table("prediction_logs").select("id", count="exact").limit(1).execute()
    print(f"✓ Count query succeeded: {result.count} total rows")
except Exception as e:
    print(f"✗ Count query failed: {e}")

# Test 2: Try to fetch a single row
print("\n--- Test 2: Fetch single row from prediction_logs ---")
try:
    result = sb.table("prediction_logs").select("id, target_date, city").limit(1).execute()
    if result.data:
        print(f"✓ Fetch succeeded: {len(result.data)} row(s)")
        print(f"  Sample: {result.data[0]}")
    else:
        print("✓ Fetch succeeded but no data")
except Exception as e:
    print(f"✗ Fetch failed: {e}")

# Test 3: Try to fetch with eq filter (city filter)
print("\n--- Test 3: Fetch with city filter ---")
try:
    result = sb.table("prediction_logs").select("id, target_date, city").eq("city", "nyc").limit(1).execute()
    if result.data:
        print(f"✓ Filtered fetch succeeded: {len(result.data)} row(s)")
        print(f"  Sample: {result.data[0]}")
    else:
        print("✓ Filtered fetch succeeded but no data")
except Exception as e:
    print(f"✗ Filtered fetch failed: {e}")

# Test 4: Check nws_observations table
print("\n--- Test 4: Check nws_observations table ---")
try:
    result = sb.table("nws_observations").select("id").limit(1).execute()
    if result.data:
        print(f"✓ nws_observations accessible: {len(result.data)} row(s)")
    else:
        print("✓ nws_observations accessible but no data")
except Exception as e:
    print(f"✗ nws_observations not accessible: {e}")
