#!/usr/bin/env python3
"""Test Supabase PostgREST API directly with requests library."""

import os
import json
import httpx

sb_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
sb_key = os.environ.get("SUPABASE_SERVICE_ROLE", "")

if not sb_url or not sb_key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE environment variables")

# PostgREST endpoint
postgrest_url = f"{sb_url}/rest/v1"

# Headers for service role access
headers = {
    "apikey": sb_key,
    "Authorization": f"Bearer {sb_key}",
    "Content-Type": "application/json",
}

print(f"Testing PostgREST API at: {postgrest_url}")
print(f"Using apikey: {sb_key[:30]}...")

# Test 1: Get count of prediction_logs
print("\n--- Test 1: Count prediction_logs ---")
try:
    url = f"{postgrest_url}/prediction_logs?select=id&count=exact"
    response = httpx.get(url, headers=headers, timeout=10.0)
    print(f"Status: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    if response.status_code == 200:
        print(f"✓ Success! Count header: {response.headers.get('content-range')}")
    else:
        print(f"✗ Error: {response.text}")
except Exception as e:
    print(f"✗ Request failed: {e}")

# Test 2: Get first row
print("\n--- Test 2: Fetch first prediction_logs row ---")
try:
    url = f"{postgrest_url}/prediction_logs?select=id,target_date,city&limit=1"
    response = httpx.get(url, headers=headers, timeout=10.0)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success! Data: {data}")
    else:
        print(f"✗ Error: {response.text}")
except Exception as e:
    print(f"✗ Request failed: {e}")

# Test 3: Get with city filter
print("\n--- Test 3: Fetch prediction_logs for NYC ---")
try:
    url = f"{postgrest_url}/prediction_logs?select=id,target_date,city&city=eq.nyc&limit=1"
    response = httpx.get(url, headers=headers, timeout=10.0)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success! Data: {json.dumps(data, indent=2)}")
    else:
        print(f"✗ Error: {response.text}")
except Exception as e:
    print(f"✗ Request failed: {e}")
