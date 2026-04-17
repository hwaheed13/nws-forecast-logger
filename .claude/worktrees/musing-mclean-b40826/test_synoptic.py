#!/usr/bin/env python3
"""
Quick test to verify Synoptic API token and connectivity.
Run: python test_synoptic.py
"""
import os
import sys

print("=" * 60)
print("SYNOPTIC API TEST")
print("=" * 60)

# Check 1: Is SYNOPTIC_TOKEN set?
token = os.environ.get("SYNOPTIC_TOKEN", "").strip()
if not token:
    print("❌ SYNOPTIC_TOKEN environment variable is NOT set")
    print("\nTo fix:")
    print("  1. Go to GitHub repo Settings → Secrets and variables → Actions")
    print("  2. Verify SYNOPTIC_TOKEN secret exists")
    print("  3. If missing, create it with your Synoptic API token from synopticdata.com")
    sys.exit(1)
else:
    print(f"✅ SYNOPTIC_TOKEN is set (length: {len(token)} chars)")

# Check 2: Try calling get_synoptic_obs_features
print("\nTesting get_synoptic_obs_features()...")
try:
    from synoptic_client import get_synoptic_obs_features
    result = get_synoptic_obs_features(
        lat=40.7834,
        lon=-73.965,
        nws_last=72.5,
        radius_miles=5.0,
        city="nyc"
    )

    # Count how many non-NaN values we got
    import numpy as np
    populated = sum(
        1 for v in result.values()
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    )

    print(f"✅ API call succeeded")
    print(f"   Populated: {populated}/{len(result)} fields")

    if populated == 0:
        print("   ⚠️  BUT: All fields are NaN — API returned no data")
        print("       Possible reasons:")
        print("       - Invalid/revoked SYNOPTIC_TOKEN")
        print("       - Synoptic API rate limit hit")
        print("       - No stations found in 5-mile radius")
        sys.exit(1)
    else:
        print("\n✅ Synoptic API is working correctly!")
        print(f"   Sample data: {dict(list(result.items())[:3])}")
        sys.exit(0)

except Exception as e:
    print(f"❌ API call failed with exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
