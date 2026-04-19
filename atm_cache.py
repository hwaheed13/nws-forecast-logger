"""
Atmospheric snapshot caching layer.
Caches snapshots (Synoptic + Open-Meteo + NWS data) to gracefully handle API failures.

Strategy:
1. Always try to fetch fresh data first
2. Only use cache if fetch fails (returns incomplete/NaN data)
3. Cache TTL: 15 minutes (weather doesn't change rapidly)
4. Log when using cached vs fresh data

This prevents cards showing blank/"awaiting" when APIs are rate-limited or down,
while keeping model accuracy high (99% gets fresh data).
"""

import os
import json
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

CACHE_FILE = "atm_snapshot_cache.json"
CACHE_TTL_SECONDS = 900  # 15 minutes


def _is_valid_value(v: Any) -> bool:
    """Check if a value is NOT NaN/None."""
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return True


def _count_valid_values(data: Dict[str, Any]) -> int:
    """Count how many non-NaN values in a dict."""
    return sum(1 for v in data.values() if _is_valid_value(v))


def _load_cache() -> Dict[str, Any]:
    """Load cache from disk."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"  ⚠️ Cache load failed: {e}")
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    """Save cache to disk."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"  ⚠️ Cache save failed: {e}")


def _is_cache_fresh(city: str) -> bool:
    """Check if cache for a city is still valid (within TTL)."""
    cache = _load_cache()
    if city not in cache:
        return False

    entry = cache[city]
    if "timestamp" not in entry:
        return False

    try:
        ts = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
        age_seconds = (datetime.now(ts.tzinfo) - ts).total_seconds()
        is_fresh = age_seconds < CACHE_TTL_SECONDS

        if not is_fresh:
            print(f"  ⏰ {city.upper()} cache expired ({age_seconds:.0f}s old, TTL={CACHE_TTL_SECONDS}s)")

        return is_fresh
    except Exception as e:
        print(f"  ⚠️ Cache TTL check failed: {e}")
        return False


def get_cached_snapshot(city: str) -> Optional[Dict[str, Any]]:
    """
    Get cached snapshot if it's fresh.
    Returns dict of {key: value} or None if not cached/stale.
    """
    if not _is_cache_fresh(city):
        return None

    cache = _load_cache()
    data = cache.get(city, {}).get("data", {})
    if data:
        n_valid = _count_valid_values(data)
        print(f"  💾 Using cached {city.upper()} snapshot ({n_valid} valid fields)")
    return data if data else None


def cache_snapshot(city: str, snapshot: Dict[str, Any]) -> None:
    """
    Cache a complete atmospheric snapshot.
    Only cache if it has substantial data (>50% fields populated).
    """
    n_valid = _count_valid_values(snapshot)
    n_total = len(snapshot)
    percent = round(100.0 * n_valid / n_total, 1) if n_total > 0 else 0

    # Only cache if it's reasonably complete
    if n_valid < n_total * 0.5:
        print(f"  ⏭️  Skipping cache: snapshot too incomplete ({percent}% valid)")
        return

    cache = _load_cache()
    cache[city] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ttl_seconds": CACHE_TTL_SECONDS,
        "data": snapshot,
        "valid_fields": n_valid,
        "total_fields": n_total,
    }
    _save_cache(cache)
    print(f"  💾 Cached {city.upper()} snapshot ({n_valid}/{n_total} fields, {percent}% complete)")


def _cache_age_seconds(city: str) -> Optional[float]:
    """Return current cache age in seconds for a city, or None if no cache."""
    cache = _load_cache()
    if city not in cache:
        return None
    ts_raw = cache[city].get("timestamp")
    if not ts_raw:
        return None
    try:
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        return (datetime.now(ts.tzinfo) - ts).total_seconds()
    except Exception:
        return None


def fill_missing_from_cache(city: str, snapshot: Dict[str, Any]):
    """
    Fill missing (NaN/None) fields in snapshot using cached values.

    Returns (snapshot, meta) where meta is:
        {"filled_keys": [...], "cache_age_s": float_or_None}

    meta is always returned (even when nothing was filled) so callers can
    persist cache provenance into prediction_logs for audit:
    "was this prediction made on live or cached features, and how stale?"

    Usage: After fetching fresh data, fill any gaps with cached values.
    """
    cached = get_cached_snapshot(city)
    if not cached:
        return snapshot, {"filled_keys": [], "cache_age_s": None}

    filled_keys = []
    for key, cached_val in cached.items():
        if key not in snapshot or not _is_valid_value(snapshot[key]):
            if _is_valid_value(cached_val):
                snapshot[key] = cached_val
                filled_keys.append(key)

    age_s = _cache_age_seconds(city)
    if filled_keys:
        age_tag = f", cache age {age_s:.0f}s" if age_s is not None else ""
        print(f"  🔄 Filled {len(filled_keys)} missing fields from {city.upper()} cache{age_tag}")

    return snapshot, {"filled_keys": filled_keys, "cache_age_s": age_s}


def clear_cache() -> None:
    """Clear all cached snapshots."""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print("  🗑️  Cache cleared")
    except Exception as e:
        print(f"  ⚠️ Cache clear failed: {e}")
