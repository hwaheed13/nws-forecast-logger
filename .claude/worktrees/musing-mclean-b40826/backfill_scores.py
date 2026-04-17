#!/usr/bin/env python3
"""
backfill_scores.py
──────────────────
Sets ml_result_canonical (WIN/MISS) on prediction_logs rows that have:
  - ml_bucket_canonical set
  - ml_actual_high set
  - ml_result_canonical is null

Bucket scoring rules (mirrors prediction_writer.py):
  "55-56"  → WIN if actual_int in {55, 56}
  ">=70"   → WIN if actual_int >= 70
  "<=53"   → WIN if actual_int <= 53
  "<=55"   → WIN if actual_int <= 55
  Boundary buckets use the lower/upper bound from the canonical string.

Also sets ml_result on same rows if not already set (latest bucket = canonical
for all backfilled rows since we only have one prediction per day).

Dry-run by default — pass --apply to write.

Usage:
    SUPABASE_URL=... SUPABASE_SERVICE_ROLE=... python3 backfill_scores.py
    SUPABASE_URL=... SUPABASE_SERVICE_ROLE=... python3 backfill_scores.py --apply
"""

import os, sys, json, re, urllib.request, urllib.parse

DRY_RUN = "--apply" not in sys.argv

SUPA_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPA_KEY = os.environ["SUPABASE_SERVICE_ROLE"]

HEADERS = {
    "apikey": SUPA_KEY,
    "Authorization": f"Bearer {SUPA_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

def sb_get(path_and_query):
    url = f"{SUPA_URL}/rest/v1/{path_and_query}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())

def sb_patch(table, query_str, data):
    url = f"{SUPA_URL}/rest/v1/{table}?{query_str}"
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={**HEADERS, "Prefer": "return=minimal"},
        method="PATCH"
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status

def score_bucket(bucket: str, actual: int) -> str:
    """Return 'WIN' or 'MISS' for a given bucket string and actual high."""
    bucket = bucket.strip()

    # ">=N" boundary
    m = re.match(r'^>=\s*(\d+)$', bucket)
    if m:
        return "WIN" if actual >= int(m.group(1)) else "MISS"

    # "<=N" boundary
    m = re.match(r'^<=\s*(\d+)$', bucket)
    if m:
        return "WIN" if actual <= int(m.group(1)) else "MISS"

    # "N-M" range
    m = re.match(r'^(\d+)-(\d+)$', bucket)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return "WIN" if lo <= actual <= hi else "MISS"

    return "MISS"  # unknown format → conservative

# ── Fetch rows needing scoring ────────────────────────────────────────────────
print("Fetching canonical rows with actual_high but no result_canonical ...")
rows = sb_get(
    "prediction_logs"
    "?ml_bucket_canonical=not.is.null"
    "&ml_actual_high=not.is.null"
    "&ml_result_canonical=is.null"
    "&select=idempotency_key,city,target_date,ml_bucket_canonical,ml_bucket,ml_actual_high,ml_result"
    "&order=city.asc,target_date.asc"
    "&limit=2000"
)
print(f"  Found {len(rows)} rows to score")

if not rows:
    print("Nothing to do.")
    sys.exit(0)

# ── Preview ───────────────────────────────────────────────────────────────────
print(f"\n{'City':<6} {'Date':<12} {'Canonical':<10} {'Final':<10} {'Actual':>6}  {'Score':<6}  {'Already?'}")
print("-" * 70)

to_patch = []
for r in rows:
    actual = int(float(r["ml_actual_high"]))
    canonical = r["ml_bucket_canonical"]
    result_canonical = score_bucket(canonical, actual)

    # For backfilled rows, ml_bucket = canonical (same prediction)
    # Use stored ml_bucket if available, else fall back to canonical
    final_bucket = r.get("ml_bucket") or canonical
    result_latest = r.get("ml_result") or score_bucket(final_bucket, actual)

    already_scored = "✓" if r.get("ml_result") else ""
    icon = "✅" if result_canonical == "WIN" else "❌"
    print(f"{r['city']:<6} {r['target_date']:<12} {canonical:<10} {final_bucket:<10} {actual:>6}  "
          f"{icon} {result_canonical:<4}  {already_scored}")

    to_patch.append({
        "idempotency_key": r["idempotency_key"],
        "ml_result_canonical": result_canonical,
        # Only set ml_result if not already set
        "ml_result": result_latest if not r.get("ml_result") else None,
    })

wins   = sum(1 for p in to_patch if p["ml_result_canonical"] == "WIN")
misses = sum(1 for p in to_patch if p["ml_result_canonical"] == "MISS")
print(f"\nSummary: {wins} WIN, {misses} MISS "
      f"({wins/(wins+misses)*100:.0f}% canonical win rate across backfill)")

if DRY_RUN:
    print(f"\n{'─'*60}")
    print("DRY RUN — no changes written.")
    print("Re-run with --apply to commit.")
    sys.exit(0)

# ── Apply ─────────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("Applying patches ...")
ok = err = 0
for p in to_patch:
    idem = urllib.parse.quote(p["idempotency_key"])
    data = {"ml_result_canonical": p["ml_result_canonical"]}
    if p["ml_result"]:
        data["ml_result"] = p["ml_result"]
    try:
        status = sb_patch("prediction_logs", f"idempotency_key=eq.{idem}", data)
        icon = "✅" if p["ml_result_canonical"] == "WIN" else "❌"
        print(f"  {icon} patched (HTTP {status})")
        ok += 1
    except Exception as e:
        print(f"  ❌ FAILED {p['idempotency_key']}: {e}")
        err += 1

print(f"\nDone. {ok} patched, {err} errors.")
