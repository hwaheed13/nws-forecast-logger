#!/usr/bin/env python3
"""
backfill_canonical.py
─────────────────────
Retroactively sets ml_bucket_canonical and ml_f_canonical on prediction_logs
rows that have an ML prediction (ml_f, ml_bucket not null) but were written
before ml_bucket_canonical was introduced (pre ~April 7, 2026).

Rules:
  - For each (city, target_date), find the EARLIEST row that has ml_bucket.
  - That row becomes the canonical — set ml_bucket_canonical = ml_bucket,
    ml_f_canonical = ml_f, is_canonical = true.
  - Skip dates that already have a canonical row (don't overwrite).
  - Only process rows where lead_used in ('today_for_today', 'D0').
  - Dry-run by default — pass --apply to actually write.

Usage:
    SUPABASE_URL=... SUPABASE_SERVICE_ROLE=... python3 backfill_canonical.py
    SUPABASE_URL=... SUPABASE_SERVICE_ROLE=... python3 backfill_canonical.py --apply
"""

import os, sys, json, urllib.request, urllib.parse
from collections import defaultdict

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
    req = urllib.request.Request(url, data=body, headers={**HEADERS, "Prefer": "return=minimal"}, method="PATCH")
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status

# ── 1. Fetch all rows with ML predictions, no canonical yet ──────────────────
print("Fetching all ML prediction rows (no canonical) ...")
rows = sb_get(
    "prediction_logs"
    "?ml_f=not.is.null"
    "&ml_bucket=not.is.null"
    "&ml_bucket_canonical=is.null"
    "&lead_used=in.(today_for_today,D0)"
    "&select=idempotency_key,city,target_date,timestamp,ml_f,ml_bucket,ml_confidence,ml_bucket_probs"
    "&order=city.asc,target_date.asc,timestamp.asc"
    "&limit=2000"
)
print(f"  Found {len(rows)} rows without canonical")

# ── 2. Also fetch dates that already have a canonical (to skip) ───────────────
print("Fetching dates that already have canonical ...")
existing_canonical = sb_get(
    "prediction_logs"
    "?ml_bucket_canonical=not.is.null"
    "&select=city,target_date"
    "&limit=2000"
)
already_done = set((r["city"], r["target_date"]) for r in existing_canonical)
print(f"  {len(already_done)} (city, date) pairs already have canonical")

# ── 3. Group by (city, target_date), pick earliest row ───────────────────────
by_date = defaultdict(list)
for r in rows:
    key = (r["city"], r["target_date"])
    if key not in already_done:
        by_date[key].append(r)

# Already sorted by timestamp asc, so first row per key = canonical
to_backfill = []
for key, date_rows in sorted(by_date.items()):
    canonical_row = date_rows[0]  # earliest
    to_backfill.append(canonical_row)

print(f"\n{len(to_backfill)} rows to backfill as canonical:")
print(f"{'City':<6} {'Date':<12} {'Bucket':<10} {'ML°F':<8} {'Confidence':<12} {'Idem Key'}")
print("-" * 80)
for r in to_backfill:
    conf = f"{r['ml_confidence']:.1%}" if r.get('ml_confidence') else "N/A"
    print(f"{r['city']:<6} {r['target_date']:<12} {r['ml_bucket']:<10} {r['ml_f']:<8.1f} {conf:<12} {r['idempotency_key']}")

if DRY_RUN:
    print(f"\n{'─'*60}")
    print("DRY RUN — no changes written.")
    print("Re-run with --apply to commit these changes.")
    sys.exit(0)

# ── 4. Apply patches ──────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("Applying patches ...")
ok = err = 0
for r in to_backfill:
    idem = urllib.parse.quote(r["idempotency_key"])
    data = {
        "ml_bucket_canonical": r["ml_bucket"],
        "ml_f_canonical": r["ml_f"],
        "is_canonical": True,
    }
    try:
        status = sb_patch("prediction_logs", f"idempotency_key=eq.{idem}", data)
        print(f"  ✅ {r['city']} {r['target_date']} → {r['ml_bucket']} (HTTP {status})")
        ok += 1
    except Exception as e:
        print(f"  ❌ {r['city']} {r['target_date']} FAILED: {e}")
        err += 1

print(f"\nDone. {ok} patched, {err} errors.")
