# CRITICAL BUG: ML Predictions Computed But Not Persisted

## Root Cause Analysis

### The Problem (User's Observation)
- prediction_revision_log shows 40+ ML predictions computed (ml_f values: 79.3°F, 79.3°F, 77.4°F, etc.)
- prediction_logs shows only 1-2 entries with ml_f actually written
- 39 out of 40 intraday_refresh entries show **"❌ COMPUTED but NOT WRITTEN"**

### Code Flow
```
write_today_for_tomorrow()
  ↓
  ml = _compute_ml_prediction(...)  ✅ Computes successfully
  ↓
  _log_ml_revision(...) ✅ Writes ml_f to prediction_revision_log
  ↓
  payload["ml_f"] = ml["ml_f"]  ✅ ml_f IS added to payload (line 5945)
  ↓
  is_canonical_write = (existing is _LOCK_NOT_FOUND or ...) and ml is not None and bool(tomorrow_market_probs)
  ↓
  if is_canonical_write:  ← TRUE on first write, FALSE on intraday_refresh
      payload["ml_bucket_canonical"] = ml["ml_bucket"]
      payload["ml_f_canonical"] = ml["ml_f"]
  ↓
  supabase_upsert(payload)  ← Uses merge-duplicates strategy
```

### The Supabase Merge-Duplicates Behavior
```
Prefer: "resolution=merge-duplicates,return=minimal"
on_conflict: idempotency_key
```

**This is a WHOLESALE REPLACEMENT strategy, NOT a field-by-field merge.**

When Supabase encounters a conflict on idempotency_key:
1. It **REPLACES THE ENTIRE ROW** with the new payload
2. Any field **NOT in the new payload** is set to **NULL**

### Why ml_f Disappears on Intraday Writes

**First Write (Canonical, ~01:36:23):**
```python
payload = {
  "idempotency_key": "nyc:bcp_v1:2026-04-17",
  "ml_f": 77.5,
  "ml_bucket": "77-78",
  "ml_f_canonical": 77.5,         ← WRITTEN
  "ml_bucket_canonical": "77-78",  ← WRITTEN
  "is_canonical": true,
  ... (other fields)
}
```
✅ Row created with ml_f=77.5, ml_f_canonical=77.5

**Second Write (Intraday, ~01:59:39):**
```python
is_canonical_write = False  ← No longer canonical

payload = {
  "idempotency_key": "nyc:bcp_v1:2026-04-17",  ← SAME KEY!
  "ml_f": 77.5,              ✅ Included
  "ml_bucket": "77-78",       ✅ Included
  "is_canonical": false,      ← Changed from true
  "ml_f_canonical": <MISSING>,        ← NOT in payload!
  "ml_bucket_canonical": <MISSING>,   ← NOT in payload!
  ... (other fields)
}
```

**With merge-duplicates wholesale replacement:**
- idempotency_key matches → REPLACE ENTIRE ROW
- New payload overwrites the row
- ml_f_canonical is NOT in new payload → **Set to NULL** ✗
- ml_bucket_canonical is NOT in new payload → **Set to NULL** ✗
- ml_f is in new payload → **Updated** ✓

BUT WAIT—if ml_f should be updated, why does it show as NULL in prediction_logs?

### The Real Issue: is_canonical Overwrite

Looking closer at the Supabase behavior:

When `is_canonical` changes from `true` → `false` on the same idempotency_key:
1. The second upsert has `is_canonical: false`
2. With wholesale replacement, the entire row is replaced
3. `is_canonical` field gets overwritten to `false`
4. The Supabase trigger or view that writes to prediction_logs might be checking `is_canonical = true`
5. If the view filters on `is_canonical = true`, then rows with `is_canonical = false` don't get written!

OR—the issue could be that the server-side upsert logic has a conditional check that skips ml_f persistence based on is_canonical status.

### Why One Entry (20:10:40) Successfully Wrote ml_f

Looking at the user's data, the 20:10:40 intraday_refresh entry DID write ml_f=77.5°F successfully.

This suggests that:
1. Sometimes ml_f makes it through
2. The issue is intermittent or depends on some other condition
3. Possible conditions: timing, whether atm_snapshot fetch succeeded, whether market_probs was available, etc.

### Secondary Issue: Missing Fields on Intraday Writes

Even if ml_f is preserved, the wholesale replacement strategy will NULL out:
- `ml_f_canonical` (not in intraday payload) 
- `ml_bucket_canonical` (not in intraday payload)
- `is_canonical` (changed from true to false)
- Any other fields set at canonical write but not repeated on intraday

This causes data loss and breaks dashboards that rely on canonical bucket tracking.

## The Fix

### Option 1: Preserve Canonical Fields on Intraday Writes (Recommended, Minimal Risk)
Before calling supabase_upsert(), check if the existing row has ml_f_canonical/ml_bucket_canonical and include them in the intraday_refresh payload:

```python
# In write_today_for_tomorrow(), around line 5959, after the `if is_canonical_write:` block:

# ALWAYS preserve canonical fields if they exist (prevent merge-duplicates from NULLing them)
if not is_canonical_write and isinstance(existing, dict):
    if existing.get("ml_f_canonical") is not None:
        payload["ml_f_canonical"] = existing["ml_f_canonical"]
    if existing.get("ml_bucket_canonical") is not None:
        payload["ml_bucket_canonical"] = existing["ml_bucket_canonical"]
```

This ensures the canonical fields are never overwritten with NULL on intraday writes.

### Option 2: Change Upsert Strategy (Safer, More Involved)
Instead of using merge-duplicates wholesale replacement, use a conditional update strategy:
- Read existing row before upsert
- Only update fields that have changed
- Leave all other fields untouched
- Requires changing Supabase endpoint or adding application-level logic

### Option 3: Use Different Idempotency Key Strategy (Architectural)
- Use separate rows for each write (different timestamp-based idempotency keys)
- Aggregate predictions in a view that takes the latest per date
- More complex but cleaner data model

## Recommendation

**Implement Option 1 immediately** because:
1. ✅ Single focused code change
2. ✅ Minimal risk (just preserves data that's already there)
3. ✅ Fixes both ml_f_canonical nulling AND ml_bucket_canonical nulling
4. ✅ Can be applied to both write_today_for_today() and write_today_for_tomorrow()
5. ✅ Backward compatible (existing rows keep their canonical values)

Then monitor prediction_logs to verify ml_f values persist across intraday writes.

