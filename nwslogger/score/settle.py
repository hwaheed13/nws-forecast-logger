"""Bucket scoring against Kalshi market structure.

Moved VERBATIM from prediction_writer.py in renovation Phase 2 (2026-07-26).
prediction_writer imports these names from here — behavior is frozen by
tests/test_golden_scoring.py. Public aliases without the underscore are the
package-native API; the underscored names remain for legacy call sites.
"""
from __future__ import annotations

import json
from typing import Optional


def _find_kalshi_bucket_for_temp(predicted_temp: float, kalshi_buckets: dict) -> Optional[str]:
    """
    Given a predicted temperature, find which Kalshi bucket contains it.

    Kalshi buckets change daily. Structure is always:
      - "<=X" (lower edge: X and below)
      - "A-B" (range: covers integer temps A through B inclusive)
      - ">=Y" (upper edge: Y and above)

    The predicted_temp is rounded to nearest integer, then we find
    which bucket that integer falls into.
    """
    temp_int = int(round(predicted_temp))

    for label in kalshi_buckets:
        # Upper edge: ">=70" means 70 and above
        if label.startswith(">="):
            try:
                threshold = int(label[2:])
                if temp_int >= threshold:
                    return label
            except ValueError:
                continue

        # Lower edge: "<=47" means 47 and below
        elif label.startswith("<="):
            try:
                threshold = int(label[2:])
                if temp_int <= threshold:
                    return label
            except ValueError:
                continue

        # Standard range: "68-69" means integer temps 68 and 69
        elif "-" in label:
            parts = label.split("-")
            if len(parts) == 2:
                try:
                    lo, hi = int(parts[0]), int(parts[1])
                    if lo <= temp_int <= hi:
                        return label
                except ValueError:
                    continue

    return None


def _bucket_center_temp(ml_bucket: str) -> Optional[float]:
    """Best-effort center temp implied by an ML bucket label.

    Used only when an explicit center (ml_f / ml_f_canonical) isn't passed.
    "86-87" → 86.5 ; "<=47" → 47 ; ">=70" → 70.
    """
    if not ml_bucket:
        return None
    try:
        if "-" in ml_bucket and not ml_bucket.startswith(("<=", ">=")):
            lo, hi = ml_bucket.split("-")
            return (float(lo) + float(hi)) / 2.0
        if ml_bucket.startswith("<="):
            return float(ml_bucket[2:])
        if ml_bucket.startswith(">="):
            return float(ml_bucket[2:])
    except (ValueError, TypeError):
        return None
    return None


def _score_bucket(ml_bucket: str, actual_int: int, kalshi_snapshot_raw,
                  ml_center: Optional[float] = None) -> bool:
    """Return True (WIN) iff the actual high lands in the SAME Kalshi bucket
    the model bet on.

    A Kalshi bet is: "the predicted center temp picks one bucket; you win if
    the settled high falls in that bucket." So scoring must mirror exactly how
    the live dashboard chooses the bucket — `_find_kalshi_bucket_for_temp(ml_f)`
    — and check whether the actual maps to that same bucket.

    The old implementation derived the ML bucket from the label's LOW EDGE,
    added a label-string shortcut, and OR'd two mapping attempts. That could
    return a match even when the predicted center and the actual were many
    degrees apart (e.g. 6/07: center 86.5°F, actual 81°F scored WIN). Using
    the center temp on BOTH sides removes that whole class of false wins.

    `ml_center` (ml_f_canonical or ml_f) is preferred; if absent we fall back
    to the bucket label's implied midpoint.
    """
    center = ml_center if ml_center is not None else _bucket_center_temp(ml_bucket)

    if kalshi_snapshot_raw and center is not None:
        try:
            mkt = (json.loads(kalshi_snapshot_raw)
                   if isinstance(kalshi_snapshot_raw, str) else kalshi_snapshot_raw)
            if mkt:
                actual_kalshi = _find_kalshi_bucket_for_temp(float(actual_int), mkt)
                ml_kalshi = _find_kalshi_bucket_for_temp(float(center), mkt)
                # Both must map into the live market structure; a WIN is strict
                # same-bucket equality. If either fails to map (degenerate or
                # partial snapshot), fall through to the direct range check
                # below rather than risk a spurious match.
                if actual_kalshi is not None and ml_kalshi is not None:
                    return ml_kalshi == actual_kalshi
        except Exception:
            pass

    # Fallback: direct bucket check (no usable Kalshi snapshot).
    # Kalshi "68-69" covers both 68°F and 69°F — inclusive on both ends.
    if ml_bucket.startswith("<="):
        try: return actual_int <= int(ml_bucket[2:])
        except ValueError: return False
    elif ml_bucket.startswith(">="):
        try: return actual_int >= int(ml_bucket[2:])
        except ValueError: return False
    elif "-" in ml_bucket:
        parts = ml_bucket.split("-")
        if len(parts) == 2:
            try:
                lo, hi = int(parts[0]), int(parts[1])
                return lo <= actual_int <= hi
            except ValueError: pass
    return False


# Package-native names (new code should use these).
find_kalshi_bucket_for_temp = _find_kalshi_bucket_for_temp
bucket_center_temp = _bucket_center_temp
score_bucket = _score_bucket
