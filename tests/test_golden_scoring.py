"""Golden tests — bucket scoring behavior (renovation Phase 1).

These freeze CURRENT behavior of the WIN/MISS scoring path before any code
moves. If one of these fails after a refactor, the refactor changed behavior —
that is a bug in the refactor by definition (behavior changes require changing
the golden on purpose, in its own commit, with a reason).
"""
import prediction_writer as pw

# A realistic Kalshi market structure (labels are how Kalshi keys buckets).
MKT = {"<=75": {}, "76-77": {}, "78-79": {}, "80-81": {}, "82-83": {}, ">=84": {}}


class TestFindKalshiBucket:
    def test_range_inclusive_both_ends(self):
        assert pw._find_kalshi_bucket_for_temp(80.0, MKT) == "80-81"
        assert pw._find_kalshi_bucket_for_temp(81.0, MKT) == "80-81"

    def test_rounding_to_nearest_int(self):
        assert pw._find_kalshi_bucket_for_temp(81.4, MKT) == "80-81"
        assert pw._find_kalshi_bucket_for_temp(81.5, MKT) == "82-83"

    def test_edges(self):
        assert pw._find_kalshi_bucket_for_temp(74.0, MKT) == "<=75"
        assert pw._find_kalshi_bucket_for_temp(75.0, MKT) == "<=75"
        assert pw._find_kalshi_bucket_for_temp(84.0, MKT) == ">=84"
        assert pw._find_kalshi_bucket_for_temp(90.0, MKT) == ">=84"


class TestBucketCenterTemp:
    def test_range_label(self):
        assert pw._bucket_center_temp("86-87") == 86.5

    def test_edge_labels(self):
        assert pw._bucket_center_temp("<=47") == 47.0
        assert pw._bucket_center_temp(">=70") == 70.0

    def test_garbage(self):
        assert pw._bucket_center_temp("") is None
        assert pw._bucket_center_temp("not-a-bucket") is None


class TestScoreBucketNoSnapshot:
    """Fallback path: direct range check, inclusive both ends."""

    def test_range_win(self):
        assert pw._score_bucket("80-81", 80, None) is True
        assert pw._score_bucket("80-81", 81, None) is True

    def test_range_miss(self):
        assert pw._score_bucket("80-81", 79, None) is False
        assert pw._score_bucket("80-81", 82, None) is False

    def test_edge_buckets(self):
        assert pw._score_bucket("<=47", 47, None) is True
        assert pw._score_bucket("<=47", 48, None) is False
        assert pw._score_bucket(">=70", 70, None) is True
        assert pw._score_bucket(">=70", 69, None) is False


class TestScoreBucketWithSnapshot:
    """Strict same-Kalshi-bucket scoring (the 2026-06-07 false-WIN fix)."""

    def test_center_and_actual_same_bucket_wins(self):
        assert pw._score_bucket("80-81", 81, MKT, ml_center=80.6) is True

    def test_false_win_case_2026_06_07(self):
        # Center 86.5 (bucket ">=84"), actual 81 (bucket "80-81"): the OLD
        # low-edge/label-shortcut logic could score this a WIN. It is a MISS.
        assert pw._score_bucket("86-87", 81, MKT, ml_center=86.5) is False

    def test_center_maps_to_edge_bucket(self):
        assert pw._score_bucket(">=84", 85, MKT, ml_center=84.2) is True
        assert pw._score_bucket(">=84", 83, MKT, ml_center=84.2) is False

    def test_label_midpoint_fallback_when_no_center(self):
        # bucket_2 path: label is a real Kalshi label, midpoint maps to itself.
        assert pw._score_bucket("82-83", 82, MKT) is True
        assert pw._score_bucket("82-83", 80, MKT) is False

    def test_degenerate_snapshot_falls_back_to_range_check(self):
        assert pw._score_bucket("80-81", 80, {}, ml_center=80.5) is True
        assert pw._score_bucket("80-81", 82, {}, ml_center=80.5) is False
