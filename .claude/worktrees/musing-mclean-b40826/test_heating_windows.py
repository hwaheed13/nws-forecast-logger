#!/usr/bin/env python3
"""
Comprehensive test of the heating window implementation.
Verifies that the heating windows match the historical analysis.
"""

from heating_windows import get_heating_window, is_in_heating_window, get_season
from datetime import datetime

def test_heating_windows():
    print("=" * 80)
    print("HEATING WINDOW IMPLEMENTATION VERIFICATION")
    print("=" * 80)

    # Test cases based on historical analysis
    test_cases = [
        # NYC tests (all seasons have 10 AM - 4 PM)
        ("nyc", datetime(2026, 1, 15), 10, True, "NYC winter 10 AM"),
        ("nyc", datetime(2026, 1, 15), 14, True, "NYC winter 2 PM"),
        ("nyc", datetime(2026, 1, 15), 15, True, "NYC winter 3 PM"),
        ("nyc", datetime(2026, 1, 15), 16, False, "NYC winter 4 PM (outside window)"),
        ("nyc", datetime(2026, 1, 15), 9, False, "NYC winter 9 AM (outside window)"),

        ("nyc", datetime(2026, 6, 15), 10, True, "NYC summer 10 AM"),
        ("nyc", datetime(2026, 6, 15), 15, True, "NYC summer 3 PM"),
        ("nyc", datetime(2026, 6, 15), 16, False, "NYC summer 4 PM (outside window)"),

        # LAX tests - seasonal variations
        ("lax", datetime(2026, 1, 15), 10, True, "LAX winter 10 AM"),
        ("lax", datetime(2026, 1, 15), 15, True, "LAX winter 3 PM"),
        ("lax", datetime(2026, 1, 15), 16, False, "LAX winter 4 PM (outside window)"),

        ("lax", datetime(2026, 4, 15), 10, True, "LAX spring 10 AM"),
        ("lax", datetime(2026, 4, 15), 12, True, "LAX spring 12 PM"),
        ("lax", datetime(2026, 4, 15), 13, False, "LAX spring 1 PM (outside window)"),
        ("lax", datetime(2026, 4, 15), 15, False, "LAX spring 3 PM (marine layer)"),

        ("lax", datetime(2026, 7, 15), 10, True, "LAX summer 10 AM"),
        ("lax", datetime(2026, 7, 15), 11, True, "LAX summer 11 AM"),
        ("lax", datetime(2026, 7, 15), 12, False, "LAX summer 12 PM (strong marine cap)"),
        ("lax", datetime(2026, 7, 15), 14, False, "LAX summer 2 PM (strong marine cap)"),

        ("lax", datetime(2026, 10, 15), 10, True, "LAX fall 10 AM"),
        ("lax", datetime(2026, 10, 15), 12, True, "LAX fall 12 PM"),
        ("lax", datetime(2026, 10, 15), 13, False, "LAX fall 1 PM (slight cooling)"),
    ]

    passed = 0
    failed = 0

    for city, date, hour, expected, description in test_cases:
        result = is_in_heating_window(city, hour, date)
        status = "✓" if result == expected else "✗"

        if result == expected:
            passed += 1
        else:
            failed += 1

        window_start, window_end = get_heating_window(city, date)
        season = get_season(date)

        print(f"{status} {description}")
        print(f"   Result: {result}, Expected: {expected}")
        print(f"   City: {city.upper()}, Season: {season}, Window: {window_start:02d}:00-{window_end:02d}:00, Hour: {hour:02d}:00")
        print()

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = test_heating_windows()
    exit(0 if success else 1)
