#!/usr/bin/env python3
"""
Coverage regression guard.

Compares the just-written coverage_report.json against the version
committed on `main` (fetched via `git show origin/main:coverage_report.json`).
Fails (exit 1) if any tracked feature went from non-zero on main to zero
in this run — that's the exact failure mode behind the cli_date bug
that silently dropped v13 entrainment from 1225 → 0.

Also fails if a feature dropped by more than DROP_THRESHOLD (50%) — large
proportional drops without a 100% drop are also suspicious (e.g. a
groupby that started excluding most rows).

Run after training in the workflow. Continue-on-error MUST be false.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DROP_THRESHOLD = 0.50  # fail if a non-zero feature drops by >=50%
PREFIXES = ["", "lax_"]                       # NYC + LAX
MIN_BASELINE_FOR_DROP_CHECK = 20               # ignore drop% on tiny baselines


def _load_main(prefix: str) -> dict:
    fname = f"{prefix}coverage_report.json"
    try:
        out = subprocess.check_output(
            ["git", "show", f"origin/main:{fname}"],
            stderr=subprocess.DEVNULL,
        )
        return json.loads(out)
    except subprocess.CalledProcessError:
        return {}  # first run — nothing to compare against
    except json.JSONDecodeError:
        return {}


def _load_local(prefix: str) -> dict:
    p = Path(f"{prefix}coverage_report.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _check(prefix: str) -> list[str]:
    main_report = _load_main(prefix)
    new_report = _load_local(prefix)
    label = prefix or "nyc"
    failures: list[str] = []

    if not new_report:
        print(f"[{label}] no coverage_report.json — skipping")
        return failures

    for version, counts in new_report.items():
        if version.startswith("_"):
            continue
        prev = main_report.get(version, {})
        for feat, count in counts.items():
            prev_count = prev.get(feat, 0)
            if prev_count > 0 and count == 0:
                failures.append(
                    f"[{label}] {version}.{feat}: {prev_count} → 0 (TOTAL REGRESSION)"
                )
            elif prev_count >= MIN_BASELINE_FOR_DROP_CHECK and count > 0:
                drop = (prev_count - count) / prev_count
                if drop >= DROP_THRESHOLD:
                    failures.append(
                        f"[{label}] {version}.{feat}: {prev_count} → {count} "
                        f"(-{drop:.0%}, threshold {DROP_THRESHOLD:.0%})"
                    )
            print(f"[{label}] {version}.{feat}: {prev_count} → {count}")

    return failures


def main() -> int:
    all_failures: list[str] = []
    for prefix in PREFIXES:
        all_failures.extend(_check(prefix))

    if all_failures:
        print("\n" + "=" * 70)
        print("COVERAGE REGRESSION DETECTED — failing the run")
        print("=" * 70)
        for f in all_failures:
            print(f"  ✗ {f}")
        print(
            "\nThis guard exists because we have repeatedly shipped silent\n"
            "feature-coverage regressions (cli_date vs target_date, missing\n"
            "Supabase columns, etc.). If the drop is intentional, update the\n"
            "previous coverage_report.json on main first or add a bypass\n"
            "label to the workflow.\n"
        )
        return 1

    print("\n✓ Coverage report shows no regressions vs main.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
