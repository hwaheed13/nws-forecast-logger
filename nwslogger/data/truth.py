"""Official ground truth — the ONLY actuals API.

The settle value is the NWS CLI daily high (what Kalshi pays on). Two ways to
get it, consolidated here in renovation Phase 2 (2026-07-26) from
production_report.py and backfill_official_actuals.py:

- load_official_actuals(): the logged CLI actuals from the city's forecast log
  CSV (live path — populated daily by the logger workflows).
- fetch_official_highs(): the IEM CLI archive (backfill path — authoritative
  history for any station/year).

Nothing else in the codebase should define its own notion of "actual high."
The audit found Open-Meteo grid temps masquerading as actuals cost the model
a mean 2.1°F of label error for a year.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request
from pathlib import Path

IEM_CLI_URL = "https://mesonet.agron.iastate.edu/json/cli.py?station={station}&year={year}"


def load_official_actuals(nws_csv: str) -> dict[str, float]:
    """{date_iso: official_high} from the city's forecast-log CSV ('actual' rows)."""
    out: dict[str, float] = {}
    p = Path(nws_csv)
    if not p.exists():
        return out
    with p.open() as f:
        for r in csv.DictReader(f):
            if r.get("forecast_or_actual") == "actual" and r.get("actual_high"):
                d = r.get("cli_date") or r.get("target_date")
                try:
                    out[str(d)] = float(r["actual_high"])
                except (ValueError, TypeError):
                    pass
    return out


def _get(url: str, retries: int = 3) -> dict:
    last_exc = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_exc = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}") from last_exc


def fetch_official_highs(station: str, years: list[int]) -> dict[str, float]:
    """{date_iso: official_high_F} from the IEM CLI archive (settle source)."""
    highs: dict[str, float] = {}
    for year in years:
        data = _get(IEM_CLI_URL.format(station=station, year=year))
        rows = data.get("results", [])
        n = 0
        for r in rows:
            d, h = r.get("valid"), r.get("high")
            if d and h is not None and h != "M":
                try:
                    highs[str(d)] = float(h)
                    n += 1
                except (ValueError, TypeError):
                    pass
        print(f"  {station} {year}: {n} official CLI highs")
        time.sleep(1)  # be polite
    return highs
