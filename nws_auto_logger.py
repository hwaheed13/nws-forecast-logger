# nws_auto_logger.py

import csv
import datetime
import os
import re
import time
from typing import Optional, Tuple, Dict, List

import requests
from bs4 import BeautifulSoup
import pytz

# =========================
# Time / Config
# =========================
NYC_TZ = pytz.timezone("America/New_York")

def now_nyc() -> datetime.datetime:
    return datetime.datetime.now(NYC_TZ)

def today_nyc() -> datetime.date:
    return now_nyc().date()

CSV_FILE = "nws_forecast_log.csv"
NWS_API_ENDPOINT = "https://api.weather.gov/points/40.7834,-73.965"

# Optional local-loop schedule (if you ever use main_loop)
FETCH_TIMES = ["19:30", "21:00", "23:00", "05:00", "06:00", "07:00", "09:00",
               "10:00", "10:50", "11:00", "12:00", "13:00", "14:00", "15:00"]

# =========================
# CSV helpers
# =========================
HEADER = [
    "timestamp", "target_date", "forecast_or_actual", "forecast_time",
    "predicted_high", "forecast_detail", "cli_date", "actual_high", "high_time"
]

def ensure_csv_header() -> None:
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            csv.writer(f).writerow(HEADER)

def _read_all_rows() -> Tuple[List[dict], List[str]]:
    """Returns (rows, fieldnames). If file missing, ensures header."""
    ensure_csv_header()
    rows: List[dict] = []
    fieldnames = HEADER
    with open(CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)
    return rows, fieldnames

def _write_all_rows(rows: List[dict], fieldnames: List[str]) -> None:
    with open(CSV_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def upsert_actual_row(cli_date_iso: str, temp: str, time_clean: str) -> None:
    """
    Insert or replace the 'actual' row for cli_date_iso (YYYY-MM-DD).
    If that date already exists as an actual, update it in-place.
    """
    rows, fns = _read_all_rows()
    now_s = now_nyc().strftime("%Y-%m-%d %H:%M:%S")
    target_date = cli_date_iso  # keep aligned with cli_date for actual rows

    updated = False
    for r in rows:
        if r.get("forecast_or_actual") == "actual" and r.get("cli_date") == cli_date_iso:
            if r.get("actual_high") != temp or (r.get("high_time") or "") != time_clean:
                r["timestamp"]    = now_s
                r["target_date"]  = target_date
                r["forecast_or_actual"] = "actual"
                r["forecast_time"] = ""
                r["predicted_high"] = ""
                r["forecast_detail"] = ""
                r["cli_date"]     = cli_date_iso
                r["actual_high"]  = temp
                r["high_time"]    = time_clean
            updated = True
            break

    if not updated:
        rows.append({
            "timestamp": now_s,
            "target_date": target_date,
            "forecast_or_actual": "actual",
            "forecast_time": "",
            "predicted_high": "",
            "forecast_detail": "",
            "cli_date": cli_date_iso,
            "actual_high": temp,
            "high_time": time_clean
        })

    _write_all_rows(rows, fns)

# =========================
# Forecast helpers
# =========================
def _period_date_local(iso_ts: str) -> datetime.date:
    """
    Convert NWS period startTime (ISO8601 with TZ) to **NYC local date**.
    This avoids UTC date drift causing "tomorrow shows up under today".
    """
    dt = datetime.datetime.fromisoformat(iso_ts)
    if dt.tzinfo is None:
        # safety: assume UTC if missing, then convert
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(NYC_TZ).date()

def get_forecast_data() -> list:
    resp = requests.get(NWS_API_ENDPOINT, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    forecast_url = resp.json()["properties"]["forecast"]
    forecast_resp = requests.get(forecast_url, headers={"User-Agent": "Mozilla/5.0"})
    forecast_resp.raise_for_status()
    return forecast_resp.json()["properties"]["periods"]

def get_best_forecast(periods: list, for_tomorrow: bool = False) -> Tuple[Optional[dict], Optional[str]]:
    """
    Pick the daytime period whose **NYC-local date** equals target.
    """
    target = (today_nyc() + datetime.timedelta(days=1 if for_tomorrow else 0))
    for p in periods:
        p_date = _period_date_local(p["startTime"])
        if p_date == target and p.get("isDaytime"):
            return p, p.get("name")
    return None, None

def _get_last_forecast_row_for_date(target_date: str) -> Optional[dict]:
    """Return the most recent forecast row for target_date, or None."""
    if not os.path.exists(CSV_FILE):
        return None
    last = None
    with open(CSV_FILE, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("forecast_or_actual") == "forecast" and row.get("target_date") == target_date:
                last = row
    return last

def forecast_changed_since_last(target_date: str, new_value: str) -> bool:
    """True if no prior forecast for the date, or the last recorded value differs."""
    last = _get_last_forecast_row_for_date(target_date)
    if last is None:
        return True
    return (last.get("predicted_high") or "") != new_value

def actual_exists_for_date(target_date: str) -> bool:
    """True if an actual row for this date already exists (freeze further forecasts)."""
    if not os.path.exists(CSV_FILE):
        return False
    with open(CSV_FILE, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("forecast_or_actual") == "actual" and row.get("cli_date") == target_date:
                return True
    return False

def log_forecast() -> None:
    """
    Log **today's** forecast iff it changed since the last forecast line
    AND no actual exists yet for today.
    """
    print("🔍 Fetching today’s forecast...")
    try:
        periods = get_forecast_data()
    except Exception as e:
        print(f"❌ Error fetching forecast: {e}")
        return

    period, _ = get_best_forecast(periods, for_tomorrow=False)
    if not period:
        print("⚠️ No valid forecast found for today.")
        return

    target_date = today_nyc().isoformat()

    # Freeze forecasts once actual exists
    if actual_exists_for_date(target_date):
        print(f"⏭️ Actual already logged for {target_date}; freezing forecast capture.")
        return

    new_val = str(period["temperature"])
    if not forecast_changed_since_last(target_date, new_val):
        print(f"⏭️ Unchanged since last for {target_date}: {new_val}°F")
        return

    now_local = now_nyc().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, mode="a", newline="") as f:
        csv.writer(f).writerow([
            now_local,                # timestamp (ET)
            target_date,              # target_date (For Date)
            "forecast",
            now_local,                # forecast_time (ET)
            new_val,
            period.get("detailedForecast", ""),
            "", "", ""
        ])
    print(f"✅ Logged forecast for today: {new_val}°F")

def log_forecast_for_tomorrow() -> None:
    """
    Log **tomorrow’s** forecast if it changed vs the last recorded for tomorrow.
    """
    print("🔍 Fetching tomorrow’s forecast...")
    try:
        resp = requests.get(NWS_API_ENDPOINT, headers={"User-Agent": "Mozilla/5.0"}).json()
        forecast_url = resp["properties"]["forecast"]
        forecast = requests.get(forecast_url, headers={"User-Agent": "Mozilla/5.0"}).json()
    except Exception as e:
        print(f"❌ Error fetching forecast: {e}")
        return

    tomorrow = today_nyc() + datetime.timedelta(days=1)
    for period in forecast["properties"]["periods"]:
        p_date = _period_date_local(period["startTime"])
        if p_date == tomorrow and period.get("isDaytime"):
            new_val = str(period["temperature"])
            if forecast_changed_since_last(str(tomorrow), new_val):
                now_local = now_nyc().strftime("%Y-%m-%d %H:%M:%S")
                with open(CSV_FILE, mode="a", newline="") as f:
                    csv.writer(f).writerow([
                        now_local,              # timestamp (ET)
                        str(tomorrow),          # For Date
                        "forecast",
                        now_local,              # forecast_time (ET)
                        new_val,
                        period.get("detailedForecast", ""),
                        "", "", ""
                    ])
                print(f"✅ Logged forecast for {tomorrow}: {new_val}°F")
            else:
                print(f"⏭️ Unchanged since last for {tomorrow}: {new_val}°F")
            break
    else:
        print("⚠️ No forecast found for tomorrow.")

# =========================
# CLI parsing (robust)
# =========================
_TIME_TOKEN = re.compile(r'^\d{3,4}$|^\d{1,2}:\d{2}$', re.ASCII)

def _normalize_cli_time(raw: str, ampm: Optional[str]) -> str:
    """
    Normalize times like '154' -> '1:54', '0154' -> '1:54', '1226' -> '12:26',
    or pass through 'H:MM'. Preserve AM/PM if present.
    """
    raw = (raw or "").strip()
    ap = (ampm or "").strip().upper()
    if ":" in raw:
        hh, mm = raw.split(":", 1)
    else:
        digits = raw.zfill(4)
        hh, mm = digits[:2], digits[2:]
        if hh.startswith("0"):
            hh = hh[1:]
    try:
        hh_i = int(hh)
    except ValueError:
        return raw
    clean = f"{hh_i}:{mm}"
    return f"{clean} {ap}".strip()

def _parse_cli_sections(cli_text: str) -> Dict[str, Optional[Tuple[str, str]]]:
    """
    Parse the CLI <pre> content and return
        { 'TODAY': (temp, time), 'YESTERDAY': (temp, time) }
    Values may be None if not present yet.
    Handles header lines like "TODAY" / "YESTERDAY" followed by a line with "MAXIMUM".
    """
    current = None
    today: Optional[Tuple[str, str]] = None
    yesterday: Optional[Tuple[str, str]] = None

    for raw_line in cli_text.splitlines():
        line_up = raw_line.strip().upper()

        if line_up.startswith("TODAY"):
            current = "TODAY"; continue
        if line_up.startswith("YESTERDAY"):
            current = "YESTERDAY"; continue

        if current and "MAXIMUM" in line_up:
            parts = line_up.split()
            try:
                i = parts.index("MAXIMUM")
            except ValueError:
                continue

            temp = None
            tkn_time = None
            ampm = None

            if i + 1 < len(parts) and parts[i+1].isdigit():
                temp = parts[i+1]
            if i + 2 < len(parts) and _TIME_TOKEN.match(parts[i+2]):
                tkn_time = parts[i+2]
            if i + 3 < len(parts) and parts[i+3] in ("AM", "PM"):
                ampm = parts[i+3]

            if temp and tkn_time:
                time_clean = _normalize_cli_time(tkn_time, ampm)
                if current == "TODAY" and not today:
                    today = (temp, time_clean)
                if current == "YESTERDAY" and not yesterday:
                    yesterday = (temp, time_clean)

    return {"TODAY": today, "YESTERDAY": yesterday}

# =========================
# Actual (provisional + final-upsert)
# =========================
def log_actual_today_if_after_6pm_local() -> None:
    """
    After 6pm ET, log 'TODAY MAXIMUM' from v1 as a provisional actual.
    Morning job can overwrite (replace) yesterday’s row if it changes.
    """
    n = now_nyc()
    if n.hour < 18:
        print("⏭️ Skipping: it’s before 6pm ET")
        return

    try:
        url = ("https://forecast.weather.gov/product.php"
               "?site=NWS&issuedby=NYC&product=CLI&format=CI&version=1&glossary=0")
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
        pre = BeautifulSoup(html, "html.parser").find("pre")
        if not pre:
            print("❌ CLI report not found (v1).")
            return

        sections = _parse_cli_sections(pre.text)
        today_pair = sections.get("TODAY")
        if not today_pair:
            print("⚠️ TODAY MAXIMUM not found; skipping.")
            return

        temp, time_clean = today_pair
        cli_date = today_nyc().isoformat()

        # de-dupe actual for today
        ensure_csv_header()
        with open(CSV_FILE, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("forecast_or_actual") == "actual" and row.get("cli_date") == cli_date:
                    print(f"⏭️ Actual for {cli_date} already logged. Skipping.")
                    return

        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                n.strftime("%Y-%m-%d %H:%M:%S"),
                cli_date, "actual",
                "", "", "",
                cli_date,
                temp,
                time_clean
            ])
        print(f"✅ Logged TODAY actual (provisional): {temp}°F at {time_clean} for {cli_date}")

    except Exception as e:
        print(f"❌ Error logging today's actual high: {e}")

def upsert_yesterday_actual_if_morning_local() -> None:
    """
    Between midnight–noon ET, prefer the latest CLI (v1) 'YESTERDAY MAXIMUM'.
    Upsert (replace-in-place or insert) the row for yesterday so the CSV stays clean.
    """
    n = now_nyc()
    if not (0 <= n.hour < 12):
        print("⏭️ Skipping: it’s afternoon/evening ET")
        return

    try:
        url = ("https://forecast.weather.gov/product.php"
               "?site=NWS&issuedby=NYC&product=CLI&format=CI&version=1&glossary=0")
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
        pre = BeautifulSoup(html, "html.parser").find("pre")
        if not pre:
            print("❌ CLI v1 not available")
            return

        sections = _parse_cli_sections(pre.text)
        yday_pair = sections.get("YESTERDAY")
        if not yday_pair:
            print("⏭️ No YESTERDAY MAXIMUM on v1 yet — will check later.")
            return

        temp, time_clean = yday_pair
        yday_iso = (n.date() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        upsert_actual_row(yday_iso, temp, time_clean)
        print(f"✅ Upserted YESTERDAY actual: {temp}°F at {time_clean} for {yday_iso}")

    except Exception as e:
        print(f"❌ Error upserting yesterday's actual: {e}")

# =========================
# (Optional) Simple loop mode
# =========================
def already_logged(entry_type: str, identifier: str) -> bool:
    """Legacy: substring check used only by the loop mode."""
    if not os.path.exists(CSV_FILE):
        return False
    with open(CSV_FILE, "r") as f:
        return any(identifier in line and entry_type in line for line in f)

def main_loop() -> None:
    """
    Legacy local loop; not used by Actions, but safe to keep.
    Runs once per minute:
      - logs forecasts at your FETCH_TIMES (idempotent, change-aware)
      - provisional today’s actual after 6pm ET (idempotent)
      - finalizes yesterday between midnight–noon ET (upsert)
    """
    print("NWS Auto Logger started. Ctrl+C to stop.")
    ensure_csv_header()
    while True:
        n = datetime.datetime.now()
        n_str = n.strftime("%H:%M")
        for sched in FETCH_TIMES:
            if n_str == sched:
                # Forecasts
                log_forecast()
                log_forecast_for_tomorrow()

        # These contain their own ET gating + de-dupe/upsert logic.
        log_actual_today_if_after_6pm_local()
        upsert_yesterday_actual_if_morning_local()

        time.sleep(60)
