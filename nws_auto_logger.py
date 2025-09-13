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

# Only used by optional local loop mode
FETCH_TIMES = ["19:30", "21:00", "23:00", "05:00", "06:00", "07:00", "09:00",
               "10:00", "10:50", "11:00", "12:00", "13:00", "14:00", "15:00"]

# =========================
# CSV helpers
# =========================
BASE_HEADER = [
    "timestamp", "target_date", "forecast_or_actual", "forecast_time",
    "predicted_high", "forecast_detail", "cli_date", "actual_high", "high_time"
]
BCP_FIELD = "bias_corrected_prediction"  # persisted field

def ensure_csv_header() -> None:
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            csv.writer(f).writerow(BASE_HEADER + [BCP_FIELD])

def _read_all_rows() -> Tuple[List[dict], List[str]]:
    """Read NWS + AccuWeather rows together."""
    ensure_csv_header()
    rows: List[dict] = []
    fieldnames = BASE_HEADER + [BCP_FIELD]

    for fname in [CSV_FILE, "accuweather_log.csv"]:
        if os.path.exists(fname):
            with open(fname, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for fn in reader.fieldnames:
                        if fn not in fieldnames:
                            fieldnames.append(fn)
                for r in reader:
                    rows.append(r)
    return rows, fieldnames

def _write_all_rows(rows: List[dict], fieldnames: List[str]) -> None:
    with open(CSV_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def _append_row(row: Dict[str, str]) -> None:
    """
    Append using current file header so column count always matches.
    Missing keys will be empty; extra keys are ignored.
    """
    ensure_csv_header()
    with open(CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        fns = reader.fieldnames or (BASE_HEADER + [BCP_FIELD])
    safe_row = {k: row.get(k, "") for k in fns}
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writerow(safe_row)

def _upgrade_header_to_include_bcp(rows: List[dict], fieldnames: List[str]) -> List[str]:
    """
    If the file exists without the BCP column, add it and blank-fill rows.
    """
    if BCP_FIELD in (fieldnames or []):
        return fieldnames
    new_fns = list(fieldnames or BASE_HEADER)
    if BCP_FIELD not in new_fns:
        new_fns.append(BCP_FIELD)
    for r in rows:
        if BCP_FIELD not in r:
            r[BCP_FIELD] = ""
    _write_all_rows(rows, new_fns)
    return new_fns

# =========================
# Forecast helpers
# =========================
def _period_date_local(start_iso: str) -> datetime.date:
    """Convert API period startTime ISO to America/New_York date."""
    dt = datetime.datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    return dt.astimezone(NYC_TZ).date()

def get_forecast_periods() -> List[dict]:
    r = requests.get(NWS_API_ENDPOINT, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    forecast_url = r.json()["properties"]["forecast"]
    r2 = requests.get(forecast_url, headers={"User-Agent": "Mozilla/5.0"})
    r2.raise_for_status()
    return r2.json()["properties"]["periods"]

def pick_today_day_period(periods: List[dict]) -> Optional[dict]:
    """Pick the *daytime* period whose start is today's local date."""
    t = today_nyc()
    for p in periods:
        if p.get("isDaytime") and _period_date_local(p["startTime"]) == t:
            return p
    return None

def pick_tomorrow_day_period(periods: List[dict]) -> Optional[dict]:
    """Pick the *daytime* period whose start is tomorrow's local date."""
    tm = today_nyc() + datetime.timedelta(days=1)
    for p in periods:
        if p.get("isDaytime") and _period_date_local(p["startTime"]) == tm:
            return p
    return None

def _get_last_forecast_row_for_date(target_date: str) -> Optional[dict]:
    """Return the last (most recent) forecast row for target_date, or None."""
    if not os.path.exists(CSV_FILE):
        return None
    last = None
    with open(CSV_FILE, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("forecast_or_actual") == "forecast" and row.get("target_date") == target_date:
                last = row
    return last

def forecast_changed_since_last(target_date: str, new_value: str) -> bool:
    """True if no prior forecast for the date, or the last one differs."""
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

# =========================
# Bias helpers (shared)
# =========================
def _minutes_from_hhmm_ampm(s: str) -> Optional[int]:
    """Parses 'H:MM', 'HH:MM', optionally with ' AM/PM', returns minutes since midnight."""
    if not s:
        return None
    s = s.strip()
    ap = None
    m = re.search(r'\b(AM|PM)\b', s, re.IGNORECASE)
    if m:
        ap = m.group(1).upper()
        s = re.sub(r'\s*(AM|PM)\s*', '', s, flags=re.IGNORECASE)
    m2 = re.match(r'^\s*(\d{1,2}):(\d{2})\s*$', s)
    if not m2:
        return None
    hh, mm = int(m2.group(1)), int(m2.group(2))
    if ap:
        hh = (hh % 12) + (12 if ap == "PM" else 0)
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return hh * 60 + mm

def _minutes_from_forecast_time_cell(s: str) -> Optional[int]:
    """
    CSV 'forecast_time' is 'YYYY-MM-DD HH:MM:SS' (ET).
    Return minutes since midnight (24h) or None.
    """
    if not s or len(s) < 16:
        return None
    try:
        hh = int(s[11:13])
        mm = int(s[14:16])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return hh * 60 + mm
    except Exception:
        return None
    return None

def _float_or_none(x: str) -> Optional[float]:
    try:
        v = float(x)
        if not (v == v):  # NaN
            return None
        return v
    except Exception:
        return None

def _compute_avg_bias_and_today_mean(rows: List[dict], today_iso: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Dashboard-equivalent:
      - For each completed day: bias = actual_high - mean(pre-high forecasts).
      - avgBias = mean of daily biases.
      - todayMean = mean(pre-high forecasts for today) if available.
    """
    by_date: Dict[str, List[dict]] = {}
    for r in rows:
        d = r.get("cli_date") if r.get("forecast_or_actual") == "actual" else r.get("target_date")
        if not d:
            continue
        by_date.setdefault(d, []).append(r)

    biases: List[float] = []
    today_mean: Optional[float] = None

    for d, rs in by_date.items():
        act = next((x for x in rs if x.get("forecast_or_actual") == "actual" and _float_or_none(x.get("actual_high")) is not None), None)
        if not act:
            continue
        actual_high = _float_or_none(act.get("actual_high"))
        high_time = (act.get("high_time") or "").strip()
        if actual_high is None:
            continue

        fc_vals: List[float] = []
        high_min = _minutes_from_hhmm_ampm(high_time) if high_time else None
        for x in rs:
            if x.get("forecast_or_actual") != "forecast":
                continue
            ph = _float_or_none(x.get("predicted_high"))
            if ph is None:
                continue
            if high_min is not None:
                fc_min = _minutes_from_forecast_time_cell(x.get("forecast_time") or "")
                if fc_min is None or fc_min > high_min:
                    continue
            fc_vals.append(ph)

        if fc_vals:
            mean_fc = sum(fc_vals) / len(fc_vals)
            biases.append(actual_high - mean_fc)
            if d == today_iso:
                today_mean = mean_fc

    avg_bias = (sum(biases) / len(biases)) if biases else None
    return avg_bias, today_mean

def _compute_avg_bias_excluding(rows: List[dict], exclude_date_iso: str) -> Optional[float]:
    """Average bias across completed days, excluding exclude_date_iso (pass '' to exclude nothing)."""
    by_date: Dict[str, List[dict]] = {}
    for r in rows:
        d = r.get("cli_date") if r.get("forecast_or_actual") == "actual" else r.get("target_date")
        if not d:
            continue
        by_date.setdefault(d, []).append(r)

    biases: List[float] = []
    for d, rs in by_date.items():
        if exclude_date_iso and d == exclude_date_iso:
            continue
        act = next((x for x in rs if x.get("forecast_or_actual") == "actual" and _float_or_none(x.get("actual_high")) is not None), None)
        if not act:
            continue
        actual_high = _float_or_none(act.get("actual_high"))
        high_time = (act.get("high_time") or "").strip()
        if actual_high is None:
            continue

        fc_vals: List[float] = []
        high_min = _minutes_from_hhmm_ampm(high_time) if high_time else None
        for x in rs:
            if x.get("forecast_or_actual") != "forecast":
                continue
            ph = _float_or_none(x.get("predicted_high"))
            if ph is None:
                continue
            if high_min is not None:
                fc_min = _minutes_from_forecast_time_cell(x.get("forecast_time") or "")
                if fc_min is None or fc_min > high_min:
                    continue
            fc_vals.append(ph)

        if fc_vals:
            mean_fc = sum(fc_vals) / len(fc_vals)
            biases.append(actual_high - mean_fc)

    return (sum(biases) / len(biases)) if biases else None

def _compute_today_pre_high_mean(rows: List[dict], today_iso: str) -> Optional[float]:
    """Mean of today's forecasts that occurred before today's high time."""
    rs = [r for r in rows if (r.get("forecast_or_actual") == "forecast" and r.get("target_date") == today_iso) or
                           (r.get("forecast_or_actual") == "actual"   and r.get("cli_date")    == today_iso)]
    act = next((x for x in rs if x.get("forecast_or_actual") == "actual" and _float_or_none(x.get("actual_high")) is not None), None)
    if not act:
        return None
    high_time = (act.get("high_time") or "").strip()
    high_min = _minutes_from_hhmm_ampm(high_time) if high_time else None

    vals: List[float] = []
    for x in rs:
        if x.get("forecast_or_actual") != "forecast":
            continue
        ph = _float_or_none(x.get("predicted_high"))
        if ph is None:
            continue
        if high_min is not None:
            fc_min = _minutes_from_forecast_time_cell(x.get("forecast_time") or "")
            if fc_min is None or fc_min > high_min:
                continue
        vals.append(ph)

    if not vals:
        return None
    return sum(vals) / len(vals)

# =========================
# Forecast logging
# =========================
def log_forecast() -> None:
    """Capture today's forecast high if it changed and no actual is logged yet."""
    print("🔍 Fetching today’s forecast...")
    periods = get_forecast_periods()
    period = pick_today_day_period(periods)
    if not period:
        print("⚠️ No valid daytime period found for *today*.")
        return

    target_date = today_nyc().isoformat()

    # Freeze forecasts once actual exists
    if actual_exists_for_date(target_date):
        print(f"⏭️ Actual already logged for {target_date}; freezing forecast capture.")
        return

    new_val = str(period.get("temperature"))
    if not forecast_changed_since_last(target_date, new_val):
        print(f"⏭️ Unchanged since last for {target_date}: {new_val}°F")
        return

    now_local = now_nyc().strftime("%Y-%m-%d %H:%M:%S")
    _append_row({
        "timestamp": now_local,
        "target_date": target_date,
        "forecast_or_actual": "forecast",
        "forecast_time": now_local,
        "predicted_high": new_val,
        "forecast_detail": period.get("detailedForecast", ""),
        "cli_date": "",
        "actual_high": "",
        "high_time": "",
        # bias_corrected_prediction intentionally blank for today's rolling rows;
        # we freeze it when the actual posts.
    })
    print(f"✅ Logged forecast for today: {new_val}°F")

def log_forecast_for_tomorrow() -> None:
    """Capture tomorrow's forecast high if it changed (and persist BCP on that row)."""
    print("🔍 Fetching tomorrow’s forecast...")
    periods = get_forecast_periods()
    period = pick_tomorrow_day_period(periods)
    if not period:
        print("⚠️ No valid daytime period found for *tomorrow*.")
        return

    tm = (today_nyc() + datetime.timedelta(days=1)).isoformat()
    new_val = str(period.get("temperature"))

    if not forecast_changed_since_last(tm, new_val):
        print(f"⏭️ Unchanged since last for {tm}: {new_val}°F")
        return

    now_local = now_nyc().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure BCP column exists and compute a persisted BCP for tomorrow using
    # the latest historical average bias (include all completed days).
    rows, fns = _read_all_rows()
    fns = _upgrade_header_to_include_bcp(rows, fns)
    avg_bias_all = _compute_avg_bias_excluding(rows, exclude_date_iso="")  # exclude nothing
    bcp_str = ""
    if avg_bias_all is not None:
        try:
            bcp_val = float(new_val) + float(avg_bias_all)
            bcp_str = f"{bcp_val:.1f}"
        except Exception:
            bcp_str = ""

    _append_row({
        "timestamp": now_local,
        "target_date": tm,
        "forecast_or_actual": "forecast",
        "forecast_time": now_local,
        "predicted_high": new_val,
        "forecast_detail": period.get("detailedForecast", ""),
        "cli_date": "",
        "actual_high": "",
        "high_time": "",
        "bias_corrected_prediction": bcp_str,
    })
    print(f"✅ Logged forecast for {tm}: {new_val}°F (BCP: {bcp_str or 'n/a'})")

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
    Returns { 'TODAY': (temp, time) | None, 'YESTERDAY': (temp, time) | None }
    Handles layouts where 'TODAY'/'YESTERDAY' is one line and 'MAXIMUM ...' on the next.
    """
    current = None
    today_pair: Optional[Tuple[str, str]] = None
    yday_pair: Optional[Tuple[str, str]] = None

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
                t_clean = _normalize_cli_time(tkn_time, ampm)
                if current == "TODAY" and not today_pair:
                    today_pair = (temp, t_clean)
                if current == "YESTERDAY" and not yday_pair:
                    yday_pair = (temp, t_clean)

    return {"TODAY": today_pair, "YESTERDAY": yday_pair}

# =========================
# Actual (provisional + final-upsert)
# =========================
def log_actual_today_if_after_6pm_local() -> None:
    """
    After 6pm ET, log 'TODAY MAXIMUM' from v1 as a provisional actual.
    Also freeze today's BCP snapshot (as-seen pre-high) into the latest forecast row.
    """
    now = now_nyc()
    if now.hour < 18:
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
        pair = sections.get("TODAY")
        if not pair:
            print("⚠️ TODAY MAXIMUM not found; skipping.")
            return

        temp, time_clean = pair
        cli_date = today_nyc().isoformat()

        ensure_csv_header()
        actual_already = False
        with open(CSV_FILE, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("forecast_or_actual") == "actual" and row.get("cli_date") == cli_date:
                    actual_already = True
                    break

        if not actual_already:
            _append_row({
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "target_date": cli_date,
                "forecast_or_actual": "actual",
                "forecast_time": "",
                "predicted_high": "",
                "forecast_detail": "",
                "cli_date": cli_date,
                "actual_high": temp,
                "high_time": time_clean
            })
            print(f"✅ Logged TODAY actual (provisional): {temp}°F at {time_clean} for {cli_date}")
        else:
            print(f"⏭️ Actual for {cli_date} already logged. Skipping append.")

        # --- Freeze today's BCP snapshot into the latest today-forecast row (once) ---
        try:
            rows, fns = _read_all_rows()
            fns = _upgrade_header_to_include_bcp(rows, fns)
            today_iso = cli_date

            # Average bias EXCLUDING today
            avg_bias_excl_today = _compute_avg_bias_excluding(rows, today_iso)
            # Today's mean of pre-high forecasts
            today_pre_mean = _compute_today_pre_high_mean(rows, today_iso)

            if avg_bias_excl_today is not None and today_pre_mean is not None:
                bcp_freeze = today_pre_mean + avg_bias_excl_today
                bcp_str = f"{bcp_freeze:.1f}"

                todays_fc = [r for r in rows
                             if r.get("forecast_or_actual") == "forecast"
                             and r.get("target_date") == today_iso
                             and _float_or_none(r.get("predicted_high")) is not None]
                if todays_fc:
                    todays_fc.sort(key=lambda r: (r.get("timestamp") or r.get("forecast_time") or ""))
                    latest_fc = todays_fc[-1]
                    if not (latest_fc.get(BCP_FIELD) or "").strip():
                        latest_fc[BCP_FIELD] = bcp_str
                        _write_all_rows(rows, fns)
                        print(f"✅ Frozen today's BCP snapshot at actual time: {bcp_str}°F")
                    else:
                        print("⏭️ Today's latest forecast row already has a BCP; leaving as-is.")
            else:
                print("⏭️ Could not compute frozen BCP (missing bias or pre-high mean).")
        except Exception as e:
            print(f"⚠️ Error freezing today's BCP snapshot: {e}")

    except Exception as e:
        print(f"❌ Error logging today's actual high: {e}")

def upsert_actual_row(cli_date_iso: str, temp: str, time_clean: str) -> None:
    """
    Insert or replace an 'actual' row for cli_date_iso (YYYY-MM-DD).
    """
    rows, fns = _read_all_rows()
    now_s = now_nyc().strftime("%Y-%m-%d %H:%M:%S")
    target_date = cli_date_iso

    updated = False
    for r in rows:
        if r.get("forecast_or_actual") == "actual" and r.get("cli_date") == cli_date_iso:
            if r.get("actual_high") != temp or (r.get("high_time") or "") != time_clean:
                r["timestamp"]         = now_s
                r["target_date"]       = target_date
                r["forecast_or_actual"]= "actual"
                r["forecast_time"]     = ""
                r["predicted_high"]    = ""
                r["forecast_detail"]   = ""
                r["cli_date"]          = cli_date_iso
                r["actual_high"]       = temp
                r["high_time"]         = time_clean
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

def upsert_yesterday_actual_if_morning_local() -> None:
    """
    Between midnight–noon ET, prefer the latest CLI (v1) 'YESTERDAY MAXIMUM'.
    """
    now = now_nyc()
    if not (0 <= now.hour < 12):
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
        yday_iso = (now.date() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        upsert_actual_row(yday_iso, temp, time_clean)
        print(f"✅ Upserted YESTERDAY actual: {temp}°F at {time_clean} for {yday_iso}")

    except Exception as e:
        print(f"❌ Error upserting yesterday's actual: {e}")

# =========================
# Bias-corrected snapshot (compat)
# =========================
def write_today_bcp_snapshot_if_after_6pm() -> None:
    """
    Legacy/compat: after 6pm ET, write BCP into today's latest forecast row.
    With the new freeze-in-actual logic above, this will usually just no-op
    (because the BCP will already be set).
    """
    now = now_nyc()
    if now.hour < 18:
        print("⏭️ Skipping BCP snapshot: it’s before 6pm ET")
        return

    rows, fns = _read_all_rows()
    fns = _upgrade_header_to_include_bcp(rows, fns)

    today_iso = today_nyc().isoformat()
    todays_forecasts = [r for r in rows
                        if r.get("forecast_or_actual") == "forecast"
                        and r.get("target_date") == today_iso
                        and _float_or_none(r.get("predicted_high")) is not None]

    if not todays_forecasts:
        print("⏭️ No forecasts for today; skipping BCP snapshot.")
        return

    todays_forecasts.sort(key=lambda r: (r.get("timestamp") or r.get("forecast_time") or ""))
    latest = todays_forecasts[-1]

    if (latest.get(BCP_FIELD) or "").strip():
        print("⏭️ BCP already set on latest forecast row; leaving as-is.")
        return

    avg_bias, today_mean = _compute_avg_bias_and_today_mean(rows, today_iso)
    if avg_bias is None or today_mean is None:
        print("⏭️ Insufficient data to compute BCP; skipping.")
        return

    bcp_value = today_mean + avg_bias
    latest[BCP_FIELD] = f"{bcp_value:.1f}"
    _write_all_rows(rows, fns)
    print(f"✅ Wrote today's bias-corrected prediction: {bcp_value:.1f}°F (into latest forecast row for {today_iso})")

# =========================
# (Optional) Simple loop mode
# =========================
def already_logged(entry_type: str, identifier: str) -> bool:
    """Legacy duplicate check used by local loop mode only."""
    if not os.path.exists(CSV_FILE):
        return False
    with open(CSV_FILE, "r") as f:
        return any(identifier in line and entry_type in line for line in f)

def main_loop() -> None:
    """
    Legacy local loop; generally unnecessary when running via GitHub Actions.
    """
    print("NWS Auto Logger started. Ctrl+C to stop.")
    ensure_csv_header()
    while True:
        n = datetime.datetime.now()
        n_str = n.strftime("%H:%M")
        for sched in FETCH_TIMES:
            if n_str == sched:
                if not already_logged("forecast", n.strftime("%Y-%m-%d %H")):
                    log_forecast()
                log_forecast_for_tomorrow()

        log_actual_today_if_after_6pm_local()        # no-op before 6pm ET
        upsert_yesterday_actual_if_morning_local()   # no-op after noon ET
        write_today_bcp_snapshot_if_after_6pm()      # typically no-op now

        time.sleep(60)

# =========================
# One-shot entrypoint for cron/Actions
# =========================
def run_all_once():
    """
    Safe to run at any time. Each step is individually gated:
      - Forecast captures de-dupe and freeze after actual exists.
      - Provisional actual runs after 6pm ET only.
      - Morning upsert runs midnight–noon ET only.
      - BCP snapshot freezes at actual time; tomorrow rows persist BCP on write.
    """
    ensure_csv_header()
    try:
        log_forecast()
    except Exception as e:
        print(f"⚠️ log_forecast error: {e}")
    try:
        log_forecast_for_tomorrow()
    except Exception as e:
        print(f"⚠️ log_forecast_for_tomorrow error: {e}")
    try:
        log_actual_today_if_after_6pm_local()
    except Exception as e:
        print(f"⚠️ log_actual_today_if_after_6pm_local error: {e}")
    try:
        upsert_yesterday_actual_if_morning_local()
    except Exception as e:
        print(f"⚠️ upsert_yesterday_actual_if_morning_local error: {e}")
    # Compat snapshot; will usually skip since we freeze at actual time
    try:
        write_today_bcp_snapshot_if_after_6pm()
    except Exception as e:
        print(f"⚠️ write_today_bcp_snapshot_if_after_6pm error: {e}")

if __name__ == "__main__":
    run_all_once()
