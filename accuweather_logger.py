# accuweather_logger.py
# Appends AccuWeather forecast rows (D0 today & D1 tomorrow) to accuweather_log.csv
# Columns: timestamp,target_date,forecast_or_actual,forecast_time,predicted_high,
#          forecast_detail,cli_date,actual_high,high_time,bias_corrected_prediction,source

import os, sys, csv, datetime
import requests
import pytz

CSV_PATH = "accuweather_log.csv"  # adjust if your CSV is elsewhere
assert CSV_PATH.endswith("accuweather_log.csv"), "Accu logger must write only to accuweather_log.csv"
print(f"[Accu logger] CSV_PATH={CSV_PATH}", file=sys.stderr)

ACCU_API_KEY = os.environ.get("ACCU_API_KEY")
ACCU_LOCATION_KEY = os.environ.get("ACCU_LOCATION_KEY")  # e.g. 349727
TZ_NAME = os.environ.get("TZ", "America/New_York")

if not ACCU_API_KEY or not ACCU_LOCATION_KEY:
    print("accuweather_logger: Missing ACCU_API_KEY or ACCU_LOCATION_KEY env vars", file=sys.stderr)
    sys.exit(0)  # don't fail the workflow; just skip

tz = pytz.timezone(TZ_NAME)

# ===== MIDNIGHT-4AM FREEZE CHECK =====
# Force America/New_York timezone so skip logic always works
et = pytz.timezone("America/New_York")
current_time_et = datetime.datetime.now(et)
current_hour_et = current_time_et.hour

if 0 <= current_hour_et < 4:
    print(f"[Accu logger] Skipping: {current_time_et.strftime('%Y-%m-%d %H:%M %Z')} (midnight–4am ET freeze period)", file=sys.stderr)
    print("::notice title=AccuWeather Skip::Skipped API pull during midnight–4am ET freeze window")
    sys.exit(0)
# ===== END FREEZE CHECK =====


def now_et():
    return datetime.datetime.now(tz)

def iso_date(dt):
    return dt.strftime("%Y-%m-%d")

def stamp(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def is_after_6pm_local(dt):
    return dt.hour >= 18

def actual_already_logged(csv_path, target_iso):
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("target_date") == target_iso and row.get("forecast_or_actual") == "actual":
                    return True
    except FileNotFoundError:
        return False
    return False

def existing_highs(csv_path, target_iso, source="AccuWeather"):
    """Return set of predicted_high values already logged for this target_date/source."""
    highs = set()
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if (
                    (row.get("target_date") or "").strip() == target_iso
                    and (row.get("forecast_or_actual") or "").lower() == "forecast"
                    and (row.get("source") or "") == source
                ):
                    ph = (row.get("predicted_high") or "").strip()
                    if ph != "":
                        try:
                            highs.add(int(ph))
                        except Exception:
                            pass
    except FileNotFoundError:
        return highs
    return highs

def fetch_accuweather():
    url = f"http://dataservice.accuweather.com/forecasts/v1/daily/5day/{ACCU_LOCATION_KEY}?apikey={ACCU_API_KEY}&details=true&metric=false"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()

def row_for(day_obj, offset):
    target_dt = now_et().date() + datetime.timedelta(days=offset)
    target_str = target_dt.strftime("%Y-%m-%d")
    fc_time = now_et().strftime("%Y-%m-%d %H:%M:%S")
    max_f = day_obj.get("Temperature", {}).get("Maximum", {}).get("Value")
    try:
        max_f = int(round(float(max_f)))
    except Exception:
        max_f = ""
    detail = day_obj.get("Day", {}).get("IconPhrase", "")
    return [
        stamp(now_et()),
        target_str,
        "forecast",
        fc_time,
        max_f,
        detail,
        "", "", "", "",
        "AccuWeather",
    ]

def rows_for_today_and_tomorrow():
    now = now_et()
    daily = fetch_accuweather().get("DailyForecasts", [])
    rows = []

    # --- D0 (today) ---
    today_str = iso_date(now)
    max_f_today = daily[0].get("Temperature", {}).get("Maximum", {}).get("Value")
    try:
        max_f_today = int(round(float(max_f_today)))
    except Exception:
        max_f_today = ""

    already_highs_today = existing_highs(CSV_PATH, today_str, "AccuWeather")

    if (
        not is_after_6pm_local(now)
        and not actual_already_logged(CSV_PATH, today_str)
        and max_f_today != "" 
        and max_f_today not in already_highs_today
    ):
        print(f"Appending new D0 forecast {max_f_today} for {today_str}")
        rows.append(row_for(daily[0], 0))
    else:
        print(f"Skipped D0 {today_str}, max={max_f_today}, already={already_highs_today}")

    # --- D1 (tomorrow) ---
    tomorrow_str = iso_date(now + datetime.timedelta(days=1))
    max_f_tomorrow = daily[1].get("Temperature", {}).get("Maximum", {}).get("Value")
    try:
        max_f_tomorrow = int(round(float(max_f_tomorrow)))
    except Exception:
        max_f_tomorrow = ""

    already_highs_tomorrow = existing_highs(CSV_PATH, tomorrow_str, "AccuWeather")

    if max_f_tomorrow != "" and max_f_tomorrow not in already_highs_tomorrow:
        print(f"Appending new D1 forecast {max_f_tomorrow} for {tomorrow_str}")
        rows.append(row_for(daily[1], 1))
    else:
        print(f"Skipped D1 {tomorrow_str}, max={max_f_tomorrow}, already={already_highs_tomorrow}")

    return rows

def append_rows(csv_path, rows):
    if not rows:
        return
    header = [
        "timestamp","target_date","forecast_or_actual","forecast_time",
        "predicted_high","forecast_detail","cli_date","actual_high",
        "high_time","bias_corrected_prediction","source"
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerows(rows)

def main():
    rows = rows_for_today_and_tomorrow()
    append_rows(CSV_PATH, rows)

if __name__ == "__main__":
    main()
