# accuweather_logger.py
# Appends AccuWeather forecast rows (D0 today & D1 tomorrow) to nws_forecast_log.csv
# Columns: timestamp,target_date,forecast_or_actual,forecast_time,predicted_high,forecast_detail,cli_date,actual_high,high_time,bias_corrected_prediction,source

import os, sys, csv, datetime
import requests
import pytz

CSV_PATH = "nws_forecast_log.csv"  # adjust if your CSV is elsewhere

ACCU_API_KEY = os.environ.get("ACCU_API_KEY")
ACCU_LOCATION_KEY = os.environ.get("ACCU_LOCATION_KEY")  # e.g. 349727
TZ_NAME = os.environ.get("TZ", "America/New_York")

if not ACCU_API_KEY or not ACCU_LOCATION_KEY:
    print("accuweather_logger: Missing ACCU_API_KEY or ACCU_LOCATION_KEY env vars", file=sys.stderr)
    sys.exit(0)  # don't fail the workflow; just skip

tz = pytz.timezone(TZ_NAME)

def now_et():
    return datetime.datetime.now(tz)

def stamp(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")  # "YYYY-MM-DD HH:MM:SS"

def iso_date(dt):
    return dt.strftime("%Y-%m-%d")           # date portion only

def is_after_6pm_local(dt=None):
    dt = dt or now_et()
    return (dt.hour, dt.minute, dt.second) > (18, 0, 0)

def actual_already_logged(csv_path, target_iso):
    """True if an 'actual' row exists for the given target_date."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            if header and "target_date" in header and "forecast_or_actual" in header:
                i_target = header.index("target_date")
                i_kind   = header.index("forecast_or_actual")
            else:
                # Fallback to fixed positions in your schema:
                # timestamp(0), target_date(1), forecast_or_actual(2), ...
                i_target, i_kind = 1, 2
            for row in r:
                if len(row) <= max(i_target, i_kind):
                    continue
                if row[i_target] == target_iso and row[i_kind].strip().lower() == "actual":
                    return True
    except FileNotFoundError:
        return False
    except Exception:
        return False
    return False


def clean_detail(text):
    if text is None:
        return ""
    # CSV-safe: leave quoting to csv.writer, just collapse internal newlines
    return " ".join(str(text).split())

def fetch_accuweather_5day(location_key, api_key):
    url = f"https://dataservice.accuweather.com/forecasts/v1/daily/5day/{location_key}"
    params = {"apikey": api_key, "details": "true", "metric": "false"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def rows_for_today_and_tomorrow(j):
    daily = (j or {}).get("DailyForecasts", []) or []
    if len(daily) < 2:
        return []

    now = now_et()
    ts  = stamp(now)

    def row_for(day_obj, offset_days):
        tgt = now + datetime.timedelta(days=offset_days)
        target_iso_str = iso_date(tgt)
        max_f = day_obj.get("Temperature", {}).get("Maximum", {}).get("Value")
        try:
            max_f = int(round(float(max_f)))
        except Exception:
            max_f = ""
        detail = day_obj.get("Day", {}).get("IconPhrase") or day_obj.get("Night", {}).get("IconPhrase") or (j.get("Headline", {}) or {}).get("Text")
        detail = clean_detail(detail)
        return [
            ts,                   # timestamp (pulled)
            target_iso_str,       # target_date
            "forecast",           # forecast_or_actual
            ts,                   # forecast_time
            max_f,                # predicted_high (F)
            detail,               # forecast_detail (short)
            "", "", "", "",       # cli_date, actual_high, high_time, bias_corrected_prediction
            "AccuWeather"         # source
        ]

    rows = []
    # D0 (today): only if it's before 6pm ET AND there is no actual logged yet
    today_str = iso_date(now)
    if (not is_after_6pm_local(now)) and (not actual_already_logged(CSV_PATH, today_str)):
        rows.append(row_for(daily[0], 0))

    # D1 (tomorrow): always log
    rows.append(row_for(daily[1], 1))

    return rows

def append_rows(rows):
    if not rows:
        return
    # Ensure header already has ,source at end (you added this earlier)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)

def main():
    try:
        j = fetch_accuweather_5day(ACCU_LOCATION_KEY, ACCU_API_KEY)
        rows = rows_for_today_and_tomorrow(j)
        append_rows(rows)
        print(f"accuweather_logger: wrote {len(rows)} row(s)")
    except Exception as e:
        # Don't crash the workflow; just log
        print(f"accuweather_logger ERROR: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
