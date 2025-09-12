# accuweather_logger.py
# Appends AccuWeather forecast rows (D0 today & D1 tomorrow) to nws_forecast_log.csv
# Columns: timestamp,target_date,forecast_or_actual,forecast_time,
#          predicted_high,forecast_detail,cli_date,actual_high,high_time,
#          bias_corrected_prediction,source

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

def iso_date(dt):
    return dt.strftime("%Y-%m-%d")

def stamp(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def is_after_6pm_local(dt):
    return dt.hour >= 18

def existing_highs(csv_path, target_iso, source="AccuWeather"):
    """Return set of predicted_high values already logged for this target_date/source."""
    highs = set()
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            i_target, i_kind, i_pred, i_source = 1, 2, 4, 10
            if header:
                try: i_target = header.index("target_date")
                except ValueError: pass
                try: i_kind = header.index("forecast_or_actual")
                except ValueError: pass
                try: i_pred = header.index("predicted_high")
                except ValueError: pass
                try: i_source = header.index("source")
                except ValueError: pass
            for row in r:
                if not row or len(row) <= max(i_target, i_kind, i_pred, i_source):
                    continue
                if (row[i_target] or "").strip() != target_iso: continue
                if (row[i_kind] or "").strip().lower() != "forecast": continue
                if (row[i_source] or "").strip() != source: continue
                ph = (row[i_pred] or "").strip()
                if ph != "":
                    try: highs.add(int(ph))
                    except Exception: pass
    except FileNotFoundError:
        return highs
    return highs

def actual_already_logged(csv_path, target_iso):
    """Check if an actual high is already logged for target_iso."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if (row.get("target_date") or "").strip() != target_iso:
                    continue
                if (row.get("forecast_or_actual") or "").strip().lower() == "actual":
                    if row.get("actual_high"):
                        return True
    except FileNotFoundError:
        return False
    return False

def fetch_daily_forecasts():
    url = f"http://dataservice.accuweather.com/forecasts/v1/daily/5day/{ACCU_LOCATION_KEY}?apikey={ACCU_API_KEY}&details=true&metric=false"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("DailyForecasts", [])

def row_for(forecast, offset_days):
    target_dt = now_et().date() + datetime.timedelta(days=offset_days)
    target_iso = target_dt.strftime("%Y-%m-%d")
    fc_time = forecast.get("Date", "")
    max_f = forecast.get("Temperature", {}).get("Maximum", {}).get("Value", "")
    phrase = forecast.get("Day", {}).get("IconPhrase", "")
    return [
        stamp(now_et()),          # timestamp
        target_iso,               # target_date
        "forecast",               # forecast_or_actual
        fc_time,                  # forecast_time
        str(max_f),               # predicted_high
        phrase,                   # forecast_detail
        "", "", "", "",           # cli_date, actual_high, high_time, bias_corrected_prediction
        "AccuWeather"             # source
    ]

def rows_for_today_and_tomorrow(daily):
    rows = []
    now = now_et()

    # --- D0 (today): before 6pm, no actual yet, and only if value is new ---
    today_str = iso_date(now)
    max_f_today = daily[0].get("Temperature", {}).get("Maximum", {}).get("Value")
    try:
        max_f_today = int(round(float(max_f_today)))
    except Exception:
        max_f_today = ""
    already_today = existing_highs(CSV_PATH, today_str, "AccuWeather")
    if (not is_after_6pm_local(now)) \
        and (not actual_already_logged(CSV_PATH, today_str)) \
        and (max_f_today != "" and max_f_today not in already_today):
        rows.append(row_for(daily[0], 0))

    # --- D1 (tomorrow): only if value is new ---
    tomorrow_str = iso_date(now + datetime.timedelta(days=1))
    max_f_tomorrow = daily[1].get("Temperature", {}).get("Maximum", {}).get("Value")
    try:
        max_f_tomorrow = int(round(float(max_f_tomorrow)))
    except Exception:
        max_f_tomorrow = ""
    already_tomorrow = existing_highs(CSV_PATH, tomorrow_str, "AccuWeather")
    if max_f_tomorrow != "" and max_f_tomorrow not in already_tomorrow:
        rows.append(row_for(daily[1], 1))

    return rows

def append_rows(csv_path, rows):
    if not rows: return
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "timestamp","target_date","forecast_or_actual","forecast_time",
                "predicted_high","forecast_detail","cli_date","actual_high","high_time",
                "bias_corrected_prediction","source"
            ])
        w.writerows(rows)

def main():
    try:
        daily = fetch_daily_forecasts()
    except Exception as e:
        print("accuweather_logger: fetch failed:", e, file=sys.stderr)
        return
    rows = rows_for_today_and_tomorrow(daily)
    append_rows(CSV_PATH, rows)

if __name__ == "__main__":
    main()
