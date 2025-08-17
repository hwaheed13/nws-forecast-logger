import datetime
import requests
import csv
import os
import time
from bs4 import BeautifulSoup

import pytz

NYC_TZ = pytz.timezone("America/New_York")

def now_nyc():
    return datetime.datetime.now(NYC_TZ)

def today_nyc():
    return now_nyc().date()


CSV_FILE = "nws_forecast_log.csv"
FETCH_TIMES = ["19:30", "21:00", "23:00", "5:00", "06:00", "07:00", "09:00", "10:00", "10:50", "11:00", "12:00", "13:00", "14:00", "15:00"]
NWS_API_ENDPOINT = "https://api.weather.gov/points/40.7834,-73.965"

def ensure_csv_header():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "target_date", "forecast_or_actual", "forecast_time", "predicted_high", "forecast_detail", "cli_date", "actual_high", "high_time"])

def already_logged(entry_type, identifier):
    if not os.path.exists(CSV_FILE):
        return False
    with open(CSV_FILE) as f:
        return any(identifier in line and entry_type in line for line in f)

def get_forecast_data():
    resp = requests.get(NWS_API_ENDPOINT, headers={"User-Agent": "Mozilla/5.0"})
    forecast_url = resp.json()["properties"]["forecast"]
    forecast_resp = requests.get(forecast_url, headers={"User-Agent": "Mozilla/5.0"})
    return forecast_resp.json()["properties"]["periods"]

def get_best_forecast(periods, for_tomorrow=False):
    target_date = (datetime.date.today() + datetime.timedelta(days=1 if for_tomorrow else 0)).isoformat()
    for p in periods:
        if p["startTime"][:10] == target_date and p["isDaytime"]:
            return p, p["name"]
    return None, None

def forecast_already_logged(target_date, predicted_high):
    if not os.path.exists(CSV_FILE):
        return False
    with open(CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["forecast_or_actual"] == "forecast"
                and row["target_date"] == target_date
                and row["predicted_high"] == predicted_high
            ):
                return True
    return False

def log_forecast_for_tomorrow():
    print("🔍 Fetching tomorrow’s forecast...")
    resp = requests.get(NWS_API_ENDPOINT, headers={"User-Agent": "Mozilla/5.0"}).json()
    forecast_url = resp["properties"]["forecast"]
    forecast = requests.get(forecast_url, headers={"User-Agent": "Mozilla/5.0"}).json()

    tomorrow = today_nyc() + datetime.timedelta(days=1)  # <— local tomorrow

    for period in forecast["properties"]["periods"]:
        start_date = datetime.datetime.fromisoformat(period["startTime"]).date()
        if start_date == tomorrow:
            if not forecast_already_logged(str(tomorrow), str(period["temperature"])):
                now_local = now_nyc()
                with open(CSV_FILE, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        now_local.strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
                        str(tomorrow),                            # For Date
                        "forecast",
                        now_local.strftime("%Y-%m-%d %H:%M:%S"),  # forecast_time in ET
                        str(period["temperature"]),
                        period["detailedForecast"],
                        "",
                        "",
                        ""
                    ])
                print(f"✅ Logged forecast for {tomorrow}: {period['temperature']}°F")
            else:
                print(f"⏭️ Forecast for {tomorrow} with {period['temperature']}°F already logged.")
            break
    else:
        print("⚠️ No forecast found for tomorrow.")


def log_forecast():
    print("🔍 Fetching today’s forecast...")
    periods = get_forecast_data()
    period, label = get_best_forecast(periods, for_tomorrow=False)
    if not period:
        print("⚠️ No valid forecast found for today.")
        return

    target_date = today_nyc().isoformat()  # <— local date
    if forecast_already_logged(target_date, str(period["temperature"])):
        print(f"⏭️ Forecast for today with {period['temperature']}°F already logged.")
        return

    now_local = now_nyc()
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            now_local.strftime("%Y-%m-%d %H:%M:%S"),  # timestamp in ET
            target_date,                              # target_date (For Date)
            "forecast",
            now_local.strftime("%Y-%m-%d %H:%M:%S"),  # forecast_time in ET
            str(period["temperature"]),
            period["detailedForecast"],
            "",
            "",
            ""
        ])
    print(f"✅ Logged forecast for today: {period['temperature']}°F")


def log_actual():


    try:
        url = "https://forecast.weather.gov/product.php?site=NWS&issuedby=NYC&product=CLI&format=CI&version=1&glossary=0"
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        pre = soup.find("pre")
        if not pre:
            print("❌ CLI report not found.")
            return

        lines = pre.text.splitlines()
        temp = None
        time_clean = None
        now = now_nyc()
        cli_date = now.date().isoformat()

        for line in lines:
            if "MAXIMUM" in line and "YESTERDAY" not in line.upper():
                parts = line.split()
                try:
                    idx = parts.index("MAXIMUM")
                    temp = parts[idx + 1]
                    time_raw = parts[idx + 2]
                    am_pm = parts[idx + 3] if idx + 3 < len(parts) else ""

                    if len(time_raw) == 3:
                        time_clean = f"{int(time_raw[0])}:{time_raw[1:]} {am_pm}"
                    else:
                        time_clean = f"{int(time_raw[:2])}:{time_raw[2:]} {am_pm}"
                    break
                except Exception:
                    continue

        if not temp or not time_clean:
            print("⚠️ MAXIMUM temperature not found in today’s section.")
            return

        # Check if already logged
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["forecast_or_actual"] == "actual" and row["cli_date"] == cli_date:
                        print(f"⏭️ Actual for {cli_date} already logged. Skipping.")
                        return

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime("%Y-%m-%d %H:%M:%S"),
                cli_date,
                "actual",
                "", "", "",
                cli_date,
                temp,
                time_clean
            ])
        print(f"✅ Logged actual high: {temp}°F at {time_clean} for {cli_date}")

    except Exception as e:
        print(f"❌ Error logging actual high: {e}")




def log_yesterday_actual():
    try:
        url = "https://forecast.weather.gov/product.php?site=NWS&issuedby=NYC&product=CLI&format=CI&version=2&glossary=0"
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        pre = soup.find("pre")
        if not pre:
            print("❌ CLI report not found.")
            return

        lines = pre.text.splitlines()
        for line in lines:
            if "MAXIMUM" in line:
                parts = line.split()
                if "MAXIMUM" in parts:
                    idx = parts.index("MAXIMUM")
                    if idx + 2 < len(parts):
                        temp = parts[idx + 1]
                        time_raw = parts[idx + 2]
                        am_pm = parts[idx + 3] if idx + 3 < len(parts) else ""
                        if len(time_raw) == 3:
                            time_clean = f"{int(time_raw[0])}:{time_raw[1:]} {am_pm}"
                        else:
                            time_clean = f"{int(time_raw[:2])}:{time_raw[2:]} {am_pm}"

                        now = now_nyc()
                        target_date = now.date() - datetime.timedelta(days=1)
                        cli_date = target_date.strftime("%Y-%m-%d")

                        with open(CSV_FILE, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                now.strftime("%Y-%m-%d %H:%M:%S"),
                                target_date.strftime("%Y-%m-%d"),
                                "actual",
                                "", "", "",
                                cli_date,
                                temp,
                                time_clean
                            ])
                        print(f"✅ Logged yesterday's actual high: {temp}°F at {time_clean}")
                        return

        print("⚠️ MAXIMUM temperature not found in CLI report.")
    except Exception as e:
        print(f"❌ Error logging yesterday's actual high: {e}")

def main_loop():
    print("NWS Auto Logger started. Ctrl+C to stop.")
    ensure_csv_header()
    while True:
        now = datetime.datetime.now()
        now_str = now.strftime("%H:%M")
        for sched in FETCH_TIMES:
            if now_str == sched:
                if not already_logged("forecast", now.strftime("%Y-%m-%d %H")):
                    log_forecast()
                log_forecast_for_tomorrow()
        now = now_nyc()
        now_str = now.strftime("%H:%M")
        if now.hour == 18 and not already_logged("actual", now.strftime("%Y-%m-%d")):
            log_actual()
        time.sleep(60)

def log_actual_for_date_via_version(target_date_iso: str, version: int, force: bool = False):
    """
    Backfill an 'actual' high for a specific date by fetching a specific
    NWS CLI report version. Example:
        log_actual_for_date_via_version("2025-08-10", 10)
    """
    try:
        url = (
            "https://forecast.weather.gov/product.php"
            "?site=NWS&issuedby=NYC&product=CLI&format=CI"
            f"&version={version}&glossary=0"
        )
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")
        pre = soup.find("pre")
        if not pre:
            print(f"CLI v{version} not found or malformed.")
            return

        # Parse the first "MAXIMUM" that is NOT explicitly labeled YESTERDAY
        # (older versions are anchored to their original issuance day,
        # so the first non-YESTERDAY MAXIMUM is the day you want).
        lines = pre.text.splitlines()
        temp = None
        time_clean = None
        for line in lines:
            if "MAXIMUM" in line and "YESTERDAY" not in line.upper():
                parts = line.split()
                try:
                    idx = parts.index("MAXIMUM")
                    temp = parts[idx + 1]
                    time_raw = parts[idx + 2]
                    am_pm = parts[idx + 3] if idx + 3 < len(parts) else ""
                    if len(time_raw) == 3:
                        time_clean = f"{int(time_raw[0])}:{time_raw[1:]} {am_pm}"
                    else:
                        time_clean = f"{int(time_raw[:2])}:{time_raw[2:]} {am_pm}"
                    break
                except Exception:
                    continue

        if not temp or not time_clean:
            print(f"MAXIMUM not found in CLI v{version}.")
            return

        # De-dupe unless force=True
        if os.path.exists(CSV_FILE) and not force:
            with open(CSV_FILE, newline="") as f:
                for row in csv.DictReader(f):
                    if row["forecast_or_actual"] == "actual" and row["cli_date"] == target_date_iso:
                        print(f"Actual for {target_date_iso} already logged. Use force=True to override.")
                        return

        now = datetime.datetime.now()
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
                target_date_iso,                    # target_date
                "actual",                           # forecast_or_actual
                "", "", "",                         # forecast_time, predicted_high, forecast_detail
                target_date_iso,                    # cli_date
                temp,                               # actual_high
                time_clean                          # high_time
            ])
        print(f"Backfilled actual high for {target_date_iso} from CLI v{version}: {temp}°F at {time_clean}")

    except Exception as e:
        print(f"Error backfilling {target_date_iso} via CLI v{version}: {e}")

import pytz

def log_actual_today_if_after_6pm_local():
    """Log today's actual high, but only after 6pm ET."""
    nyc = pytz.timezone("America/New_York")
    now = datetime.datetime.now(nyc)
    if now.hour >= 18:
        print("🕕 After 6pm ET — attempting to log today’s actual high")
        log_actual()
    else:
        print("⏭️ Skipping: it’s before 6pm ET")

def upsert_yesterday_actual_if_morning_local():
    """Log yesterday’s actual high, but only if it’s morning (midnight–noon ET)."""
    nyc = pytz.timezone("America/New_York")
    now = datetime.datetime.now(nyc)
    if 0 <= now.hour < 12:
        print("🌅 Morning ET — attempting to log yesterday’s actual high")
        log_yesterday_actual()
    else:
        print("⏭️ Skipping: it’s afternoon/evening ET")


# ========== ✅ MANUAL TRIGGERS (uncomment as needed) ==========

# log_forecast()              # Log today's forecast
# log_forecast_for_tomorrow() # Log tomorrow’s forecast
# log_actual()                # Log actual high (today or yesterday depending on time)
# log_yesterday_actual()      # Force actual high for yesterday using CLI v2

# main_loop()                 # Auto-logger loop (scheduled mode)

# --- one-time backfills ---
# log_actual_for_date_via_version("2025-08-02", 27)
# log_actual_for_date_via_version("2025-08-12", 7)

# main_loop()  # re-enable after backfills if you had commented it out

