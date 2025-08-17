from nws_auto_logger import (
    ensure_csv_header,
    log_forecast,                       # today's forecast (only if new)
    log_forecast_for_tomorrow,          # tomorrow's forecast (only if new)
    log_actual_today_if_after_6pm_local,# only runs after 6pm ET
    upsert_yesterday_actual_if_morning_local # only runs midnight–noon ET
)
import os
import sys

"""
USAGE (GitHub Actions or local):
  TASK=smart_all python run_smart.py
  TASK=forecast_today python run_smart.py
  TASK=forecast_tomorrow python run_smart.py
  TASK=actual_if_available python run_smart.py
  TASK=yesterday_actual python run_smart.py
"""

def main():
    ensure_csv_header()
    task = os.environ.get("TASK", "smart_all").strip()

    if task == "forecast_today":
        log_forecast()
    elif task == "forecast_tomorrow":
        log_forecast_for_tomorrow()
    elif task == "actual_if_available":
        log_actual_today_if_after_6pm_local()
    elif task == "yesterday_actual":
        upsert_yesterday_actual_if_morning_local()
    else:
        # smart_all: do everything safely / idempotently
        log_forecast()
        log_forecast_for_tomorrow()
        log_actual_today_if_after_6pm_local()
        upsert_yesterday_actual_if_morning_local()

if __name__ == "__main__":
    main()
