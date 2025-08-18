# run_smart.py
import os
from nws_auto_logger import (
    ensure_csv_header,
    log_forecast,
    log_forecast_for_tomorrow,
    log_actual_today_if_after_6pm_local,
    upsert_yesterday_actual_if_morning_local,
)

TASK = (os.getenv("TASK") or "smart_all").strip().lower()

def main():
    ensure_csv_header()

    if TASK == "forecast_today":
        log_forecast()
    elif TASK == "forecast_tomorrow":
        log_forecast_for_tomorrow()
    elif TASK == "actual_provisional":
        # Only logs after 6pm ET; otherwise no-ops
        log_actual_today_if_after_6pm_local()
    elif TASK == "actual_finalize":
        # Only runs between midnight–noon ET; otherwise no-ops
        upsert_yesterday_actual_if_morning_local()
    else:
        # smart_all (default): run everything idempotently
        log_forecast()
        log_forecast_for_tomorrow()
        log_actual_today_if_after_6pm_local()
        upsert_yesterday_actual_if_morning_local()

if __name__ == "__main__":
    main()
