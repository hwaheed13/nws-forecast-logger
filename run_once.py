# run_once.py — single-shot execution for manual runs or GitHub Actions
from nws_auto_logger import (
    ensure_csv_header,
    log_forecast,                 # today's forecast (idempotent)
    log_forecast_for_tomorrow,    # tomorrow's forecast (idempotent)
    log_actual_today_if_after_6pm_local,      # only after 6pm ET
    upsert_yesterday_actual_if_morning_local, # only midnight–noon ET
)

if __name__ == "__main__":
    ensure_csv_header()
    log_forecast()
    log_forecast_for_tomorrow()
    log_actual_today_if_after_6pm_local()
    upsert_yesterday_actual_if_morning_local()
