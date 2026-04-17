from nws_auto_logger import ensure_csv_header, log_forecast_for_tomorrow
if __name__ == "__main__":
    ensure_csv_header()
    log_forecast_for_tomorrow()  # idempotent

