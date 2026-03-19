from nws_auto_logger import ensure_csv_header, upsert_yesterday_actual_if_morning_local
if __name__ == "__main__":
    ensure_csv_header()
    upsert_yesterday_actual_if_morning_local()
