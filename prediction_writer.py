# prediction_writer.py
# -----------------------------------------------------------------------------
# Compute and persist predictions to Supabase (prediction_logs) without
# modifying raw CSV logs. Designed to be called from run_smart.py or as a CLI.
#
# Requires these env vars in your runtime (e.g., GitHub Actions secrets):
#   SUPABASE_URL=https://<project>.supabase.co
#   SUPABASE_SERVICE_ROLE=<service-role-key>   (NOT the anon key)
#
# Table DDL (Supabase SQL):
#   drop table if exists prediction_logs cascade;
#   create table prediction_logs (
#     id bigserial primary key,
#     as_of timestamptz not null,
#     target_date date not null,
#     record_type text not null check (record_type in ('today_for_today','today_for_tomorrow')),
#     bcp_f numeric,
#     nws_latest_f numeric,
#     accu_latest_f numeric,
#     avg_bias_excl_today numeric,
#     today_pre_mean numeric,
#     gate_f numeric,
#     model_version text,
#     notes text,
#     source text default 'nws_auto_logger'
#   );
#   create unique index prediction_logs_unique on prediction_logs (target_date, record_type, as_of);
#   alter table prediction_logs enable row level security;
#   create policy "svc can insert" on prediction_logs for insert to service_role with check (true);
#   create policy "anon can select" on prediction_logs for select to anon using (true);
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import json
import argparse
import urllib.request
from typing import Optional, Tuple

# Import from your existing logger module
from nws_auto_logger import (
    now_nyc,
    today_nyc,
    _read_all_rows,
    _compute_avg_bias_excluding,
    _compute_today_pre_high_mean,
    _float_or_none,
    compute_today_gate_f,
)

MODEL_VERSION = os.environ.get("PREDICTION_MODEL_VERSION", "bcp_v1")


# ----------------------------- Supabase Writer -------------------------------

def _supabase_endpoint() -> Tuple[str, str]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE", "")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE env vars.")
    return f"{url}/rest/v1/prediction_logs", key


def supabase_upsert_prediction(row: dict) -> None:
    """
    Upsert a prediction row into prediction_logs on (target_date, record_type, as_of).
    Uses service-role key. Prints errors verbosely on failure.
    """
    endpoint, key = _supabase_endpoint()
    data = json.dumps(row, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}?on_conflict=target_date,record_type,as_of",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
            "apikey": key,
            "Authorization": f"Bearer {key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            # We don’t need the body; successful POST returns 201/204
            _ = resp.read()
        print("✅ Supabase upsert ok:", {k: row.get(k) for k in ("record_type", "target_date", "as_of", "bcp_f")})
    except Exception as e:
        # If server responds with error body, show it
        if hasattr(e, "read"):
            try:
                body = e.read().decode("utf-8", errors="ignore")
                print(f"❌ Supabase error: {getattr(e, 'code', '?')} {body}")
            except Exception:
                print(f"❌ Supabase error: {e}")
        else:
            print(f"❌ Supabase error: {e}")
        # Do not raise—logging should not crash your workflow


# ------------------------------ Data Helpers ---------------------------------

def _latest_forecast_for_date(rows: list[dict], date_iso: str, source: Optional[str]) -> Optional[float]:
    """
    Returns the latest (max by timestamp/forecast_time ordering) forecasted high for date_iso
    from the specified source:
      source=None   → NWS (i.e., not AccuWeather)
      source='accu' → AccuWeather only
    """
    # Filter to candidate rows
    cands = []
    for r in rows:
        if r.get("forecast_or_actual") != "forecast":
            continue
        if r.get("target_date") != date_iso:
            continue
        if _float_or_none(r.get("predicted_high")) is None:
            continue

        src = (r.get("source") or "").lower()
        if source == "accu":
            if src != "accuweather":
                continue
        else:
            # Treat anything not explicitly 'accuweather' as NWS
            if src == "accuweather":
                continue

        # Use timestamp or forecast_time as a sort key
        key = (r.get("timestamp") or r.get("forecast_time") or "")
        cands.append((key, float(r["predicted_high"])))

    if not cands:
        return None

    # latest by key; if keys are equal, max temp is fine
    cands.sort(key=lambda kv: kv[0])
    return cands[-1][1]


# --------------------------- Public Entry Points -----------------------------

def write_today_for_today(target_date_iso: Optional[str] = None) -> None:
    """
    Compute 'today_for_today' prediction and write to Supabase.
    Should be called after the day's 'actual' is available (e.g., >= 6pm ET).
    """
    if not target_date_iso:
        target_date_iso = today_nyc().isoformat()

    rows, _ = _read_all_rows(include_accu=True)

    avg_bias_excl_today = _compute_avg_bias_excluding(rows, target_date_iso)
    today_pre_mean = _compute_today_pre_high_mean(rows, target_date_iso)

    if avg_bias_excl_today is None or today_pre_mean is None:
        print("⏭️ Not enough data for today_for_today (need avg_bias_excl_today and today_pre_mean).")
        return

    bcp = today_pre_mean + avg_bias_excl_today
    nws_latest = _latest_forecast_for_date(rows, target_date_iso, source=None)
    accu_latest = _latest_forecast_for_date(rows, target_date_iso, source="accu")

    row = {
        "as_of": now_nyc().isoformat(),
        "target_date": target_date_iso,
        "record_type": "today_for_today",
        "bcp_f": float(f"{bcp:.1f}"),
        "nws_latest_f": nws_latest,
        "accu_latest_f": accu_latest,
        "avg_bias_excl_today": avg_bias_excl_today,
        "today_pre_mean": today_pre_mean,
        "gate_f": compute_today_gate_f(),
        "model_version": MODEL_VERSION,
        "notes": "frozen at actual time",
        "source": "nws_auto_logger",
    }
    supabase_upsert_prediction(row)


def write_today_for_tomorrow(tomorrow_iso: Optional[str] = None) -> None:
    """
    Compute 'today_for_tomorrow' prediction and write to Supabase.
    Intended to be called whenever you log/refresh the D+1 forecast today.
    """
    if not tomorrow_iso:
        tomorrow_iso = (today_nyc()).replace(day=today_nyc().day)  # placeholder; better below
        tomorrow_iso = (today_nyc()).isoformat()  # keep it simple if caller doesn't pass
        # We will still recompute below using arithmetic:
    tm_iso = (today_nyc()).isoformat()
    # Correct tm_iso: compute tomorrow from 'today_nyc'
    from datetime import timedelta
    tm_iso = (today_nyc() + timedelta(days=1)).isoformat()

    rows, _ = _read_all_rows(include_accu=True)

    # Average bias across all completed days (exclude nothing)
    avg_bias_all = _compute_avg_bias_excluding(rows, exclude_date_iso="")

    # NWS latest for tomorrow (from NWS rows)
    nws_latest_tm = _latest_forecast_for_date(rows, tm_iso, source=None)
    # Accu latest for tomorrow (optional)
    accu_latest_tm = _latest_forecast_for_date(rows, tm_iso, source="accu")

    bcp_tm = None
    if nws_latest_tm is not None and avg_bias_all is not None:
        bcp_tm = float(f"{(nws_latest_tm + avg_bias_all):.1f}")

    row = {
        "as_of": now_nyc().isoformat(),
        "target_date": tm_iso,
        "record_type": "today_for_tomorrow",
        "bcp_f": bcp_tm,
        "nws_latest_f": nws_latest_tm,
        "accu_latest_f": accu_latest_tm,
        "avg_bias_excl_today": avg_bias_all,
        "today_pre_mean": None,
        "gate_f": None,
        "model_version": MODEL_VERSION,
        "notes": "snapshot from today",
        "source": "nws_auto_logger",
    }
    supabase_upsert_prediction(row)


def write_both_snapshots() -> None:
    """
    Convenience wrapper: write today_for_today (for today) and today_for_tomorrow (for tomorrow).
    Safe to call any time; if inputs aren’t available yet, it will no-op with clear prints.
    """
    try:
        write_today_for_today(today_nyc().isoformat())
    except Exception as e:
        print(f"⚠️ write_today_for_today failed: {e}")
    try:
        write_today_for_tomorrow()
    except Exception as e:
        print(f"⚠️ write_today_for_tomorrow failed: {e}")


# ---------------------------------- CLI --------------------------------------

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Write prediction snapshots to Supabase.")
    sub = p.add_subparsers(dest="cmd", required=True)

    tft = sub.add_parser("today_for_today", help="Write today's prediction for today")
    tft.add_argument("--date", help="Target date (YYYY-MM-DD). Defaults to local today.")

    tftm = sub.add_parser("today_for_tomorrow", help="Write today's prediction for tomorrow")
    tftm.add_argument("--date", help="Tomorrow date (YYYY-MM-DD). Defaults to local tomorrow.")

    both = sub.add_parser("both", help="Write both snapshots (today_for_today and today_for_tomorrow)")
    return p


def main():
    parser = _build_cli()
    args = parser.parse_args()

    if args.cmd == "today_for_today":
        write_today_for_today(args.date)
    elif args.cmd == "today_for_tomorrow":
        # if --date provided, pass it; otherwise function computes tomorrow
        write_today_for_tomorrow(args.date)
    elif args.cmd == "both":
        write_both_snapshots()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
