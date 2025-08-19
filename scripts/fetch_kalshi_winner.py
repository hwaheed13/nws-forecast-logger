#!/usr/bin/env python3
import os, sys, json, urllib.request, urllib.parse, datetime

TZ = datetime.timezone(datetime.timedelta(hours=-4))  # America/New_York (EDT). Adjust to -5 in winter if you want.
now_et = datetime.datetime.now(TZ)

# Before 3pm ET, we fetch yesterday's winner; after 3pm ET, today's.
target = now_et.date()
if now_et.hour < 15:
    target = target - datetime.timedelta(days=1)

date_str = target.isoformat()
event_ticker = f"KXHIGHNY-{date_str}"
base = "https://api.elections.kalshi.com/trade-api/v2"

def get_json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode("utf-8"))

# 1) settlements → find YES
settle_url = f"{base}/settlements?event_ticker={urllib.parse.quote(event_ticker)}"
settle = get_json(settle_url)
records = settle.get("settlements") or settle.get("data") or []
winner = None
for rec in records:
    outcome = str(rec.get("result") or rec.get("outcome") or "").lower()
    if outcome == "yes":
        winner = rec
        break

out = {
  "event_ticker": event_ticker,
  "date": date_str,
  "status": "pending",
  "market_ticker": None,
  "label": None,
  "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
}

if winner:
    m = winner.get("market_ticker") or winner.get("ticker")
    out["market_ticker"] = m
    # 2) markets/<ticker> → get human label
    m_json = get_json(f"{base}/markets/{urllib.parse.quote(m)}")
    market = m_json.get("market") or m_json.get("data") or {}
    raw = market.get("title") or market.get("subtitle") or m or "Winner"
    label = raw.replace("(Yes)", "").strip()
    # strip any leading series/title prefix like "NYC High Temperature: "
    if ": " in label:
        label = label.split(": ", 1)[1]
    out["label"] = label
    out["status"] = "settled"

# Write to repo root (or /public if that’s your pages root)
os.makedirs("public", exist_ok=True)  # change to "" if your Pages root is repo root
with open(os.path.join("public", "kalshi_winner.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
