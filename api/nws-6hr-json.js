// /api/nws-6hr-json.js
import * as cheerio from "cheerio";

const ALLOW = new Set([
  "https://dailydewpoint.com",
  "https://app.dailydewpoint.com",
  "http://localhost:3000",
]);

export default async function handler(req, res) {
  // ----- CORS -----
  const origin = req.headers.origin || "";
  const allow = ALLOW.has(origin) ? origin : "https://dailydewpoint.com";
  res.setHeader("Access-Control-Allow-Origin", allow);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "GET,OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    req.headers["access-control-request-headers"] || "Content-Type, Authorization"
  );
  res.setHeader("Cache-Control", "no-store");
  if (req.method === "OPTIONS") return res.status(204).end();

  // ----- Upstream scrape -----
  const station = req.query.station || "KNYC";
  const url = `https://forecast.weather.gov/data/obhistory/${station}.html`;

  try {
    const r = await fetch(url);
    if (!r.ok) {
      return res.status(r.status).json({ error: `Upstream error: ${r.statusText}` });
    }
    const html = await r.text();
    const $ = cheerio.load(html);

    const now = new Date();
    const cutoff = new Date(now.getTime() - 12 * 60 * 60 * 1000); // last 12 hrs
    const targetHours = new Set([1, 7, 13, 19]); // :51 times

    const rows = [];
    $("table tr").each((_, tr) => {
      const cells = $(tr).find("td");
      if (cells.length < 9) return;

      const dayStr  = $(cells[0]).text().trim(); // day-of-month
      const timeStr = $(cells[1]).text().trim(); // e.g., "7:51" or "07:51"
      const maxStr  = $(cells[8]).text().trim(); // "6-Hr Max" column

      if (!timeStr || !maxStr) return;

      // Parse "H:MM" or "HH:MM" without assuming leading zero
      const m = timeStr.match(/^(\d{1,2}):(\d{2})$/);
      if (!m) return;
      const hh = parseInt(m[1], 10);
      const mm = parseInt(m[2], 10);

      // Only accept the four daily snapshots at :51 past the hour
      if (!targetHours.has(hh) || mm !== 51) return;

      const tempVal = parseFloat(maxStr);
      if (Number.isNaN(tempVal)) return;

      const dt = parseObsTime(dayStr, hh, mm, now);
      if (!dt || dt < cutoff) return;

      const fmtTime = formatAsETClock(hh, mm);
      rows.push({ dt, value: tempVal, time: fmtTime, source: "6hrMax" });
    });

    if (!rows.length) return res.status(404).json({ error: "No 6-Hr Max rows found" });

    rows.sort((a, b) => a.dt - b.dt);
    const latest = rows[rows.length - 1];
    res.json(latest);
  } catch (err) {
    console.error("Proxy fetch error:", err);
    res.status(500).json({ error: "Proxy fetch error", detail: err.message });
  }
}

// Build a Date using the current year/month and the day/hour/minute from the page.
function parseObsTime(dayStr, hh, mm, nowRef) {
  const day = parseInt(dayStr, 10);
  if (Number.isNaN(day)) return null;
  const dt = new Date(nowRef.getFullYear(), nowRef.getMonth(), day, hh, mm);

  // Handle month rollover (e.g., page shows last month’s last day on the 1st)
  if (dt - nowRef > 20 * 24 * 60 * 60 * 1000) {
    dt.setMonth(dt.getMonth() - 1);
  }
  return dt;
}

function formatAsETClock(hh, mm) {
  const hour12 = hh % 12 || 12;
  const ampm = hh < 12 ? "AM" : "PM";
  return `${hour12}:${String(mm).padStart(2, "0")} ${ampm} ET`;
}
