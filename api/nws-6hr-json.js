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
    const r = await fetch(url, {
      headers: {
        "User-Agent": "dailydewpoint (contact: you@example.com)",
        "Cache-Control": "no-cache",
        Pragma: "no-cache",
      },
    });
    if (!r.ok) return res.status(r.status).json({ error: `Upstream error: ${r.statusText}` });

    const html = await r.text();
    const $ = cheerio.load(html);

    const now = new Date();
    const cutoff = new Date(now.getTime() - 12 * 60 * 60 * 1000); // last 12 hrs
    // Support both standard (1,7,13,19) and DST-shifted (0,6,12,18) :51 snapshots
    const targetHours = new Set([0, 1, 6, 7, 12, 13, 18, 19]);

    // Today's date *in ET* — needed for month rollover logic. The server runs
    // in UTC on Vercel, so we cannot use now.getFullYear/getMonth/getDate.
    const etTodayParts = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      year: 'numeric', month: '2-digit', day: '2-digit'
    }).formatToParts(now);
    const getPart = (parts, t) => parseInt(parts.find(p => p.type === t).value, 10);
    const etTodayYear = getPart(etTodayParts, 'year');
    const etTodayMonth = getPart(etTodayParts, 'month') - 1; // 0-indexed
    const etTodayDay = getPart(etTodayParts, 'day');

    const rows = [];
    $("table tr").each((_, tr) => {
      const cells = $(tr).find("td");
      if (cells.length < 15) return; // guard against header/separator rows

      const dayStr  = $(cells[0]).text().trim();   // day-of-month (e.g., "31", "01")
      const timeStr = $(cells[1]).text().trim();   // "7:51" or "07:51"
      const maxStr  = $(cells[8]).text().trim();   // 6-Hr Max column per NWS layout

      if (!timeStr || !maxStr) return;

      // Parse "H:MM" or "HH:MM" (no leading-zero assumption)
      const m = timeStr.match(/^(\d{1,2}):(\d{2})$/);
      if (!m) return;
      const hh = parseInt(m[1], 10);
      const mm = parseInt(m[2], 10);
      if (!targetHours.has(hh) || mm !== 51) return;

      const tempVal = parseFloat(maxStr);
      if (!Number.isFinite(tempVal)) return;

      const dt = parseObsTimeET(dayStr, hh, mm, etTodayYear, etTodayMonth, etTodayDay);
      if (!dt) return;

      // Drop rows older than our window or accidentally in the future
      if (dt < cutoff || dt > now) return;

      // ET calendar date for this observation. en-CA locale gives YYYY-MM-DD.
      const isoDate = new Intl.DateTimeFormat('en-CA', {
        timeZone: 'America/New_York',
        year: 'numeric', month: '2-digit', day: '2-digit'
      }).format(dt);

      rows.push({
        dt,
        date: isoDate,
        value: Math.round(tempVal),
        time: formatAsETClock(hh, mm),
        source: "6hrMax",
      });
    });

    if (!rows.length) return res.json({ value: null, error: "No 6-Hr Max rows found" });

    rows.sort((a, b) => a.dt - b.dt);
    const latest = rows[rows.length - 1];
    res.json(latest);
  } catch (err) {
    console.error("Proxy fetch error:", err);
    res.status(500).json({ error: "Proxy fetch error", detail: err.message });
  }
}

/**
 * Build a Date (UTC instant) from an NWS obs-history row, treating the row's
 * hh:mm as ET wall-clock time. Handles month/year rollover when day-of-month
 * is greater than today's ET day-of-month.
 */
function parseObsTimeET(dayStr, hh, mm, etTodayYear, etTodayMonth, etTodayDay) {
  const day = parseInt(dayStr, 10);
  if (!Number.isFinite(day)) return null;

  let year = etTodayYear;
  let month = etTodayMonth;
  if (day > etTodayDay) {
    month -= 1;
    if (month < 0) { month = 11; year -= 1; }
  }

  // Convert "year-month-day hh:mm in ET wall-clock" to a UTC instant.
  // ET is UTC-5 (EST) or UTC-4 (EDT) depending on date. We compute the
  // correct offset for this specific instant by round-tripping through
  // the Intl ET formatter.
  const asIfUTC = Date.UTC(year, month, day, hh, mm, 0);
  const etParts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false
  }).formatToParts(new Date(asIfUTC));
  const get = (t) => parseInt(etParts.find(p => p.type === t).value, 10);
  const etWallAsUTC = Date.UTC(
    get('year'), get('month') - 1, get('day'),
    get('hour') % 24, get('minute')
  );
  const offsetMs = asIfUTC - etWallAsUTC;

  return new Date(asIfUTC + offsetMs);
}

function formatAsETClock(hh, mm) {
  const hour12 = hh % 12 || 12;
  const ampm = hh < 12 ? "AM" : "PM";
  return `${hour12}:${String(mm).padStart(2, "0")} ${ampm} ET`;
}
