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
    const targetTimes = new Set(["01:51", "07:51", "13:51", "19:51"]);

    const rows = [];
    $("table tr").each((_, tr) => {
      const cells = $(tr).find("td");
      if (cells.length < 9) return;

      const dayStr = $(cells[0]).text().trim();
      const timeStr = $(cells[1]).text().trim();
      const maxStr = $(cells[8]).text().trim(); // 6-Hr Max

      if (!targetTimes.has(timeStr)) return;
      if (!maxStr) return;

      const tempVal = parseFloat(maxStr);
      if (isNaN(tempVal)) return;

      const dt = parseObsTime(dayStr, timeStr, now);
      if (!dt || dt < cutoff) return;

      const [hh, mm] = timeStr.split(":").map((n) => parseInt(n, 10));
      const fmtTime = formatAsETClock(hh, mm);

      rows.push({ dt, value: tempVal, time: fmtTime, source: "6hrMax" });
    });

    if (!rows.length) return res.status(404).json({ error: "No 6-Hr Max rows found" });

    // ✅ latest
    rows.sort((a, b) => a.dt - b.dt);
    const latest = rows[rows.length - 1];
    res.json(latest);
  } catch (err) {
    console.error("Proxy fetch error:", err);
    res.status(500).json({ error: "Proxy fetch error", detail: err.message });
  }
}

function parseObsTime(dayStr, timeStr, nowRef) {
  const day = parseInt(dayStr, 10);
  if (isNaN(day) || !timeStr) return null;
  const [hh, mm] = timeStr.split(":").map(Number);
  if (isNaN(hh) || isNaN(mm)) return null;
  return new Date(nowRef.getFullYear(), nowRef.getMonth(), day, hh, mm);
}

function formatAsETClock(hh, mm) {
  const hour12 = hh % 12 || 12;
  const ampm = hh < 12 ? "AM" : "PM";
  return `${hour12}:${String(mm).padStart(2, "0")} ${ampm} ET`;
}
