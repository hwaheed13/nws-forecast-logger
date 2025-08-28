// /api/nws-6hr-json.js
import * as cheerio from "cheerio";

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");

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
    const cutoff = new Date(now.getTime() - 6 * 60 * 60 * 1000); // 6 hours ago

    const rows = [];
    $("table tr").each((i, tr) => {
      const cells = $(tr).find("td");
      if (cells.length >= 9) {
        const date = $(cells[0]).text().trim();
        const time = $(cells[1]).text().trim();
        const sixHrMaxStr = $(cells[8]).text().trim(); // ✅ always 6 hr Max
        const sixHrMaxVal = parseFloat(sixHrMaxStr);

        if (!isNaN(sixHrMaxVal)) {
          const when = parseObsTime(date, time, now);
          if (when && when >= cutoff) {
            rows.push({ timeStr: `${date} ${time}`, dt: when, maxVal: sixHrMaxVal });
          }
        }
      }
    });

    if (!rows.length) {
      return res.status(404).json({ error: "No 6 Hr Max values found in past 6 hours" });
    }

    // Pick the latest row with a valid 6 hr Max
    const maxRow = rows[rows.length - 1];

    res.json({
      value: maxRow.maxVal,
      time: maxRow.timeStr,
      count: rows.length,
      station,
    });
  } catch (err) {
    console.error("Proxy fetch error:", err);
    res.status(500).json({ error: "Proxy fetch error", detail: err.message });
  }
}

/**
 * Parse obhistory date+time strings into a Date.
 * Example: "28" (day), "18:51" (time).
 */
function parseObsTime(dayStr, hhmm, nowRef) {
  if (!dayStr || !hhmm) return null;

  const [hourStr, minStr] = hhmm.split(":");
  const day = parseInt(dayStr, 10);
  if (isNaN(day)) return null;

  const hour = parseInt(hourStr, 10);
  const min = parseInt(minStr, 10);
  if (isNaN(hour) || isNaN(min)) return null;

  const year = nowRef.getFullYear();
  const month = nowRef.getMonth();
  return new Date(year, month, day, hour, min);
}
