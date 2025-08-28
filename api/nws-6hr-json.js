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
      if (cells.length >= 8) {
        const timeStr = $(cells[0]).text().trim();   // e.g. "28 Aug 4:51 pm"
        const maxStr  = $(cells[7]).text().trim();   // "Max" column
        const maxVal  = parseFloat(maxStr);

        if (!isNaN(maxVal) && timeStr) {
          const dt = parseObsTime(timeStr, now);
          if (dt) {
            rows.push({ timeStr, dt, maxVal });
          }
        }
      }
    });

    // Only keep rows within the last 6 hours
    const recent = rows.filter(r => r.dt >= cutoff);
    if (!recent.length) {
      return res.status(404).json({ error: "No observations in past 6 hours" });
    }

    // Find the max value and the row it came from
    let maxRow = recent[0];
    for (const r of recent) {
      if (r.maxVal > maxRow.maxVal) {
        maxRow = r;
      }
    }

    res.json({
      value: maxRow.maxVal,
      time: maxRow.timeStr,
      count: recent.length,
      station,
    });
  } catch (err) {
    console.error("Proxy fetch error:", err);
    res.status(500).json({ error: "Proxy fetch error", detail: err.message });
  }
}

/**
 * Parse obhistory time string into a Date.
 * Example: "28 Aug 4:51 pm"
 */
function parseObsTime(str, nowRef) {
  const match = str.match(/(\d{1,2}) (\w{3}) (\d{1,2}):(\d{2})\s*(am|pm)/i);
  if (!match) return null;

  const [ , day, mon, hh, mm, ampm ] = match;
  const monthNames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const month = monthNames.findIndex(m => m.toLowerCase() === mon.toLowerCase());
  if (month === -1) return null;

  let hour = parseInt(hh, 10);
  if (ampm.toLowerCase() === "pm" && hour < 12) hour += 12;
  if (ampm.toLowerCase() === "am" && hour === 12) hour = 0;

  const year = nowRef.getFullYear();
  return new Date(year, month, parseInt(day, 10), hour, parseInt(mm, 10));
}
