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
        const dayStr  = $(cells[0]).text().trim();   // e.g. "28"
        const timeStr = $(cells[1]).text().trim();   // e.g. "17:51"
        const maxStr  = $(cells[8]).text().trim();   // ✅ 6-hr Max only

        const tempVal = parseFloat(maxStr);
        if (!isNaN(tempVal) && dayStr && timeStr) {
          const dt = parseObsTime(dayStr, timeStr, now);
          if (dt && dt >= cutoff) {
            const fmtTime = new Intl.DateTimeFormat("en-US", {
              timeZone: "America/New_York",
              hour: "numeric",
              minute: "2-digit",
              hour12: true
            }).format(dt);

            rows.push({
              dt,
              value: tempVal,
              timeStr: fmtTime + " ET",
              source: "6hrMax"
            });
          }
        }
      }
    });

    if (!rows.length) {
      return res.status(404).json({ error: "No recent rows found with 6-Hr Max" });
    }

    // ✅ always pick the last non-empty value
    const latest = rows[rows.length - 1];

    res.json({
      value: latest.value,
      time: latest.timeStr,
      source: latest.source,
      count: rows.length,
      station,
    });
  } catch (err) {
    console.error("Proxy fetch error:", err);
    res.status(500).json({ error: "Proxy fetch error", detail: err.message });
  }
}

function parseObsTime(dayStr, timeStr, nowRef) {
  const day = parseInt(dayStr, 10);
  if (isNaN(day)) return null;

  const [hh, mm] = timeStr.split(":").map(x => parseInt(x, 10));
  if (isNaN(hh) || isNaN(mm)) return null;

  const year  = nowRef.getFullYear();
  const month = nowRef.getMonth();
  return new Date(year, month, day, hh, mm);
}
