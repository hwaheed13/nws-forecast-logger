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
    const cutoff = new Date(now.getTime() - 12 * 60 * 60 * 1000); // look back 12 hrs

    const rows = [];
    $("table tr").each((i, tr) => {
      const cells = $(tr).find("td");
      if (cells.length >= 9) {
        const dayStr  = $(cells[0]).text().trim();   // "28"
        const timeStr = $(cells[1]).text().trim();   // "13:51", "19:51"
        const maxStr  = $(cells[8]).text().trim();   // 6-Hr Max column

        // only consider rows at the known valid times
        if (!["01:51","07:51","13:51","19:51"].includes(timeStr)) return;
        if (!maxStr) return;

        const tempVal = parseFloat(maxStr);
        if (!isNaN(tempVal)) {
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
      return res.status(404).json({ error: "No 6-Hr Max rows found" });
    }

    // take the most recent one
    const latest = rows[rows]()
