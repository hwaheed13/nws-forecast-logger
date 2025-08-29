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
    const cutoff = new Date(now.getTime() - 12 * 60 * 60 * 1000);

    const targetTimes = new Set(["01:51", "07:51", "13:51", "19:51"]);
    const rows = [];

    $("table tr").each((i, tr) => {
      const cells = $(tr).find("td");
      if (cells.length < 9) return; // skip malformed rows

      const dayStr  = $(cells[0]).text().trim();
      const timeStr = $(cells[1]).text().trim();
      const maxStr  = $(cells[8]).text().trim();

      if (!targetTimes.has(timeStr)) return; // skip if not a 6-hr marker
      if (!maxStr) return; // skip if blank

      const tempVal = parseFloat(maxStr);
      if (isNaN(tempVal)) return;

      const dt = parseObsTime(dayStr, timeStr, now);
      if (!dt) return;

      if (dt >= cutoff) {
        const fmtTime = new Intl.DateTimeFormat("en-US", {
          timeZone: "America/New_York",
          hour: "numeric",
          minute: "2-digit",
          hour12: true
        }).format(dt);

        rows.push({
          dt,
          value: tempVal,
          time: fmtTime + " ET",
          source: "6hrMax"
        });
      }
    });

    if (!rows.length) {
      return res.status(404).json({ error: "No 6-Hr Max rows found" });
    }

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

  const [hh, mm] = timeStr.split(":").map((x) => parseInt(x, 10));
  if (isNaN(hh) || isNaN(mm)) return null;

  return new Date(nowRef.getFullYear(), nowRef.getMonth(), day, hh, mm);
}
