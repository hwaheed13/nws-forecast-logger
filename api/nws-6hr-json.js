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
    const todayDay = now.getDate();
    const targetTimes = new Set(["01:51", "07:51", "13:51", "19:51"]);
    const results = [];

    $("table tr").each((i, tr) => {
      const cells = $(tr).find("td");
      if (cells.length < 9) return;

      const dayStr  = $(cells[0]).text().trim();
      const timeStr = $(cells[1]).text().trim();
      const maxStr  = $(cells[8]).text().trim();

      if (!targetTimes.has(timeStr)) return;
      if (!maxStr) return;

      const day = parseInt(dayStr, 10);
      if (day !== todayDay) return; // only today’s 6hr maxes

      const tempVal = parseFloat(maxStr);
      if (isNaN(tempVal)) return;

      const dt = parseObsTime(dayStr, timeStr, now);
      if (!dt) return;

      const [hh, mm] = timeStr.split(":").map(n => parseInt(n, 10));
      const fmtTime = formatAsETClock(hh, mm);

      results.push({
        dt,
        value: tempVal,
        time: fmtTime,
        source: "6hrMax"
      });
    });

    if (!results.length) {
      return res.status(404).json({ error: "No 6-Hr Max rows found for today" });
    }

    // Return them sorted chronologically
    results.sort((a, b) => a.dt - b.dt);

    res.json({
      station,
      count: results.length,
      values: results   // 👈 now you get ALL 6-hr maxes today
    });

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

  return new Date(
    nowRef.getFullYear(),
    nowRef.getMonth(),
    day,
    hh,
    mm
  );
}

function formatAsETClock(hh, mm) {
  const hour12 = (hh % 12) || 12;
  const ampm = hh < 12 ? "AM" : "PM";
  return `${hour12}:${String(mm).padStart(2, "0")} ${ampm} ET`;
}
