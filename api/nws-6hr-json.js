// /api/nws-6hr-json.js
import fetch from "node-fetch";
import * as cheerio from "cheerio"; // npm install cheerio

export default async function handler(req, res) {
  const station = req.query.station || "KNYC";
  const url = `https://www.weather.gov/wrh/timeseries?site=${station}`;
  
  try {
    const r = await fetch(url);
    if (!r.ok) {
      res.status(r.status).send(`Upstream error: ${r.statusText}`);
      return;
    }
    const html = await r.text();
    const $ = cheerio.load(html);

    // Find the last row that has a 6 Hr Max value
    let sixHrMax = null;
    let sixHrTime = null;
    $("table tr").each((i, tr) => {
      const cells = $(tr).find("td");
      if (cells.length > 10) {
        const dateTime = $(cells[0]).text().trim();
        const maxVal   = $(cells[11]).text().trim(); // 12th column = 6 Hr Max
        if (/^\d+$/.test(maxVal)) {
          sixHrMax = parseInt(maxVal, 10);
          sixHrTime = dateTime;
        }
      }
    });

    if (!sixHrMax) {
      res.status(404).json({ error: "No 6 Hr Max found" });
      return;
    }

    res.setHeader("Access-Control-Allow-Origin", "*");
    res.json({ value: sixHrMax, time: sixHrTime });
  } catch (err) {
    res.status(500).send("Proxy fetch error: " + err.message);
  }
}
