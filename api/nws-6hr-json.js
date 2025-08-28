// /api/nws-6hr-json.js
import fetch from "node-fetch";
import * as cheerio from "cheerio";

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

    // Build headers array
    const headers = [];
    $("table thead tr th").each((i, th) => {
      const hText = $(th).text().replace(/\s+/g, " ").trim();
      headers.push(hText);
    });

    // Find which column is "6 Hr Max"
    const colIndex = headers.findIndex(h => h.toLowerCase().includes("6 hr max"));
    if (colIndex === -1) {
      res.status(404).json({ error: "Could not locate 6 Hr Max column", headers });
      return;
    }

    // Now scan table rows
    let sixHrMax = null;
    let sixHrTime = null;
    $("table tbody tr").each((i, tr) => {
      const cells = $(tr).find("td");
      if (cells.length > colIndex) {
        const dateTime = $(cells[0]).text().trim();
        const val = $(cells[colIndex]).text().trim();
        if (/^\d+$/.test(val)) {
          sixHrMax = parseInt(val, 10);
          sixHrTime = dateTime;
        }
      }
    });

    if (!sixHrMax) {
      res.status(404).json({ error: "No 6 Hr Max found", headers });
      return;
    }

    res.setHeader("Access-Control-Allow-Origin", "*");
    res.json({ value: sixHrMax, time: sixHrTime, columnIndex: colIndex, headers });
  } catch (err) {
    res.status(500).send("Proxy fetch error: " + err.message);
  }
}
