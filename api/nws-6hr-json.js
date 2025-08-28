// /api/nws-6hr-json.js
import * as cheerio from "cheerio"; // only dep you need

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

    let sixHrMax = null, sixHrTime = null;

    // Find header index for "6 Hr Max"
    const headers = $("table tr").first().find("th").map((i, th) => $(th).text().trim()).get();
    const sixHrIndex = headers.findIndex(h => h.includes("6 Hr") && h.includes("Max"));

    if (sixHrIndex === -1) {
      res.status(500).json({ error: "Could not locate 6 Hr Max column" });
      return;
    }

    // Walk bottom-up rows for last numeric 6 Hr Max
    const rows = $("table tr").get().reverse();
    for (const tr of rows) {
      const cells = $(tr).find("td");
      if (cells.length > sixHrIndex) {
        const dateTime = $(cells[0]).text().trim();
        const maxVal   = $(cells[sixHrIndex]).text().trim();
        if (/^\d+$/.test(maxVal)) {
          sixHrMax  = parseInt(maxVal, 10);
          sixHrTime = dateTime;
          break;
        }
      }
    }

    if (!sixHrMax) {
      res.status(404).json({ error: "No 6 Hr Max found" });
      return;
    }

    res.setHeader("Access-Control-Allow-Origin", "*");
    res.json({ value: sixHrMax, time: sixHrTime });
  } catch (err) {
    console.error("Proxy crash:", err);
    res.status(500).send("Proxy fetch error: " + err.message);
  }
}
