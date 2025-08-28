export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*"); // always first

  const station = req.query.station || "KNYC";
  const url = `https://www.weather.gov/wrh/timeseries?site=${station}`;

  try {
    const r = await fetch(url);
    if (!r.ok) return res.status(r.status).send(`Upstream error: ${r.statusText}`);
    const html = await r.text();
    const $ = cheerio.load(html);

    const headers = [];
    $("table thead tr th").each((i, th) => {
      const hText = $(th).text().replace(/\s+/g, " ").trim();
      headers.push(hText);
    });
    console.log("Parsed headers:", headers);

    const colIndex = headers.findIndex(h => h.replace(/\s+/g," ").toLowerCase().includes("6 hr max"));
    if (colIndex === -1) {
      return res.status(404).json({ error: "Could not locate 6 Hr Max column", headers });
    }

    let sixHrMax = null, sixHrTime = null;
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
      return res.status(404).json({ error: "No 6 Hr Max found", headers });
    }

    res.json({ value: sixHrMax, time: sixHrTime, columnIndex: colIndex, headers });
  } catch (err) {
    console.error("Proxy fetch error:", err);
    res.status(500).json({ error: "Proxy fetch error", detail: err.message });
  }
}
