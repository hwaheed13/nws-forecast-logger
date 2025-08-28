export default async function handler(req, res) {
  const station = req.query.station || "KNYC";
  const url = `https://www.weather.gov/source/wrh/timeseries/obs.js?site=${station}`;

  try {
    const r = await fetch(url);
    if (!r.ok) {
      res.status(r.status).send(`Upstream error: ${r.statusText}`);
      return;
    }
    const txt = await r.text();

    // Add CORS headers so browser can call this
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET");
    res.setHeader("Content-Type", "application/javascript; charset=utf-8");

    res.send(txt);
  } catch (err) {
    res.status(500).send("Proxy fetch error: " + err.message);
  }
}
