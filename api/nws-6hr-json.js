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

    // Extract the var DATA = { ... };
    const match = txt.match(/var\s+DATA\s*=\s*(\{[\s\S]*?\});/);
    if (!match) {
      res.status(500).send("Could not locate DATA block");
      return;
    }

    // Send just the JSON
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.send(match[1]); // pure JSON now
  } catch (err) {
    res.status(500).send("Proxy fetch error: " + err.message);
  }
}
