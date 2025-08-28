// api/nws-6hr-json.js
export default async function handler(req, res) {
  const station = req.query.station || "KNYC";
  const url = `https://www.weather.gov/source/wrh/timeseries/obs.js?sid=${station}&interval=ALL`;

  try {
    const r = await fetch(url);
    if (!r.ok) {
      res.status(r.status).send(`Upstream error: ${r.statusText}`);
      return;
    }

    const txt = await r.text();

    const match = txt.match(/var\s+DATA\s*=\s*(\{[\s\S]*?\});/);
    if (!match) {
      res.status(500).send("Proxy could not locate DATA block. Sample:\n" + txt.slice(0, 500));
      return;
    }

    const rawJson = match[1];
    let parsed;
    try {
      parsed = JSON.parse(rawJson);
    } catch (err) {
      res.status(500).send("Found DATA, but JSON.parse failed: " + err.message);
      return;
    }

    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET");
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.json(parsed);
  } catch (err) {
    res.status(500).send("Proxy fetch error: " + err.message);
  }
}
