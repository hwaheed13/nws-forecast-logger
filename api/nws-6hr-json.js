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

    // 🔍 Extract DATA block
    const match = txt.match(/var\s+DATA\s*=\s*(\{[\s\S]*\});/);
    if (!match) {
      console.error("Proxy: Could not locate DATA block");
      res.status(500).send("Proxy could not locate DATA block");
      return;
    }

    const json = match[1];

    // ✅ Serve pure JSON
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.send(json);
  } catch (err) {
    console.error("Proxy fetch error", err);
    res.status(500).send("Proxy fetch error: " + err.message);
  }
}
