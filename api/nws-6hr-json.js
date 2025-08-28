// api/nws-6hr-json.js

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

    // 🔍 Debug log: see what obs.js starts with
    console.log("First 500 chars of obs.js:", txt.slice(0, 500));

    // Broader regex: match var DATA or var DATA1 / DATA2 etc.
    const match = txt.match(/var\s+DATA\d*\s*=\s*(\{[\s\S]*?\})(;|\n)/);

    if (!match) {
      res.status(500).send(
        "Proxy could not locate DATA block. Sample:\n" + txt.slice(0, 500)
      );
      return;
    }

    // Return just the JSON object (not the whole script)
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET");
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.send(match[1]);
  } catch (err) {
    res.status(500).send("Proxy fetch error: " + err.message);
  }
}
