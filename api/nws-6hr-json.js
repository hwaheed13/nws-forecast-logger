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

    // Try to locate the DATA object in obs.js
    const match = txt.match(/var\s+DATA\s*=\s*(\{[\s\S]*?\});/);
    if (!match) {
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.status(500).json({ error: "Could not locate DATA block" });
      return;
    }

    let data;
    try {
      data = JSON.parse(match[1]);
    } catch (err) {
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.status(500).json({ error: "Parse failed", raw: match[1].slice(0,200) });
      return;
    }

    const obs = data?.STATION?.[0]?.OBSERVATIONS;
    const highs = obs?.air_temp_high_6_hour_set_1 || [];
    const times = obs?.date_time || [];

    let latest = null;
    for (let i = highs.length - 1; i >= 0; i--) {
      if (highs[i] != null) {
        latest = { value: Math.round(highs[i]), time: times[i] || null };
        break;
      }
    }

    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Content-Type", "application/json");
    res.json(latest || { value: null });

  } catch (err) {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.status(500).json({ error: "Proxy fetch error", detail: err.message });
  }
}
