// api/nws-high-so-far.js
const ALLOW = new Set([
  "https://dailydewpoint.com",
  "http://localhost:3000",
]);

export default async function handler(req, res) {
  // ----- CORS -----
  const origin = req.headers.origin || "";
  const allow = ALLOW.has(origin) ? origin : "https://dailydewpoint.com";
  res.setHeader("Access-Control-Allow-Origin", allow);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "GET,OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    req.headers["access-control-request-headers"] || "Content-Type, Authorization"
  );

  // Strong anti-cache
  res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate");
  res.setHeader("Pragma", "no-cache");
  res.setHeader("CDN-Cache-Control", "no-store");
  if (req.method === "OPTIONS") return res.status(204).end();

  const station = (req.query.station || "KNYC").toUpperCase();
  const tz = "America/New_York";

  const toNYDate = (d) => {
    const parts = new Intl.DateTimeFormat("en-CA", {
      timeZone: tz, year: "numeric", month: "2-digit", day: "2-digit"
    }).formatToParts(d);
    const get = t => parts.find(p => p.type === t)?.value;
    return `${get("year")}-${get("month")}-${get("day")}`;
  };

  const isTopOfHourish = (iso) => {
    const d = new Date(iso);
    const ny = new Date(d.toLocaleString("en-US", { timeZone: tz }));
    const m = ny.getMinutes();
    return (m >= 45 && m <= 59) || (m >= 0 && m <= 6);
  };

  try {
    const url = `https://api.weather.gov/stations/${encodeURIComponent(station)}/observations?limit=200`;
    const r = await fetch(url, {
      headers: {
        Accept: "application/geo+json, application/json",
        "User-Agent": "dailydewpoint (contact: you@example.com)",
        // token: process.env.NWS_API_KEY
      }
    });
    if (!r.ok) return res.status(502).json({ error: "NWS upstream error", status: r.status });

    const j = await r.json();
    const feats = Array.isArray(j?.features) ? j.features : [];
    if (!feats.length) return res.status(204).end();

    const todayNY = toNYDate(new Date());

    let bestF = null, bestTs = null;

    for (const f of feats) {
      const ts = f?.properties?.timestamp;
      if (!ts) continue;
      if (toNYDate(new Date(ts)) !== todayNY) continue;
      if (!isTopOfHourish(ts)) continue;

      const c = f?.properties?.temperature?.value;
      if (c == null || !Number.isFinite(c)) continue;

      const F = c * 9/5 + 32;
      if (bestF == null || F > bestF) { bestF = F; bestTs = ts; }
    }

    if (bestF == null) {
      for (const f of feats) {
        const ts = f?.properties?.timestamp;
        if (!ts) continue;
        if (toNYDate(new Date(ts)) !== todayNY) continue;
        const c = f?.properties?.temperature?.value;
        if (c == null || !Number.isFinite(c)) continue;
        const F = c * 9/5 + 32;
        if (bestF == null || F > bestF) { bestF = F; bestTs = ts; }
      }
    }

    if (bestF == null) return res.status(204).end();

    res.status(200).json({
      station,
      highF: Math.round(bestF * 10) / 10,
      atISO: bestTs,
    });
  } catch (e) {
    console.error(e);
    res.status(502).json({ error: "Upstream error", details: String(e) });
  }
}
