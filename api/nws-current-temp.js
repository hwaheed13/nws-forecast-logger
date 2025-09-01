// api/nws-current-temp.js
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

  // Strong anti-cache (avoid stale at the edge)
  res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate");
  res.setHeader("Pragma", "no-cache");
  res.setHeader("CDN-Cache-Control", "no-store");

  if (req.method === "OPTIONS") return res.status(204).end();

  const station = (req.query.station || "KNYC").toUpperCase(); // Central Park default

  try {
    const url = `https://api.weather.gov/stations/${encodeURIComponent(station)}/observations/latest`;
    const resp = await fetch(url, {
      headers: {
        Accept: "application/geo+json, application/json",
        "User-Agent": "dailydewpoint (contact: you@example.com)",
        // token: process.env.NWS_API_KEY // optional
      }
    });

    if (!resp.ok) {
      return res.status(502).json({ error: "NWS upstream error", status: resp.status });
    }

    const j = await resp.json();
    const p = j?.properties;
    const cVal = p?.temperature?.value; // °C (can be null)

    if (cVal == null || !Number.isFinite(cVal)) {
      return res.status(204).end(); // no current temperature available
    }

    const fVal = cVal * 9/5 + 32;

    return res.status(200).json({
      station,
      currentF: Number(fVal.toFixed(1)),
      atISO: p?.timestamp || null,
    });
  } catch (e) {
    console.error(e);
    return res.status(502).json({ error: "Upstream error", details: String(e) });
  }
}
