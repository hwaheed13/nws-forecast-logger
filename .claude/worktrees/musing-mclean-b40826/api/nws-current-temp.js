// api/nws-current-temp.js
// Fetches the latest METAR observation for a given station.
//
// Primary source: Aviation Weather Center (aviationweather.gov) — typically
// publishes METARs within 1-2 min of filing, faster than api.weather.gov.
// Fallback: api.weather.gov/stations/{id}/observations/latest (5-15 min lag).
//
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

  const station = (req.query.station || "KNYC").toUpperCase();

  // ── 1. Try Aviation Weather Center first (fastest METAR feed) ───────────────
  try {
    const awcUrl = `https://aviationweather.gov/api/data/metar?ids=${encodeURIComponent(station)}&format=json&taf=false&hours=2`;
    const awcResp = await fetch(awcUrl, {
      headers: {
        Accept: "application/json",
        "User-Agent": "dailydewpoint (contact: hwaheed13@gmail.com)",
      },
      signal: AbortSignal.timeout(6000),
    });

    if (awcResp.ok) {
      const awcData = await awcResp.json();
      // AWC returns an array sorted newest-first
      const latest = Array.isArray(awcData) ? awcData[0] : null;
      // tmpf = temp in °F; temp = temp in °C
      const tempF = latest?.tmpf != null ? Number(latest.tmpf)
                  : latest?.temp  != null ? Number(latest.temp) * 9/5 + 32
                  : null;
      if (tempF != null && Number.isFinite(tempF)) {
        // AWC observation time is in "valid_time_utc" or "observation_time"
        const obsTime = latest?.valid_time_utc || latest?.observation_time || null;
        return res.status(200).json({
          station,
          currentF: Number(tempF.toFixed(1)),
          atISO: obsTime,
          source: "awc",
        });
      }
    }
  } catch (_awcErr) {
    // AWC failed — fall through to NWS API
  }

  // ── 2. Fallback: api.weather.gov ────────────────────────────────────────────
  try {
    const url = `https://api.weather.gov/stations/${encodeURIComponent(station)}/observations/latest`;
    const resp = await fetch(url, {
      headers: {
        Accept: "application/geo+json, application/json",
        "User-Agent": "dailydewpoint (contact: hwaheed13@gmail.com)",
        "Cache-Control": "no-cache",
        Pragma: "no-cache",
      },
      signal: AbortSignal.timeout(8000),
    });

    if (!resp.ok) {
      return res.status(502).json({ error: "NWS upstream error", status: resp.status });
    }

    const j = await resp.json();
    const p = j?.properties;
    const cVal = p?.temperature?.value; // °C (can be null)

    if (cVal == null || !Number.isFinite(cVal)) {
      return res.status(204).end();
    }

    const fVal = cVal * 9/5 + 32;
    return res.status(200).json({
      station,
      currentF: Number(fVal.toFixed(1)),
      atISO: p?.timestamp || null,
      source: "nws",
    });
  } catch (e) {
    console.error(e);
    return res.status(502).json({ error: "Upstream error", details: String(e) });
  }
}
