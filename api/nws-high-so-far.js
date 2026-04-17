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
  const tz = req.query.tz || "America/New_York";

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
        "Cache-Control": "no-cache",
        Pragma: "no-cache",
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

    // ── Overnight profile (hours 2-6 ET) for airmass-warmth signal ──
    // Option B overnight-exceedance: use the MINIMUM temp between 2-6am ET
    // as the baseline.  If the overnight min is still close to the forecast
    // high, the airmass is genuinely stuck warm (not just a post-midnight
    // spillover from yesterday that cools off normally).
    let overnightMinF = null, overnightMinTs = null;
    let overnightMaxF = null, overnightMaxTs = null;
    for (const f of feats) {
      const ts = f?.properties?.timestamp;
      if (!ts) continue;
      const d = new Date(ts);
      if (toNYDate(d) !== todayNY) continue;
      const nyHourStr = new Intl.DateTimeFormat("en-US", {
        timeZone: tz, hour: "numeric", hour12: false
      }).format(d);
      const nyHour = Number(nyHourStr);
      if (!Number.isFinite(nyHour) || nyHour < 2 || nyHour > 6) continue;

      const c = f?.properties?.temperature?.value;
      if (c == null || !Number.isFinite(c)) continue;
      const F = c * 9/5 + 32;
      if (overnightMinF == null || F < overnightMinF) { overnightMinF = F; overnightMinTs = ts; }
      if (overnightMaxF == null || F > overnightMaxF) { overnightMaxF = F; overnightMaxTs = ts; }
    }

    res.status(200).json({
      station,
      highF: Math.round(bestF * 10) / 10,
      atISO: bestTs,
      overnightMinF: overnightMinF != null ? Math.round(overnightMinF * 10) / 10 : null,
      overnightMinAt: overnightMinTs,
      overnightMaxF: overnightMaxF != null ? Math.round(overnightMaxF * 10) / 10 : null,
      overnightMaxAt: overnightMaxTs,
    });
  } catch (e) {
    console.error(e);
    res.status(502).json({ error: "Upstream error", details: String(e) });
  }
}
