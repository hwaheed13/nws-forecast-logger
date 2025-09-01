// api/nws-dsm.js
const ALLOW = new Set([
  "https://dailydewpoint.com",
  "https://app.dailydewpoint.com",
  "http://localhost:3000",
  "http://localhost:5173",
]);

export default async function handler(req, res) {
  // --- CORS ---
  const origin = req.headers.origin || "";
  const allow = ALLOW.has(origin) ? origin : "https://dailydewpoint.com";
  res.setHeader("Access-Control-Allow-Origin", allow);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "GET,OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    req.headers["access-control-request-headers"] || "Content-Type, Authorization"
  );
  // anti-cache so CORS headers aren’t dropped
  res.setHeader("Cache-Control", "no-store");

  if (req.method === "OPTIONS") return res.status(204).end();

  const issuedby = String(req.query.issuedby || "NYC").toUpperCase();

  try {
    const url = `https://forecast.weather.gov/product.php?site=NWS&issuedby=${encodeURIComponent(
      issuedby
    )}&product=DSM&format=CI&version=1&glossary=1`;

    const r = await fetch(url, {
      headers: {
        Accept: "text/html, text/plain",
        "User-Agent": "dailydewpoint (contact: you@example.com)",
      },
    });
    if (!r.ok) {
      return res.status(502).json({ error: "NWS DSM upstream error", status: r.status });
    }
    const text = await r.text();

    // Frontend parseDSM() expects raw text; serve as text/plain
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    return res.status(200).send(text);
  } catch (e) {
    console.error(e);
    return res.status(502).json({ error: "Upstream error", details: String(e) });
  }
}
