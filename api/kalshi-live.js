// api/kalshi-live.js
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
  res.setHeader("Cache-Control", "no-store");
  if (req.method === "OPTIONS") return res.status(204).end();
  
  // ----- Input -----
  const { date } = req.query || {};
  if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    return res.status(400).json({ error: "Missing or bad ?date=YYYY-MM-DD" });
  }
  
  const base = "https://api.elections.kalshi.com/trade-api/v2";
  const eventTicker = toKalshiEventTicker(date);
  
  try {
    // Get all markets for the event (open + others)
    const r = await fetch(
      `${base}/markets?event_ticker=${encodeURIComponent(eventTicker)}`,
      { headers: { Accept: "application/json" } }
    );
    if (!r.ok) return res.status(502).json({ error: "Upstream error" });
    
    const j = await r.json();
    const markets = Array.isArray(j?.markets) ? j.markets : [];
    
    // Prefer open/trading markets; if none, consider all
    const openish = markets.filter(m => {
      const s = String(m.status || "").toLowerCase();
      return s === "open" || s === "trading" || s === "active";
    });
    const open = openish.length ? openish : markets;
    if (!open.length) return res.status(204).end();
    
    // Build ranges array with all markets and their probabilities
    const ranges = [];
    let best = null;
    
    for (const m of open) {
      const prob = impliedYesProb(m); // 0..1 or null
      if (prob == null) continue;
      
      const label = m.subtitle || m.title || m.ticker || "Range";
      const range = parseRangeFromLabel(label);
      
      ranges.push({
        label,
        min: range.min,
        max: range.max,
        prob: Math.round(prob * 100) / 100,
        ticker: m.ticker
      });
      
      if (!best || prob > best.prob) {
        best = {
          prob,
          label,
          ticker: m.ticker
        };
      }
    }
    
    if (!best) return res.status(204).end();
    
    // Sort ranges by min temperature
    ranges.sort((a, b) => {
      if (a.min === null) return -1;
      if (b.min === null) return -1;
      return a.min - b.min;
    });
    
    return res.status(200).json({
      eventTicker,
      leadingLabel: best.label,                        // UNCHANGED - existing code works
      leadingProb: Math.round(best.prob * 100) / 100,  // UNCHANGED - existing code works
      url: "https://kalshi.com/markets/kxhighny",      // UNCHANGED - existing code works
      ranges                                            // NEW - all market ranges
    });
  } catch (err) {
    console.error(err);
    return res.status(502).json({ error: "Upstream error", details: String(err) });
  }
}

function toKalshiEventTicker(dateISO) {
  const [Y, M, D] = dateISO.split("-");
  const yy = Y.slice(-2);
  const mon = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"][Number(M)-1];
  return `KXHIGHNY-${yy}${mon}${D}`;
}

// NEW FUNCTION: Parse temperature ranges from Kalshi labels
function parseRangeFromLabel(label) {
  // Handle "X° to Y°" or "X to Y°" format
  const rangeMatch = label.match(/(\d+)°?\s*to\s*(\d+)°?/i);
  if (rangeMatch) {
    return { 
      min: parseInt(rangeMatch[1]), 
      max: parseInt(rangeMatch[2]) 
    };
  }
  
  // Handle "X° or less" OR "X° or below" format
  const lessMatch = label.match(/(\d+)°?\s*or\s*(?:less|below)/i);
  if (lessMatch) {
    return { 
      min: null, 
      max: parseInt(lessMatch[1]) 
    };
  }
  
  // Handle "X° or more" OR "X° or above" format
  const moreMatch = label.match(/(\d+)°?\s*or\s*(?:more|above)/i);
  if (moreMatch) {
    return { 
      min: parseInt(moreMatch[1]), 
      max: null 
    };
  }
  
  // Fallback - couldn't parse
  return { min: null, max: null };
}
