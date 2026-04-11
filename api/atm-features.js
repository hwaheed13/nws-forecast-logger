// api/atm-features.js
// Live atmospheric signals from Open-Meteo (no API key needed).
// Returns BL height, cloud cover, solar radiation, 850/925mb temps,
// surface wind — the key intraday features feeding the v5 ML model.
// Updates whenever the ML prediction reruns (every ~30 min).

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Cache-Control", "public, max-age=900"); // 15-min CDN cache

  const LAT = 40.7834;
  const LON = -73.965;
  const TZ  = "America/New_York";

  const HOURLY_VARS = [
    "boundary_layer_height",   // PBL mixing depth (m)
    "cloud_cover",             // % sky covered
    "shortwave_radiation",     // Solar irradiance W/m²
    "temperature_850hPa",      // 850mb warm-air advection
    "temperature_925hPa",      // 925mb near-surface advection
    "wind_speed_10m",          // Surface wind speed (mph)
    "wind_direction_10m",      // Surface wind direction (°)
    "temperature_2m",          // Surface temp (°F)
  ].join(",");

  const url =
    `https://api.open-meteo.com/v1/forecast` +
    `?latitude=${LAT}&longitude=${LON}` +
    `&hourly=${HOURLY_VARS}` +
    `&temperature_unit=fahrenheit` +
    `&wind_speed_unit=mph` +
    `&timezone=${encodeURIComponent(TZ)}` +
    `&forecast_days=2`;

  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(12000) });
    if (!r.ok) throw new Error(`Open-Meteo ${r.status}`);
    const raw = await r.json();

    const hourly = raw.hourly || {};
    const times  = hourly.time || [];

    // Build a lookup: hour string → index
    const now = new Date();
    // ET local time — use UTC offset for Eastern (EST=-5, EDT=-4)
    // We'll identify "today" rows by the date portion of the time strings,
    // which Open-Meteo returns in the requested timezone.
    const todayStr = now.toLocaleDateString("en-CA", { timeZone: TZ }); // YYYY-MM-DD

    const rows = times.map((t, i) => {
      const [datePart, timePart] = t.split("T");
      const hour = parseInt(timePart?.split(":")[0] ?? "0", 10);
      return {
        date: datePart,
        hour,
        bl:    hourly.boundary_layer_height?.[i],
        cloud: hourly.cloud_cover?.[i],
        solar: hourly.shortwave_radiation?.[i],
        t850:  hourly.temperature_850hPa?.[i],
        t925:  hourly.temperature_925hPa?.[i],
        wind:  hourly.wind_speed_10m?.[i],
        wdir:  hourly.wind_direction_10m?.[i],
        temp:  hourly.temperature_2m?.[i],
      };
    }).filter(r => r.date === todayStr);

    // Peak heating window: 10am–5pm local
    const peak = rows.filter(r => r.hour >= 10 && r.hour <= 17);
    const morning = rows.filter(r => r.hour >= 6 && r.hour <= 9);

    const avg = (arr, key) => {
      const vals = arr.map(r => r[key]).filter(v => v != null && !isNaN(v));
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    };
    const max = (arr, key) => {
      const vals = arr.map(r => r[key]).filter(v => v != null && !isNaN(v));
      return vals.length ? Math.max(...vals) : null;
    };

    // Wind direction helpers
    const degToCard = (deg) => {
      if (deg == null) return null;
      const dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"];
      return dirs[Math.round(deg / 22.5) % 16];
    };

    // Current-hour reading (or most recent available)
    const currentHour = now.getHours(); // local system hour — close enough for display
    const currentRow  = rows.reduce((best, r) =>
      r.hour <= currentHour && (!best || r.hour > best.hour) ? r : best, null);

    // BL height category
    const blCat = (m) => {
      if (m == null) return null;
      if (m < 500)  return "very-shallow";   // strong suppression
      if (m < 1000) return "shallow";         // moderate suppression
      if (m < 2000) return "moderate";        // normal mixing
      return "deep";                           // good mixing, supports high end
    };

    const blPeak   = max(peak, "bl");
    const blMean   = avg(peak, "bl");
    const cloudPk  = avg(peak, "cloud");
    const solarPk  = max(peak, "solar");
    const solarMn  = avg(peak, "solar");
    const t850Pk   = max(peak, "t850");
    const t925Pk   = max(peak, "t925");
    const windNow  = currentRow?.wind;
    const wdirNow  = currentRow?.wdir;
    const tempNow  = currentRow?.temp;
    const temp6am  = rows.find(r => r.hour === 6)?.temp;

    // Model pressure-level interpretation: warm 850mb = warm air advection
    // NYC typical: 850mb ~50°F in spring, each +5°F ≈ +2-3°F surface boost
    const warmAdv = t850Pk != null ? (t850Pk > 50 ? "warm" : t850Pk < 40 ? "cold" : "neutral") : null;

    res.status(200).json({
      fetchedAt:      new Date().toISOString(),
      date:           todayStr,
      current: {
        temp:         tempNow != null ? Math.round(tempNow * 10) / 10 : null,
        wind_mph:     windNow != null ? Math.round(windNow * 10) / 10 : null,
        wind_dir_deg: wdirNow != null ? Math.round(wdirNow) : null,
        wind_dir_card: degToCard(wdirNow),
      },
      peak_heating: {
        bl_height_max_m:  blPeak  != null ? Math.round(blPeak)  : null,
        bl_height_mean_m: blMean  != null ? Math.round(blMean)  : null,
        bl_category:      blCat(blPeak),
        cloud_cover_pct:  cloudPk != null ? Math.round(cloudPk) : null,
        solar_peak_wm2:   solarPk != null ? Math.round(solarPk) : null,
        solar_mean_wm2:   solarMn != null ? Math.round(solarMn) : null,
        temp_850hPa:      t850Pk  != null ? Math.round(t850Pk * 10) / 10 : null,
        temp_925hPa:      t925Pk  != null ? Math.round(t925Pk * 10) / 10 : null,
        warm_advection:   warmAdv,
      },
      morning: {
        temp_6am:    temp6am != null ? Math.round(temp6am * 10) / 10 : null,
        avg_wind:    avg(morning, "wind") != null ? Math.round(avg(morning, "wind") * 10) / 10 : null,
      },
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}
