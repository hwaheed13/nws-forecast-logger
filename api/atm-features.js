// api/atm-features.js
// Live atmospheric signals from Open-Meteo (no API key needed).
// Returns BL height, cloud cover, solar radiation, 850/925mb temps,
// surface wind — the key intraday features feeding the v5 ML model.
// Updates whenever the ML prediction reruns (every ~30 min).
//
// Query params:
//   city=nyc (default) | lax

const CITIES = {
  nyc: { lat: 40.7834, lon: -73.965,  tz: "America/New_York",      label: "NYC" },
  lax: { lat: 33.94,   lon: -118.39,  tz: "America/Los_Angeles",   label: "LAX" },
};

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Cache-Control", "public, max-age=900"); // 15-min CDN cache

  const cityKey = (req.query?.city || "nyc").toLowerCase();
  const cfg = CITIES[cityKey] || CITIES.nyc;
  const { lat, lon, tz } = cfg;

  const HOURLY_VARS = [
    "boundary_layer_height",   // PBL mixing depth (m)
    "cloud_cover",             // % sky covered
    "shortwave_radiation",     // Solar irradiance W/m²
    "temperature_850hPa",      // 850mb warm-air advection
    "temperature_925hPa",      // 925mb near-surface advection
    "wind_speed_10m",          // Surface wind speed (mph)
    "wind_direction_10m",      // Surface wind direction (°)
    "temperature_2m",          // Surface temp (°F)
    "dew_point_2m",            // Dewpoint at 2m — moisture measure
    "precipitation",           // Hourly precipitation (in)
    "precipitation_probability", // Chance of rain (%)
  ].join(",");

  // Current conditions — updated every 15 minutes by Open-Meteo.
  // Gives real-time values rather than the hourly forecast, so the panel
  // reflects what the atmosphere is actually doing right now.
  const CURRENT_VARS = [
    "shortwave_radiation",     // Actual solar irradiance hitting surface now (W/m²)
    "cloud_cover",             // Actual cloud cover % right now
    "temperature_850hPa",      // Current 850mb temp (model analysis, updates every 15 min)
    "temperature_925hPa",      // Current 925mb temp (model analysis, updates every 15 min)
    "wind_speed_10m",          // Current surface wind speed
    "wind_direction_10m",      // Current surface wind direction
    "dew_point_2m",            // Current dewpoint
    "precipitation",           // Current precip (in last hour)
  ].join(",");

  const url =
    `https://api.open-meteo.com/v1/forecast` +
    `?latitude=${lat}&longitude=${lon}` +
    `&hourly=${HOURLY_VARS}` +
    `&current=${CURRENT_VARS}` +
    `&temperature_unit=fahrenheit` +
    `&wind_speed_unit=mph` +
    `&timezone=${encodeURIComponent(tz)}` +
    `&forecast_days=2`;

  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(12000) });
    if (!r.ok) throw new Error(`Open-Meteo ${r.status}`);
    const raw = await r.json();

    // Real-time current conditions (15-min updates)
    const currentConditions = raw.current || {};
    const solarNow    = currentConditions.shortwave_radiation  ?? null;
    const cloudNow    = currentConditions.cloud_cover          ?? null;
    const t850Now     = currentConditions.temperature_850hPa   ?? null;
    const t925Now     = currentConditions.temperature_925hPa   ?? null;
    const windNowLive = currentConditions.wind_speed_10m       ?? null;
    const wdirNowLive = currentConditions.wind_direction_10m   ?? null;
    const dewNow      = currentConditions.dew_point_2m         ?? null;
    const precipNow   = currentConditions.precipitation        ?? null;
    const currentTime = currentConditions.time                 ?? null;

    const hourly = raw.hourly || {};
    const times  = hourly.time || [];

    const now = new Date();
    const todayStr = now.toLocaleDateString("en-CA", { timeZone: tz }); // YYYY-MM-DD

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
        dew:   hourly.dew_point_2m?.[i],
        precip: hourly.precipitation?.[i],
        precip_prob: hourly.precipitation_probability?.[i],
      };
    }).filter(r => r.date === todayStr);

    // Peak heating window: 10am–5pm local
    const peak    = rows.filter(r => r.hour >= 10 && r.hour <= 17);
    const morning = rows.filter(r => r.hour >= 6  && r.hour <= 9);

    const avg = (arr, key) => {
      const vals = arr.map(r => r[key]).filter(v => v != null && !isNaN(v));
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    };
    const max = (arr, key) => {
      const vals = arr.map(r => r[key]).filter(v => v != null && !isNaN(v));
      return vals.length ? Math.max(...vals) : null;
    };

    const degToCard = (deg) => {
      if (deg == null) return null;
      const dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"];
      return dirs[Math.round(deg / 22.5) % 16];
    };

    const currentHour = parseInt(now.toLocaleString("en-US", { timeZone: tz, hour: "numeric", hour12: false }), 10);
    const currentRow  = rows.reduce((best, r) =>
      r.hour <= currentHour && (!best || r.hour > best.hour) ? r : best, null);

    const blCat = (m) => {
      if (m == null) return null;
      if (m < 500)  return "very-shallow";
      if (m < 1000) return "shallow";
      if (m < 2000) return "moderate";
      return "deep";
    };

    // Sea-breeze onshore directions differ by city:
    //   NYC: E / ENE / ESE / SE  (ocean is to the east)
    //   LAX: SW / W / WSW / SSW  (ocean is to the west)
    const SEA_BREEZE_DIRS = cityKey === "lax"
      ? ["SW", "W", "WSW", "SSW", "S"]
      : ["E", "ENE", "ESE", "SE"];

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
    const wdirCard = degToCard(wdirNow);

    // Precipitation aggregates:
    //  - remaining_day_in: sum of hourly precip from current hour through 11pm
    //  - max_prob_rest:    max precip-probability % from now through end of day
    // Both are "what's coming" signals for the solar-window-before-rain pattern.
    const restOfDay = rows.filter(r => r.hour >= currentHour);
    const precipRemaining = restOfDay
      .map(r => r.precip).filter(v => v != null && !isNaN(v))
      .reduce((a, b) => a + b, 0);
    const precipProbMax = Math.max(0, ...restOfDay
      .map(r => r.precip_prob).filter(v => v != null && !isNaN(v)));
    const precipHoursAhead = restOfDay.find(r => r.precip != null && r.precip > 0.01)?.hour;

    // Dewpoint — mean dewpoint during peak heating window (moist air caps temps)
    const dewPeak  = avg(peak, "dew");

    const warmAdv = t850Pk != null ? (t850Pk > 50 ? "warm" : t850Pk < 40 ? "cold" : "neutral") : null;

    res.status(200).json({
      fetchedAt:      new Date().toISOString(),
      date:           todayStr,
      city:           cityKey,
      current: {
        temp:              tempNow  != null ? Math.round(tempNow  * 10) / 10 : null,
        wind_mph:          windNow  != null ? Math.round(windNow  * 10) / 10 : null,
        wind_dir_deg:      wdirNow  != null ? Math.round(wdirNow) : null,
        wind_dir_card:     wdirCard,
        sea_breeze:        wdirCard ? SEA_BREEZE_DIRS.includes(wdirCard) : false,
        // Real-time values (15-min Open-Meteo updates) — more dynamic than hourly forecast
        solar_now_wm2:      solarNow    != null ? Math.round(solarNow)              : null,
        cloud_cover_now:    cloudNow    != null ? Math.round(cloudNow)              : null,
        temp_850hPa_now:    t850Now     != null ? Math.round(t850Now   * 10) / 10  : null,
        temp_925hPa_now:    t925Now     != null ? Math.round(t925Now   * 10) / 10  : null,
        wind_mph_now:       windNowLive != null ? Math.round(windNowLive * 10) / 10 : null,
        wind_dir_deg_now:   wdirNowLive != null ? Math.round(wdirNowLive)           : null,
        wind_dir_card_now:  degToCard(wdirNowLive),
        dewpoint_now:       dewNow      != null ? Math.round(dewNow * 10) / 10     : null,
        precip_now_in:      precipNow   != null ? Math.round(precipNow * 100) / 100 : null,
        current_obs_time:   currentTime,
      },
      peak_heating: {
        bl_height_max_m:  blPeak  != null ? Math.round(blPeak)  : null,
        bl_height_mean_m: blMean  != null ? Math.round(blMean)  : null,
        bl_category:      blCat(blPeak),
        cloud_cover_pct:  cloudPk != null ? Math.round(cloudPk) : null,
        solar_peak_wm2:   solarPk != null ? Math.round(solarPk) : null,
        solar_mean_wm2:   solarMn != null ? Math.round(solarMn) : null,
        temp_850hPa:      t850Pk  != null ? Math.round(t850Pk  * 10) / 10 : null,
        temp_925hPa:      t925Pk  != null ? Math.round(t925Pk  * 10) / 10 : null,
        dewpoint_mean:    dewPeak != null ? Math.round(dewPeak * 10) / 10 : null,
        warm_advection:   warmAdv,
      },
      precip_forecast: {
        remaining_today_in: Math.round(precipRemaining * 100) / 100,
        max_prob_rest_pct:  Number.isFinite(precipProbMax) ? Math.round(precipProbMax) : null,
        first_hour_ahead:   precipHoursAhead ?? null,
      },
      morning: {
        temp_6am: temp6am != null ? Math.round(temp6am * 10) / 10 : null,
        avg_wind: avg(morning, "wind") != null ? Math.round(avg(morning, "wind") * 10) / 10 : null,
      },
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}
