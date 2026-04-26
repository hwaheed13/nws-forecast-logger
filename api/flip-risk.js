// api/flip-risk.js
// Forward-looking flip risk and blow-past signal for the current day's prediction.
//
// Combines two complementary approaches:
//
//  1. HISTORICAL ANALOG RATE
//     Queries all scored Supabase rows where ml_bucket_canonical matches today's
//     canonical bucket. Computes:
//       - Overall flip rate (how often latest ≠ canonical)
//       - Conditional flip rate for "similar conditions" (cold 925mb, morning hour, etc.)
//       - Blow-past rate (actual exceeded highest model by 2°F+)
//       - When canonical flipped, was the flip good or bad?
//
//  2. SIGNAL-BASED SCORE (available immediately, no history needed)
//     Probability gap, ensemble stability, observed temp trajectory, wind persistence —
//     all scored as low/medium/high flip risk right now.
//
// Query params:
//   city              nyc | lax
//   canonical_bucket  e.g. "55 or less"
//   ml_confidence     0-1 float  (confidence of canonical bucket)
//   ml_f              float      (ML center temperature)
//   nws_f             float      (NWS forecast)
//   accu_f            float      (AccuWeather forecast)
//   ensemble_range    float      (HRRR-ECMWF spread in °F)
//   current_obs_temp  float      (latest observed temp, °F)
//   canonical_hour    int        (hour of day canonical was written, local)
//   t925_live         float      (live 925mb temp, °F)
//   wind_dir_card     string     (e.g. "NNE")
//   top2_confidence   float      (confidence of 2nd-best bucket, 0-1)
//
//   Station-level marine cap signals (v9 additions — the April 12 fix):
//   syn_vs_nws        float  Synoptic network mean minus NWS forecast (°F)
//                            Negative = entire station network below NWS = cap signal
//   jfk_temp          float  JFK Airport temp (°F)
//   jfk_vs_nws        float  JFK minus NWS (°F) — most sensitive leading cap indicator
//   kjfk_vs_knyc      float  JFK minus KNYC (°F) — negative = sea breeze penetrating inland
//   coastal_vs_inland float  mean(JFK,LGA) minus mean(EWR,TEB) — negative = marine gradient
//   marine_active     '1'    set if LGA-JFK gradient ≥3°F (sea breeze detected)
//   hrrr_vs_nws       float  HRRR minus NWS (°F) — negative = HRRR sensing cap NWS misses

const CITIES = {
  nyc: { tz: "America/New_York" },
  lax: { tz: "America/Los_Angeles" },
};

// Directions that suppress daytime heating for each city
const COLD_DIRS = {
  nyc: ["N", "NNE", "NE", "ENE", "E", "NNW"],
  lax: ["SW", "W", "WSW", "SSW", "S"],
};

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Cache-Control", "public, max-age=300"); // 5-min cache

  const q = req.query || {};
  const cityKey        = (q.city || "nyc").toLowerCase();
  const canonicalBkt   = q.canonical_bucket || "";
  const mlConf         = parseFloat(q.ml_confidence)   || null;
  const mlF            = parseFloat(q.ml_f)            || null;
  const nwsF           = parseFloat(q.nws_f)           || null;
  const accuF          = parseFloat(q.accu_f)          || null;
  const ensRange       = parseFloat(q.ensemble_range)  || null;
  const obsTemp        = parseFloat(q.current_obs_temp)|| null;
  // Day's peak so far (6-hr rolling max). Used to compare against ML center —
  // catches brief temp spikes that happened between live polls.
  const obsMaxToday    = parseFloat(q.obs_max_today)   || null;
  const canonHour      = parseInt(q.canonical_hour)    || null;
  const t925Live       = parseFloat(q.t925_live)       || null;
  const windDir        = (q.wind_dir_card || "").toUpperCase();
  const top2Conf       = parseFloat(q.top2_confidence) || null;

  // Station-level marine cap signals (v9)
  const synVsNws       = q.syn_vs_nws        != null && q.syn_vs_nws        !== '' ? parseFloat(q.syn_vs_nws)        : null;
  const jfkTemp        = q.jfk_temp          != null && q.jfk_temp          !== '' ? parseFloat(q.jfk_temp)          : null;
  const jfkVsNws       = q.jfk_vs_nws        != null && q.jfk_vs_nws        !== '' ? parseFloat(q.jfk_vs_nws)        : null;
  const kjfkVsKnyc     = q.kjfk_vs_knyc      != null && q.kjfk_vs_knyc      !== '' ? parseFloat(q.kjfk_vs_knyc)      : null;
  const coastalVsInland= q.coastal_vs_inland  != null && q.coastal_vs_inland  !== '' ? parseFloat(q.coastal_vs_inland) : null;
  const marineActive   = q.marine_active === '1';
  const hrrrVsNws      = q.hrrr_vs_nws       != null && q.hrrr_vs_nws       !== '' ? parseFloat(q.hrrr_vs_nws)       : null;

  // ── 1. Historical analog query from Supabase ────────────────────────────
  const supaUrl = process.env.SUPABASE_URL;
  const supaKey = process.env.SUPABASE_SERVICE_ROLE;

  let history = { total: 0, flipped: 0, flipGood: 0, flipBad: 0,
                  blowPast: 0, coldSimilar: 0, coldFlipped: 0,
                  canonWins: 0, latestWins: 0 };

  if (supaUrl && supaKey && canonicalBkt) {
    try {
      const endpoint = `${supaUrl}/rest/v1/prediction_logs`;
      const encoded  = encodeURIComponent(canonicalBkt);
      const url = `${endpoint}?city=eq.${cityKey}` +
                  `&ml_bucket_canonical=eq.${encoded}` +
                  `&ml_result=not.is.null` +
                  `&ml_actual_high=not.is.null` +
                  `&select=ml_bucket_canonical,ml_bucket,ml_result,ml_result_canonical,` +
                  `ml_actual_high,nws_d0,accuweather,atm_snapshot` +
                  `&order=target_date.desc&limit=200`;

      const r = await fetch(url, {
        headers: {
          apikey: supaKey,
          Authorization: `Bearer ${supaKey}`,
          Accept: "application/json",
        },
        signal: AbortSignal.timeout(8000),
      });

      if (r.ok) {
        const rows = await r.json();
        history.total = rows.length;

        for (const row of rows) {
          const flipped = row.ml_bucket !== row.ml_bucket_canonical;
          const canonWin  = row.ml_result_canonical === "WIN";
          const latestWin = row.ml_result === "WIN";
          const actual    = parseFloat(row.ml_actual_high) || null;

          if (canonWin)  history.canonWins++;
          if (latestWin) history.latestWins++;

          if (flipped) {
            history.flipped++;
            if (latestWin && !canonWin)  history.flipGood++;  // flip improved outcome
            if (!latestWin && canonWin)  history.flipBad++;   // flip hurt outcome
          }

          // Blow-past: actual exceeded ALL model forecasts by 2°F+
          const nws  = parseFloat(row.nws_d0)      || null;
          const accu = parseFloat(row.accuweather) || null;
          const highestModel = Math.max(
            ...[nws, accu].filter(v => v != null)
          );
          if (actual != null && highestModel > 0 && actual >= highestModel + 2) {
            history.blowPast++;
          }

          // "Cold similar" — 925mb < 45°F at canonical snapshot time
          // atm_snapshot may be a string or already-parsed object
          try {
            const snap = typeof row.atm_snapshot === "string"
              ? JSON.parse(row.atm_snapshot)
              : (row.atm_snapshot || {});
            const snap925 = snap.atm_925mb_temp_mean ?? snap["925mb_temp"] ?? null;
            if (snap925 != null && snap925 < 45) {
              history.coldSimilar++;
              if (flipped) history.coldFlipped++;
            }
          } catch (_) { /* skip unparseable snapshots */ }
        }
      }
    } catch (e) {
      // Non-fatal — fall back to signal-only mode
      console.warn("flip-risk Supabase query failed:", e.message);
    }
  }

  // ── 2. Signal-based flip risk score ─────────────────────────────────────
  const signals = [];
  let flipPoints = 0;   // higher = more likely to flip
  let blowPoints = 0;   // higher = more likely to blow past models

  // A) Probability gap between top-1 and top-2 bucket
  // Narrow gap = model is uncertain = more flip-prone
  if (mlConf != null && top2Conf != null) {
    const gap = mlConf - top2Conf;
    if (gap < 0.15) {
      flipPoints += 3;
      signals.push({ factor: "probability_gap", dir: "flip", weight: 3,
        label: `Tight margin: ${Math.round(mlConf*100)}% vs ${Math.round(top2Conf*100)}% (gap ${Math.round(gap*100)}pp)` });
    } else if (gap < 0.30) {
      flipPoints += 1;
      signals.push({ factor: "probability_gap", dir: "flip", weight: 1,
        label: `Moderate margin: ${Math.round(mlConf*100)}% vs ${Math.round(top2Conf*100)}%` });
    } else {
      signals.push({ factor: "probability_gap", dir: "hold", weight: 0,
        label: `Clear margin: ${Math.round(mlConf*100)}% vs ${Math.round(top2Conf*100)}% (${Math.round(gap*100)}pp gap)` });
    }
  }

  // B) Ensemble stability — tight spread = models agree = hard to flip
  if (ensRange != null) {
    if (ensRange > 3.0) {
      flipPoints += 2;
      blowPoints += 1;
      signals.push({ factor: "ensemble_stability", dir: "flip", weight: 2,
        label: `High model spread: ${ensRange.toFixed(1)}°F — models disagree` });
    } else if (ensRange > 1.5) {
      flipPoints += 1;
      signals.push({ factor: "ensemble_stability", dir: "flip", weight: 1,
        label: `Moderate spread: ${ensRange.toFixed(1)}°F` });
    } else {
      signals.push({ factor: "ensemble_stability", dir: "hold", weight: 0,
        label: `Stable ensemble: ${ensRange.toFixed(1)}°F spread — models agree` });
    }
  }

  // C) Observed temp trajectory — how far below predicted high is current obs?
  // Large gap at early hour = long way to flip bucket upward.
  //
  // Use the day's peak so far (max of live obs and 6-hr rolling max) for the gap
  // calculation, not just the live "now" reading. A brief mid-cycle spike to 56°F
  // counts as having reached 56°F, even if the next live poll registers 55°F.
  // The label still shows the live "now" so users see the current state, but adds
  // "(peak X°F)" when the day's peak is materially above the current reading.
  if (obsTemp != null && mlF != null && canonHour != null) {
    // Effective temp for gap math = max(live, peak-so-far)
    const effTemp  = (obsMaxToday != null && obsMaxToday > obsTemp) ? obsMaxToday : obsTemp;
    const peakNote = (obsMaxToday != null && obsMaxToday > obsTemp + 0.5)
      ? ` (peak ${obsMaxToday.toFixed(1)}°F)` : '';
    const gap = mlF - effTemp;
    if (gap > 8 && canonHour < 10) {
      // Very cold start, needs big warming to reach predicted high
      signals.push({ factor: "obs_trajectory", dir: "hold", weight: 0,
        label: `${obsTemp.toFixed(1)}°F now${peakNote}, needs +${gap.toFixed(1)}°F to reach ML center — heating gap suppresses upside flips` });
    } else if (gap < 0 && canonHour >= 10) {
      // Already exceeded predicted high — blow-past CONFIRMED
      flipPoints += 2;
      blowPoints += 3;
      signals.push({ factor: "obs_trajectory", dir: "blow", weight: 3,
        label: `${obsTemp.toFixed(1)}°F now${peakNote}, ${Math.abs(gap).toFixed(1)}°F ABOVE ML center — blow-past CONFIRMED` });
    } else if (gap < 2 && canonHour >= 10) {
      // Already near the predicted high — actual could exceed easily
      flipPoints += 1;
      blowPoints += 2;
      signals.push({ factor: "obs_trajectory", dir: "blow", weight: 2,
        label: `${obsTemp.toFixed(1)}°F now${peakNote}, only ${gap.toFixed(1)}°F below ML center — blow-past risk elevated` });
    } else {
      signals.push({ factor: "obs_trajectory", dir: "neutral", weight: 0,
        label: `${obsTemp.toFixed(1)}°F now${peakNote}, ${gap.toFixed(1)}°F below ML center` });
    }
  }

  // D) 925mb temperature — cold near-surface air = structural suppressor
  if (t925Live != null) {
    if (t925Live < 38) {
      signals.push({ factor: "925mb_temp", dir: "hold", weight: 0,
        label: `Very cold 925mb (${t925Live.toFixed(1)}°F) — strong airmass suppressor, flip unlikely` });
    } else if (t925Live < 45) {
      signals.push({ factor: "925mb_temp", dir: "hold", weight: 0,
        label: `Cold 925mb (${t925Live.toFixed(1)}°F) — moderately suppressive` });
    } else if (t925Live > 55) {
      flipPoints += 1;
      blowPoints += 2;
      signals.push({ factor: "925mb_temp", dir: "blow", weight: 2,
        label: `Warm 925mb (${t925Live.toFixed(1)}°F) — warm airmass, blow-past risk elevated` });
    }
  }

  // E) Wind direction — cold/suppressing direction = flip to warmer bucket is harder
  const coldDirs = COLD_DIRS[cityKey] || COLD_DIRS.nyc;
  if (windDir) {
    if (coldDirs.includes(windDir)) {
      signals.push({ factor: "wind_direction", dir: "hold", weight: 0,
        label: `${windDir} flow — suppressing direction for ${cityKey.toUpperCase()}, flip to warmer bucket requires wind shift` });
    } else if (["SW", "S", "SSW", "WSW"].includes(windDir)) {
      flipPoints += 1;
      blowPoints += 2;
      signals.push({ factor: "wind_direction", dir: "blow", weight: 2,
        label: `${windDir} flow — warm direction, blow-past risk elevated` });
    }
  }

  // F) Agency divergence — when ML runs colder than NWS, models disagree on heating
  if (mlF != null && nwsF != null) {
    const mlVsNws = mlF - nwsF;
    if (mlVsNws < -3) {
      // ML much colder than NWS — if NWS is right, we'd flip to a warmer bucket
      flipPoints += 1;
      signals.push({ factor: "agency_divergence", dir: "flip", weight: 1,
        label: `ML ${Math.abs(mlVsNws).toFixed(1)}°F below NWS — if agencies correct, flip to warmer bucket possible` });
    } else if (mlVsNws > 3) {
      // ML much warmer than NWS — blow-past potential if ML correct
      blowPoints += 1;
      signals.push({ factor: "agency_divergence", dir: "blow", weight: 1,
        label: `ML ${mlVsNws.toFixed(1)}°F above NWS — if ML correct, blow-past possible` });
    }
  }

  // ── G) MARINE CAP SIGNALS — the April 12 signals that were missing ──────
  // These are the field observations that directly contradict an upward flip.
  // When these fire, the flip risk score should DROP regardless of model uncertainty.
  //
  // The logic: a tight probability gap (signal A) means the model is uncertain.
  // But if the station network is uniformly 7°F below NWS, that uncertainty resolves
  // downward — it means the cap is real, not that the warmer bucket is more likely.
  // We subtract flipPoints to suppress false upward-flip signals.

  let capHoldPoints = 0;  // cap-hold evidence — subtracts from flipPoints

  // G1) Synoptic station network below NWS — the entire 20-station network is capped
  if (synVsNws != null) {
    if (synVsNws <= -6) {
      capHoldPoints += 4;
      signals.push({ factor: "synoptic_vs_nws", dir: "hold", weight: 4,
        label: `Station network ${Math.abs(synVsNws).toFixed(1)}°F below NWS (${20}+ stations) — field data strongly contradicts upward move` });
    } else if (synVsNws <= -4) {
      capHoldPoints += 3;
      signals.push({ factor: "synoptic_vs_nws", dir: "hold", weight: 3,
        label: `Station network ${Math.abs(synVsNws).toFixed(1)}°F below NWS — cap signal across full network` });
    } else if (synVsNws <= -2) {
      capHoldPoints += 1;
      signals.push({ factor: "synoptic_vs_nws", dir: "hold", weight: 1,
        label: `Station network ${Math.abs(synVsNws).toFixed(1)}°F below NWS forecast` });
    }
  }

  // G2) JFK vs NWS — coastal airport is the leading indicator of marine cap strength
  if (jfkVsNws != null) {
    if (jfkVsNws <= -10) {
      capHoldPoints += 4;
      signals.push({ factor: "jfk_vs_nws", dir: "hold", weight: 4,
        label: `JFK ${Math.abs(jfkVsNws).toFixed(0)}°F below NWS forecast — marine cap at the coast is decisive` });
    } else if (jfkVsNws <= -7) {
      capHoldPoints += 3;
      signals.push({ factor: "jfk_vs_nws", dir: "hold", weight: 3,
        label: `JFK ${Math.abs(jfkVsNws).toFixed(0)}°F below NWS — strong marine suppression at coastal station` });
    } else if (jfkVsNws <= -4) {
      capHoldPoints += 1;
      signals.push({ factor: "jfk_vs_nws", dir: "hold", weight: 1,
        label: `JFK ${Math.abs(jfkVsNws).toFixed(0)}°F below NWS — moderate coastal cooling` });
    }
  }

  // G3) Coastal-inland gradient — the physical signature of a marine cap
  // When coastal (JFK, LGA) is much colder than inland (EWR, TEB), the air mass
  // boundary is clear. This is not a model artifact — it's measured reality.
  if (coastalVsInland != null) {
    if (coastalVsInland <= -4) {
      capHoldPoints += 3;
      signals.push({ factor: "coastal_vs_inland", dir: "hold", weight: 3,
        label: `Coastal ${Math.abs(coastalVsInland).toFixed(1)}°F colder than inland — marine air mass boundary confirmed` });
    } else if (coastalVsInland <= -2) {
      capHoldPoints += 1;
      signals.push({ factor: "coastal_vs_inland", dir: "hold", weight: 1,
        label: `Coastal ${Math.abs(coastalVsInland).toFixed(1)}°F colder than inland — marine influence present` });
    }
  }

  // G4) JFK-KNYC diff — sea breeze penetration indicator
  if (kjfkVsKnyc != null && kjfkVsKnyc <= -2) {
    capHoldPoints += 1;
    signals.push({ factor: "kjfk_vs_knyc", dir: "hold", weight: 1,
      label: `JFK ${Math.abs(kjfkVsKnyc).toFixed(1)}°F colder than Central Park — sea breeze pushing inland` });
  }

  // G5) Marine active flag (sea breeze confirmed)
  if (marineActive && capHoldPoints === 0) {
    // Only add this if we don't already have stronger station signals
    capHoldPoints += 1;
    signals.push({ factor: "marine_active", dir: "hold", weight: 1,
      label: `Sea breeze active — coastal-to-inland gradient suppresses afternoon heating` });
  }

  // G6) HRRR vs NWS — when HRRR (most accurate model) is colder than NWS
  if (hrrrVsNws != null) {
    if (hrrrVsNws <= -3) {
      capHoldPoints += 2;
      signals.push({ factor: "hrrr_vs_nws", dir: "hold", weight: 2,
        label: `HRRR ${Math.abs(hrrrVsNws).toFixed(0)}°F below NWS — highest-accuracy model sensing cap NWS is missing` });
    } else if (hrrrVsNws <= -1.5) {
      capHoldPoints += 1;
      signals.push({ factor: "hrrr_vs_nws", dir: "hold", weight: 1,
        label: `HRRR ${Math.abs(hrrrVsNws).toFixed(1)}°F below NWS — HRRR cooler than NWS` });
    }
  }

  // Apply cap suppression: strong cap evidence cancels out model-uncertainty flip signals
  // Cap signals are reality — model spread just means the model is uncertain, not that
  // the warmer outcome is equally plausible when the whole station network is 7°F below.
  flipPoints = Math.max(0, flipPoints - capHoldPoints);
  // Also zero out blow-past if strong marine cap is active
  if (capHoldPoints >= 4) blowPoints = 0;

  // ── 3. Synthesize overall risk level ────────────────────────────────────
  const flipRisk = flipPoints >= 4 ? "HIGH"
                 : flipPoints >= 2 ? "MEDIUM"
                 : "LOW";

  const blowRisk = blowPoints >= 3 ? "HIGH"
                 : blowPoints >= 2 ? "MEDIUM"
                 : "NONE";

  // Historical rates (safe division)
  const flipRate      = history.total > 0 ? history.flipped     / history.total  : null;
  const coldFlipRate  = history.coldSimilar > 0 ? history.coldFlipped / history.coldSimilar : null;
  const blowPastRate  = history.total > 0 ? history.blowPast    / history.total  : null;
  const canonAccuracy = history.total > 0 ? history.canonWins   / history.total  : null;
  const flipGoodRate  = history.flipped > 0 ? history.flipGood  / history.flipped : null;

  // Human-readable summary — specific, actionable, not vague
  let summary = "";
  if (flipRisk === "LOW" && blowRisk === "NONE") {
    summary = "Canonical holding — suppressor signals intact";
  } else if (flipRisk === "HIGH") {
    summary = "Structural flip risk — bucket may shift before settlement";
  } else if (blowRisk === "HIGH") {
    summary = "Blow-past risk — actual may exceed all model forecasts";
  } else if (flipRisk === "MEDIUM" && blowRisk === "NONE") {
    summary = "Watch for wind shift or obs surge after 10 AM";
  } else if (flipRisk === "MEDIUM") {
    summary = "Mixed — flip and blow-past both possible, watch 10 AM obs";
  } else {
    summary = "Obs running warm — watch mid-morning heating rate";
  }

  res.status(200).json({
    fetchedAt: new Date().toISOString(),
    city: cityKey,
    canonical_bucket: canonicalBkt,

    flip_risk: {
      level: flipRisk,         // LOW | MEDIUM | HIGH
      score: flipPoints,       // raw score
      summary,
    },
    blow_past_risk: {
      level: blowRisk,         // NONE | MEDIUM | HIGH
      score: blowPoints,
    },

    signals,                   // array of {factor, dir, weight, label}

    history: {
      days_sampled:      history.total,
      flip_rate:         flipRate   != null ? Math.round(flipRate   * 100) : null,
      cold_analog_rate:  coldFlipRate != null ? Math.round(coldFlipRate * 100) : null,
      blow_past_rate:    blowPastRate != null ? Math.round(blowPastRate * 100) : null,
      canonical_win_pct: canonAccuracy != null ? Math.round(canonAccuracy * 100) : null,
      flip_improved_pct: flipGoodRate != null ? Math.round(flipGoodRate * 100) : null,
      n_flipped:         history.flipped,
      n_cold_similar:    history.coldSimilar,
    },
  });
}
