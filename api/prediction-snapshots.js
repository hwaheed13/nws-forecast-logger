// api/prediction-snapshots.js
// Server-side endpoint that queries prediction_logs with service role key
// so it can filter by city (anon key can't filter on city column).
import { createClient } from '@supabase/supabase-js';

export const config = { runtime: 'nodejs' };

let _sb = null;
function getSupabase() {
  const { SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY } = process.env;
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) return null;
  if (!_sb) {
    _sb = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
      auth: { persistSession: false }
    });
  }
  return _sb;
}

function withCORS(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

export default async function handler(req, res) {
  withCORS(res);
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'GET') return res.status(405).json({ error: 'GET only' });

  const supabase = getSupabase();
  if (!supabase) return res.status(500).json({ error: 'Missing Supabase config' });

  const city = req.query.city || 'nyc';
  // mode=stats returns latest prediction per day (for accuracy stats)
  // mode=table returns night_before + morning_of per day (for table display)
  const mode = req.query.mode || 'table';

  try {
    if (mode === 'stats') {
      // Latest prediction per target_date for this city
      const { data, error } = await supabase
        .from('prediction_logs')
        .select('target_date, prediction_value, lead_used')
        .eq('city', city)
        .in('lead_used', ['today_for_today', 'D0'])
        .order('timestamp', { ascending: false });

      if (error) return res.status(500).json({ error: error.message });

      // Deduplicate: keep latest per target_date
      const seen = new Set();
      const rows = (data || []).filter(row => {
        const td = String(row.target_date || '').match(/\d{4}-\d{2}-\d{2}/)?.[0];
        if (!td || seen.has(td)) return false;
        seen.add(td);
        return true;
      }).map(row => ({
        target_date: row.target_date,
        same_day_earliest_value: row.prediction_value,
        prev_day_latest_value: null,
      }));

      return res.status(200).json(rows);
    }

    // mode=table: group by target_date with morning_of and night_before
    const { data, error } = await supabase
      .from('prediction_logs')
      .select('target_date, prediction_value, lead_used, timestamp')
      .eq('city', city)
      .in('lead_used', ['today_for_today', 'today_for_tomorrow', 'D0', 'D1'])
      .order('timestamp', { ascending: false });

    if (error) return res.status(500).json({ error: error.message });

    const isMorning = (lu) => lu === 'today_for_today' || lu === 'D0';
    const isNight = (lu) => lu === 'today_for_tomorrow' || lu === 'D1';
    const byDate = {};

    for (const row of (data || [])) {
      const m = String(row.target_date || '').match(/\d{4}-\d{2}-\d{2}/);
      const td = m ? m[0] : null;
      if (!td || row.prediction_value == null) continue;
      if (!byDate[td]) byDate[td] = {};
      if (isMorning(row.lead_used) && !byDate[td].morning_of) {
        byDate[td].morning_of = Number(row.prediction_value);
      }
      if (isNight(row.lead_used) && !byDate[td].night_before) {
        byDate[td].night_before = Number(row.prediction_value);
      }
    }

    const rows = Object.entries(byDate).map(([td, vals]) => ({
      target_date: td,
      same_day_earliest_value: vals.morning_of ?? null,
      prev_day_latest_value: vals.night_before ?? null,
    }));

    return res.status(200).json(rows);
  } catch (e) {
    console.error('/api/prediction-snapshots error:', e);
    return res.status(500).json({ error: e?.message || String(e) });
  }
}
