// api/log-prediction.js
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY // server-side secret (never ship to client)
);

export default async function handler(req, res) {
  // CORS / probe for your client-side OPTIONS check
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    return res.status(200).end();
  }

  if (req.method !== 'POST') return res.status(405).end();

  try {
    const p = req.body || {};
    if (!p.target_date || !p.lead_used) {
      return res.status(400).json({ error: 'target_date and lead_used required' });
    }

    const row = {
      timestamp_et: p.timestamp_et ?? null,
      target_date: p.target_date,
      lead_used: p.lead_used,                   // 'D0' | 'D1'
      is_carryover: !!p.is_carryover,
      rep_forecast: num(p.rep_forecast),
      delta_last3: num(p.delta_last3),
      issuance_iso: p.issuance_iso ?? null,
      issuance_hour_local: int0_23(p.issuance_hour_local),
      season: p.season ?? null,                 // 'warm' | 'cool'
      bias_applied: num(p.bias_applied),
      prediction_value: num(p.prediction_value),
      uncertainty_rmse: num(p.uncertainty_rmse),
      model_name: p.model_name ?? null,
      version: p.version ?? null,

      // optional extras if you later send them
      weights_first: num(p.weights_first),
      weights_latest: num(p.weights_latest),
      weights_trimmed: num(p.weights_trimmed),
      weights_carry: num(p.weights_carry),
      representative_sig: p.representative_sig ?? null,
      forecast_signature: p.forecast_signature ?? null,
      representative_kind: p.representative_kind ?? 'blend',
      source_card: p.source_card ?? null,
      user_id: p.user_id ?? null,
    };

    const { error } = await supabase
  .from('prediction_logs')
  .upsert(row, { onConflict: 'target_date,lead_used,issuance_iso,model_name,version' });
    if (error) throw error;

    // CORS header on response too (handy if you ever call cross-origin)
    res.setHeader('Access-Control-Allow-Origin', '*');
    return res.status(201).json({ ok: true });
  } catch (e) {
    console.error('/api/log-prediction error', e);
    return res.status(500).json({ error: 'insert failed' });
  }
}

function num(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}
function int0_23(x) {
  const n = Number(x);
  return Number.isInteger(n) && n >= 0 && n <= 23 ? n : null;
}
