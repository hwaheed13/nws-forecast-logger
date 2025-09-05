// api/log-prediction.js
import { createClient } from '@supabase/supabase-js';

// Use Node runtime (Edge can't use service-role keys)
export const config = { runtime: 'nodejs18.x' }; // 'nodejs20.x' also fine

const { SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY } = process.env;
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false }
});

function withCORS(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

const num = x => (Number.isFinite(Number(x)) ? Number(x) : null);
const int0_23 = x => {
  const n = Number(x);
  return Number.isInteger(n) && n >= 0 && n <= 23 ? n : null;
};

export default async function handler(req, res) {
  withCORS(res);

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    return res.status(500).json({ error: 'Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY' });
  }

  try {
    const p = typeof req.body === 'string' ? JSON.parse(req.body || '{}') : (req.body || {});
    if (!p.target_date || !p.lead_used) {
      return res.status(400).json({ error: 'target_date and lead_used required' });
    }

    const row = {
      timestamp_et: p.timestamp_et ?? null,
      target_date: p.target_date,                 // e.g. "2025-09-06"
      lead_used: p.lead_used,                     // 'D0' | 'D1'
      is_carryover: !!p.is_carryover,
      rep_forecast: num(p.rep_forecast),
      delta_last3: num(p.delta_last3),
      issuance_iso: p.issuance_iso ?? null,       // e.g. "2025-09-05 00:00:00+00"
      issuance_hour_local: int0_23(p.issuance_hour_local),
      season: p.season ?? null,                   // 'warm' | 'cool'
      bias_applied: num(p.bias_applied),
      prediction_value: num(p.prediction_value),
      uncertainty_rmse: num(p.uncertainty_rmse),
      model_name: p.model_name ?? null,           // e.g. "seasonal_lead_bias_blended_rep"
      version: p.version ?? null,                 // e.g. "v1.1"

      // optional extras
      weights_first: num(p.weights_first),
      weights_latest: num(p.weights_latest),
      weights_trimmed: num(p.weights_trimmed),
      weights_carry: num(p.weights_carry),
      representative_sig: p.representative_sig ?? null,
      forecast_signature: p.forecast_signature ?? null,
      representative_kind: p.representative_kind ?? 'blend',
      source_card: p.source_card ?? null,
      user_id: p.user_id ?? null,

      // keeping this field is fine, but DB uniqueness is enforced by the composite key
      idempotency_key: [
        p.target_date, p.lead_used, p.issuance_iso || '', p.model_name || '', p.version || ''
      ].join('|')
    };

    // 🔧 Align onConflict with your DB's unique constraint "uq_pred_logs"
       const { error } = await supabase
      .from('prediction_logs')
      .upsert(row, {
        onConflict: 'target_date,lead_used,issuance_iso,model_name,version'
      });

    if (error) {
      console.error('Supabase upsert error:', error);
      return res.status(500).json({ error: 'insert failed', detail: error.message || String(error) });
    }

    return res.status(201).json({ ok: true });
  } catch (e) {
    console.error('/api/log-prediction exception', e);
    return res.status(500).json({ error: 'insert failed', detail: e?.message || String(e) });
  }
}
