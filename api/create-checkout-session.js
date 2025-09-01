// /api/create-checkout-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2024-06-20" });

// Canonical price IDs from Vercel env
const PRICES = {
  monthly: process.env.PRICE_ID_MONTHLY, // $3/mo
  yearly:  process.env.PRICE_ID_YEARLY,  // $30/yr
};

const SUPABASE_URL  = process.env.SUPABASE_URL  || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SERVICE_ROLE  = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_ROLE;

// Server (service role) client
const supabaseAdmin = createClient(SUPABASE_URL, SERVICE_ROLE);

// ---------- helpers ----------
function successCancelFromEnv(req) {
  // Use SUBSCRIBE_URL (e.g. https://dailydewpoint.com/dashboard/subscribe.html)
  const subscribeURL =
    process.env.SUBSCRIBE_URL ||
    `${(req.headers["x-forwarded-proto"] || "https")}://${req.headers.host}/subscribe.html`;
  return {
    success_url: `${subscribeURL}?success=1`,
    cancel_url:  `${subscribeURL}?canceled=1`,
  };
}

async function verifySupabaseUser(accessToken) {
  if (!accessToken) return null;
  const { data, error } = await supabaseAdmin.auth.getUser(accessToken);
  if (error || !data?.user) return null;
  return { id: data.user.id, email: data.user.email };
}

async function ensureStripeCustomer({ userId, email }) {
  // look for existing mapping
  const { data: prof } = await supabaseAdmin
    .from("profiles")
    .select("stripe_customer_id")
    .eq("id", userId)
    .maybeSingle();

  if (prof?.stripe_customer_id) return prof.stripe_customer_id;

  // create + persist
  const c = await stripe.customers.create({
    email,
    metadata: { supabase_user_id: userId },
  });

  await supabaseAdmin
    .from("profiles")
    .update({ stripe_customer_id: c.id })
    .eq("id", userId);

  return c.id;
}

// ---------- handler ----------
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { plan, trial, priceId, supabaseAccessToken } = body;

    // 1) Verify Supabase user
    const user = await verifySupabaseUser(supabaseAccessToken);
    if (!user) return res.status(401).json({ error: "Unauthorized (missing/invalid Supabase token)" });

    // 2) Ensure Stripe customer
    const customerId = await ensureStripeCustomer({ userId: user.id, email: user.email });

    // 3) Read current profile to gate trial
    const { data: profile, error: profileErr } = await supabaseAdmin
      .from("profiles")
      .select("has_taken_trial, subscription_status")
      .eq("id", user.id)
      .maybeSingle();

    if (profileErr) {
      console.error("profiles read error:", profileErr);
      return res.status(500).json({ error: "Server profile read error" });
    }

    const alreadyTookTrial = !!profile?.has_taken_trial;
    const { success_url, cancel_url } = successCancelFromEnv(req);

    // 4) TRIAL PATH — only if user never used one
    if (trial === true) {
      if (alreadyTookTrial) {
        // fall through to normal plan (monthly by default)
        console.log(`User ${user.id} attempted second trial — falling back to normal checkout`);
      } else {
        if (!PRICES.monthly) {
          return res.status(500).json({ error: "Monthly price is not configured on the server" });
        }

        const session = await stripe.checkout.sessions.create({
          mode: "subscription",
          customer: customerId,
          success_url,
          cancel_url,
          line_items: [{ price: PRICES.monthly, quantity: 1 }], // trial → then $3/mo unless canceled
          subscription_data: {
            trial_period_days: 3, // << FREE TRIAL LENGTH
            metadata: {
              ddp_trial: "true",
              ddp_selected_plan: "monthly",
              supabase_user_id: user.id,
            },
          },
          metadata: {
            ddp_trial: "true",
            supabase_user_id: user.id,
          },
          allow_promotion_codes: false,
        });

        return res.status(200).json({ url: session.url });
      }
    }

    // 5) NORMAL PLAN PATH
    const chosen =
      priceId
        ? priceId
        : plan === "yearly"
          ? PRICES.yearly
          : PRICES.monthly;

    if (!chosen) return res.status(400).json({ error: "Missing or invalid plan/priceId" });

    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      customer: customerId,
      success_url,
      cancel_url,
      line_items: [{ price: chosen, quantity: 1 }],
      allow_promotion_codes: true,
      metadata: {
        supabase_user_id: user.id,
        ddp_trial: "false",
        plan: plan || (chosen === PRICES.yearly ? "yearly" : "monthly"),
      },
    });

    return res.status(200).json({ url: session.url });
  } catch (err) {
    console.error("create-checkout-session error:", err);
    return res.status(500).json({ error: err.message || "Server error" });
  }
}
