// /api/create-checkout-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2024-06-20" });

// Match your Vercel env var names
const PRICES = {
  monthly: process.env.PRICE_ID_MONTHLY,  // e.g. price_1S1U96...
  yearly:  process.env.PRICE_ID_YEARLY,   // e.g. price_1S1U7h...
};

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_ROLE = process.env.SUPABASE_SERVICE_ROLE || process.env.SUPABASE_SERVICE_ROLE_KEY;

// Admin client (service role) to read/write profiles table
const supabaseAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE);

// ---- helpers ---------------------------------------------------------------

function originFrom(req) {
  // Prefer your configured subscribe URL if present
  const envSub = process.env.SUBSCRIBE_URL || process.env.NEXT_PUBLIC_SITE_URL || "";
  try {
    if (envSub) return new URL(envSub).origin;
  } catch (_) {}
  return `${req.headers["x-forwarded-proto"] || "https"}://${req.headers.host}`;
}

/** Validate Supabase access token and return { id, email } or null */
async function verifySupabaseUser(accessToken) {
  if (!accessToken) return null;
  const { data, error } = await supabaseAdmin.auth.getUser(accessToken);
  if (error || !data?.user) return null;
  return { id: data.user.id, email: data.user.email };
}

/** Ensure Stripe customer exists and is stored on profiles.stripe_customer_id */
async function ensureStripeCustomer({ userId, email }) {
  // Check existing mapping
  const { data: prof } = await supabaseAdmin
    .from("profiles")
    .select("stripe_customer_id")
    .eq("id", userId)
    .maybeSingle();

  if (prof?.stripe_customer_id) {
    // refresh metadata association (best-effort)
    await stripe.customers.update(prof.stripe_customer_id, {
      metadata: { supabase_user_id: userId },
    }).catch(() => {});
    return prof.stripe_customer_id;
  }

  // Create new Stripe customer
  const customer = await stripe.customers.create({
    email,
    metadata: { supabase_user_id: userId },
  });

  // Persist on profile
  await supabaseAdmin
    .from("profiles")
    .update({ stripe_customer_id: customer.id })
    .eq("id", userId);

  return customer.id;
}

// ---- main handler ----------------------------------------------------------

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { plan, trial, priceId, supabaseAccessToken } = body;

    // 1) Verify Supabase user (required)
    const user = await verifySupabaseUser(supabaseAccessToken);
    if (!user) {
      return res.status(401).json({ error: "Unauthorized (invalid or missing Supabase token)" });
    }

    // 2) Ensure we have a Stripe customer id stored on the profile
    const customerId = await ensureStripeCustomer({ userId: user.id, email: user.email });

    const origin = originFrom(req);
    const success_url = `${origin}/subscribe.html?success=1`;
    const cancel_url  = `${origin}/subscribe.html?canceled=1`;

    // 3) FREE TRIAL PATH (no plan required)
    // Rolls into MONTHLY ($3) automatically after 3 days unless canceled.
    if (trial === true) {
      if (!PRICES.monthly) {
        return res.status(500).json({ error: "Monthly price is not configured on the server" });
      }

      const session = await stripe.checkout.sessions.create({
        mode: "subscription",
        customer: customerId,
        success_url,
        cancel_url,
        line_items: [{ price: PRICES.monthly, quantity: 1 }],
        subscription_data: {
          trial_period_days: 3,  // <-- 3-day free trial
          metadata: {
            ddp_trial: "true",
            ddp_selected_plan: "monthly",
            supabase_user_id: user.id
          }
        },
        metadata: {
          ddp_trial: "true",
          supabase_user_id: user.id
        },
        allow_promotion_codes: false,
      });

      return res.status(200).json({ url: session.url });
    }

    // 4) NORMAL PLAN PATH ($3 monthly or $30 yearly)
    // If the client sends priceId explicitly, we use it; otherwise we map by `plan`.
    const chosen =
      priceId
        ? priceId
        : plan === "yearly"
          ? PRICES.yearly
          : PRICES.monthly;

    if (!chosen) {
      return res.status(400).json({ error: "Missing or invalid plan/priceId" });
    }

    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      customer: customerId,
      success_url,
      cancel_url,
      line_items: [{ price: chosen, quantity: 1 }],
      allow_promotion_codes: true,
      metadata: { supabase_user_id: user.id, ddp_trial: "false" },
    });

    return res.status(200).json({ url: session.url });
  } catch (err) {
    console.error("create-checkout-session error:", err);
    return res.status(500).json({ error: err.message || "Server error" });
  }
}
