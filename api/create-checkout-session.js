// pages/api/create-checkout-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2024-06-20" });

// Support either env name to avoid mismatch
const SUPABASE_SERVICE_ROLE =
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_ROLE;

const supabaseAdmin = createClient(
  process.env.SUPABASE_URL,
  SUPABASE_SERVICE_ROLE
);

// Prices: $3 monthly and $30 yearly
const PRICES = {
  monthly: process.env.STRIPE_PRICE_MONTHLY_3USD,
  yearly:  process.env.STRIPE_PRICE_YEARLY_30USD,
};

// ---- helpers ---------------------------------------------------------------

// On Vercel, prefer Origin header; fallback to https:// + Host.
function getOrigin(req) {
  const o = req.headers.origin;
  if (o && /^https?:\/\//i.test(o)) return o;
  const host = req.headers["x-forwarded-host"] || req.headers.host;
  return host ? `https://${host}` : "https://app.dailydewpoint.com";
}

async function verifySupabaseUser(accessToken) {
  if (!accessToken) return null;
  // Use service-role client to validate the token and read the user
  const { data, error } = await supabaseAdmin.auth.getUser(accessToken);
  if (error || !data?.user) return null;
  return { id: data.user.id, email: data.user.email };
}

async function ensureStripeCustomer({ userId, email }) {
  // Check mapping in profiles
  const { data: prof } = await supabaseAdmin
    .from("profiles")
    .select("stripe_customer_id")
    .eq("id", userId)
    .maybeSingle();

  if (prof?.stripe_customer_id) return prof.stripe_customer_id;

  // Create new customer and store it
  const customer = await stripe.customers.create({
    email,
    metadata: { supabase_user_id: userId },
  });

  await supabaseAdmin
    .from("profiles")
    .update({ stripe_customer_id: customer.id })
    .eq("id", userId);

  return customer.id;
}

// ---- handler ---------------------------------------------------------------

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    // Accept either raw object or JSON string (some clients send a string)
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { plan, trial, priceId, supabaseAccessToken } = body;

    // 1) Validate user via Supabase
    const user = await verifySupabaseUser(supabaseAccessToken);
    if (!user) {
      return res.status(401).json({ error: "Unauthorized (invalid or missing Supabase token)" });
    }

    // 2) Ensure Stripe customer exists and is linked
    const customerId = await ensureStripeCustomer({ userId: user.id, email: user.email });

    // 3) Build success/cancel URLs
    const origin = getOrigin(req);
    const success_url = `${origin}/subscribe.html?success=1`;
    const cancel_url  = `${origin}/subscribe.html?canceled=1`;

    // 4) FREE TRIAL PATH (no plan choice required)
    //    Starts a 7-day trial, then rolls into the MONTHLY $3 plan unless canceled.
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
          trial_period_days: 7, // <-- adjust if you want a different trial length
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

    // 5) NORMAL PLAN PATH ($3 monthly or $30 yearly). Allows coupon codes.
    const chosen =
      priceId || (plan === "yearly" ? PRICES.yearly : PRICES.monthly);

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
    console.error("create-checkout-session error", err);
    return res.status(500).json({ error: err.message || "Server error" });
  }
}
