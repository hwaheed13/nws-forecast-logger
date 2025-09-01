// /api/create-checkout-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2024-06-20" });

const PRICES = {
  monthly: process.env.PRICE_ID_MONTHLY, // $3/mo
  yearly:  process.env.PRICE_ID_YEARLY,  // $30/yr
};

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SERVICE_ROLE = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_ROLE;
const supabaseAdmin = createClient(SUPABASE_URL, SERVICE_ROLE);

// ------------------------------
// Helpers
// ------------------------------
function getReturnURLs() {
  const DASHBOARD_ORIGIN = process.env.DASHBOARD_ORIGIN || "https://app.dailydewpoint.com";
  return {
    success_url: `${DASHBOARD_ORIGIN}/subscribe.html?success=1`,
    cancel_url:  `${DASHBOARD_ORIGIN}/subscribe.html?canceled=1`,
  };
}

async function verifySupabaseUser(accessToken) {
  if (!accessToken) return null;
  const { data, error } = await supabaseAdmin.auth.getUser(accessToken);
  if (error || !data?.user) return null;
  return { id: data.user.id, email: data.user.email };
}

async function ensureStripeCustomer({ userId, email }) {
  const { data: prof } = await supabaseAdmin
    .from("profiles")
    .select("stripe_customer_id")
    .eq("id", userId)
    .maybeSingle();

  if (prof?.stripe_customer_id) return prof.stripe_customer_id;

  const customer = await stripe.customers.create({
    email,
    metadata: { supabase_user_id: userId },
  });

  await supabaseAdmin
    .from("profiles")
    .update({ stripe_customer_id: customer.id, email })
    .eq("id", userId);

  return customer.id;
}

// ------------------------------
// Main handler
// ------------------------------
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { plan, trial, priceId, supabaseAccessToken } = body;

    // 1) Verify user
    const user = await verifySupabaseUser(supabaseAccessToken);
    if (!user) return res.status(401).json({ error: "Unauthorized" });

    // 2) Ensure Stripe customer
    const customerId = await ensureStripeCustomer({ userId: user.id, email: user.email });

    // 3) Get return URLs
    const { success_url, cancel_url } = getReturnURLs();

    // 4) Check if trial already used
    const { data: profile } = await supabaseAdmin
      .from("profiles")
      .select("free_trial_used, trial_used")
      .eq("id", user.id)
      .maybeSingle();

    const alreadyTrialed = !!(profile?.free_trial_used || profile?.trial_used);

    // 5) Trial path
    if (trial === true && !alreadyTrialed) {
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
          trial_period_days: 3, // free trial length
          metadata: {
            ddp_trial: "true",
            ddp_selected_plan: "monthly",
            supabase_user_id: user.id,
          },
        },
        metadata: {
          ddp_trial: "true",
          supabase_user_id: user.id,
          plan: "monthly",
        },
        allow_promotion_codes: false,
      });

      return res.status(200).json({ url: session.url });
    }

    // 6) Normal checkout path
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
