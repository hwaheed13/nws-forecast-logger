// create-checkout-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2024-06-20" });

const PRICES = {
  monthly: process.env.STRIPE_PRICE_MONTHLY_3USD,   // e.g. price_1S1...
  yearly:  process.env.STRIPE_PRICE_YEARLY_30USD,   // e.g. price_1S1...
};

const supabaseAdmin = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE
);

// --- helpers ---------------------------------------------------------------

function originFrom(req) {
  return `${req.protocol}://${req.get("host")}`;
}

/** Validate the Supabase access token and return { id, email } or null */
async function verifySupabaseUser(accessToken) {
  if (!accessToken) return null;
  const { data, error } = await supabaseAdmin.auth.getUser(accessToken);
  if (error || !data?.user) return null;
  return { id: data.user.id, email: data.user.email };
}

/** Ensure we have a Stripe customer for this Supabase user, and persist id on profile */
async function ensureStripeCustomer({ userId, email }) {
  // Check existing mapping on profiles
  const { data: prof } = await supabaseAdmin
    .from("profiles")
    .select("stripe_customer_id")
    .eq("id", userId)
    .maybeSingle();

  if (prof?.stripe_customer_id) return prof.stripe_customer_id;

  // Create
  const customer = await stripe.customers.create({
    email,
    metadata: { supabase_user_id: userId }
  });

  // Persist
  await supabaseAdmin
    .from("profiles")
    .update({ stripe_customer_id: customer.id })
    .eq("id", userId);

  return customer.id;
}

// --- main handler ----------------------------------------------------------

export async function createCheckoutSession(req, res) {
  try {
    const { plan, trial, priceId, supabaseAccessToken } = req.body;

    // 1) Verify Supabase user (required)
    const user = await verifySupabaseUser(supabaseAccessToken);
    if (!user) {
      return res.status(401).json({ error: "Unauthorized (invalid or missing Supabase token)" });
    }

    // 2) Ensure Stripe customer
    const customerId = await ensureStripeCustomer({ userId: user.id, email: user.email });

    const origin = originFrom(req);
    const success_url = `${origin}/subscribe.html?success=1`;
    const cancel_url  = `${origin}/subscribe.html?canceled=1`;

    // 3) TRIAL PATH (no plan required): free trial, then auto-roll to MONTHLY $3
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
          trial_period_days: 7, // adjust trial length as you wish
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
        allow_promotion_codes: false
      });

      return res.json({ url: session.url });
    }

    // 4) NORMAL PLAN PATH ($3 monthly or $30 yearly)
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
      metadata: { supabase_user_id: user.id, ddp_trial: "false" }
    });

    res.json({ url: session.url });
  } catch (err) {
    console.error("create-checkout-session error:", err);
    res.status(500).json({ error: err.message || "Server error" });
  }
}
