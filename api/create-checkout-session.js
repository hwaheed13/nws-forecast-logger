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

// --- helpers ---------------------------------------------------------------

function resolveSubscribeURLs(req) {
  // Use SUBSCRIBE_URL if set, else fall back to /subscribe.html on current host
  const base =
    process.env.SUBSCRIBE_URL ||
    `${(req.headers["x-forwarded-proto"] || "https")}://${req.headers.host}/subscribe.html`;
  const u = new URL(base);
  const qs = u.searchParams;
  qs.set("success", "1");
  const success_url = u.toString();
  u.searchParams.delete("success");
  u.searchParams.set("canceled", "1");
  const cancel_url = u.toString();
  return { success_url, cancel_url };
}

async function verifySupabaseUser(accessToken) {
  if (!accessToken) return null;
  const { data, error } = await supabaseAdmin.auth.getUser(accessToken);
  if (error || !data?.user) return null;
  return { id: data.user.id, email: data.user.email };
}

async function ensureStripeCustomer({ userId, email }) {
  // check mapping
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
    .update({ stripe_customer_id: c.id, email })
    .eq("id", userId);

  return c.id;
}

// --- main handler ----------------------------------------------------------

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { plan, trial, priceId, supabaseAccessToken } = body;

    // 1) Authn
    const user = await verifySupabaseUser(supabaseAccessToken);
    if (!user) return res.status(401).json({ error: "Unauthorized (missing/invalid Supabase token)" });

    // 2) Customer
    const customerId = await ensureStripeCustomer({ userId: user.id, email: user.email });

    // 3) Profile (to gate trial)
    const { data: profile } = await supabaseAdmin
      .from("profiles")
      .select("free_trial_used, trial_used") // support either flag if present
      .eq("id", user.id)
      .maybeSingle();

    const hasTrialFlag = !!(profile?.free_trial_used || profile?.trial_used);
    const { success_url, cancel_url } = resolveSubscribeURLs(req);

    // 4) TRIAL path — only if never used
    if (trial === true && !hasTrialFlag) {
      if (!PRICES.monthly) {
        return res.status(500).json({ error: "Monthly price is not configured on the server" });
      }

      const session = await stripe.checkout.sessions.create({
        mode: "subscription",
        customer: customerId,
        success_url,
        cancel_url,
        line_items: [{ price: PRICES.monthly, quantity: 1 }], // 3-day trial → then $3/mo
        subscription_data: {
          trial_period_days: 3,
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

    // 5) Normal plan path ($3 monthly or $30 yearly)
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
