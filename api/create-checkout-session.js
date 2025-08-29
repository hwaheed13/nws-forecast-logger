// /api/create-checkout-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

export default async function handler(req, res) {
  // (Optional) CORS / preflight
  if (req.method === "OPTIONS") {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    return res.status(200).end();
  }
  res.setHeader("Access-Control-Allow-Origin", "*");
  if (req.method !== "POST") return res.status(405).end();

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { priceId: rawPriceId, plan, supabaseAccessToken, trial } = body;

    if (!supabaseAccessToken) {
      return res.status(400).json({ error: "Missing supabaseAccessToken" });
    }

    // User-scoped Supabase client
    const SUPABASE_URL = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
    const SUPABASE_ANON = process.env.SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

    const supabaseUser = createClient(SUPABASE_URL, SUPABASE_ANON, {
      global: { headers: { Authorization: `Bearer ${supabaseAccessToken}` } },
    });

    const { data: { user }, error: userErr } = await supabaseUser.auth.getUser();
    if (userErr || !user) return res.status(401).json({ error: "Not authenticated" });

    // Admin client (service role)
    const supabaseAdmin = createClient(SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

    // Get or create Stripe customer; upsert to profiles
    const { data: profile } = await supabaseAdmin
      .from("profiles")
      .select("id, email, stripe_customer_id, free_trial_used")
      .eq("id", user.id)
      .maybeSingle();

    let customerId = profile?.stripe_customer_id;
    if (!customerId) {
      const customer = await stripe.customers.create({
        email: user.email ?? profile?.email ?? undefined,
        metadata: { supabase_user_id: user.id },
      });
      customerId = customer.id;

      await supabaseAdmin.from("profiles").upsert({
        id: user.id,
        email: user.email ?? profile?.email ?? null,
        stripe_customer_id: customerId,
      });
    }

    // ── Free trial flow ────────────────────────────────────────────────────────
    if (trial) {
      // Server-side enforcement (even if UI hides the button)
      if (profile?.free_trial_used) {
        return res.status(400).json({ error: "trial_already_used" });
      }

      // Use your monthly price for the trial (or a dedicated TRIAL price if you prefer)
      const monthlyPriceId = process.env.PRICE_ID_MONTHLY;
      if (!monthlyPriceId) {
        return res.status(500).json({ error: "server_misconfigured: PRICE_ID_MONTHLY" });
      }

      const session = await stripe.checkout.sessions.create({
        mode: "subscription",
        customer: customerId,
        line_items: [{ price: monthlyPriceId, quantity: 1 }],
        // 3-day trial; Checkout will still collect a payment method
        subscription_data: {
          trial_period_days: 3,
          metadata: { free_trial: "true" },
        },
        allow_promotion_codes: true,
        success_url: `${process.env.DASHBOARD_URL}?checkout=success`,
        cancel_url: `${process.env.SUBSCRIBE_URL}?checkout=cancel`,
        metadata: { supabase_user_id: user.id, plan: "trial" },
      });

      return res.status(200).json({ id: session.id, url: session.url });
    }

    // ── Regular paid flow (monthly / yearly) ──────────────────────────────────
    const priceId =
      rawPriceId ||
      (plan === "monthly" ? process.env.PRICE_ID_MONTHLY :
       plan === "yearly"  ? process.env.PRICE_ID_YEARLY  : null);

    if (!priceId) {
      return res.status(400).json({ error: "Missing or invalid plan/priceId" });
    }

    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      customer: customerId,
      line_items: [{ price: priceId, quantity: 1 }],
      allow_promotion_codes: true,
      success_url: `${process.env.DASHBOARD_URL}?checkout=success`,
      cancel_url: `${process.env.SUBSCRIBE_URL}?checkout=cancel`,
      metadata: { supabase_user_id: user.id, plan: plan || "by_price_id" },
    });

    return res.status(200).json({ id: session.id, url: session.url });
  } catch (e) {
    console.error("create-checkout-session error:", e);
    return res.status(500).json({ error: "server_error" });
  }
}
