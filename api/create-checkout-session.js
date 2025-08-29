// /api/create-checkout-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

export default async function handler(req, res) {
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

    if (!supabaseAccessToken) return res.status(400).json({ error: "Missing supabaseAccessToken" });

    // User-scoped Supabase
    const supabaseUser = createClient(
      process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL,
      process.env.SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      { global: { headers: { Authorization: `Bearer ${supabaseAccessToken}` } } }
    );
    const { data: { user }, error: userErr } = await supabaseUser.auth.getUser();
    if (userErr || !user) return res.status(401).json({ error: "Not authenticated" });

    // Admin Supabase
    const supabaseAdmin = createClient(
      process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    // Fetch or create Stripe customer
    const { data: profile } = await supabaseAdmin
      .from("profiles")
      .select("id,email,stripe_customer_id,subscription_status,free_trial_used")
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

    // === Trial flow ===========================================================
    if (trial) {
      // Hard blocks: already active OR already has a live Stripe sub
      if (profile?.subscription_status === "active") {
        return res.status(400).json({ error: "already_subscribed" });
      }
      const existing = await stripe.subscriptions.list({ customer: customerId, status: "all", limit: 10 });
      const hasLive = existing.data.some(s =>
        ["active","trialing","past_due","unpaid"].includes(s.status) && !s.cancel_at_period_end
      );
      if (hasLive) return res.status(400).json({ error: "already_subscribed" });

      // One trial per user
      if (profile?.free_trial_used) {
        return res.status(400).json({ error: "trial_already_used" });
      }

      const monthlyPrice = process.env.PRICE_ID_MONTHLY || rawPriceId;
      if (!monthlyPrice) return res.status(400).json({ error: "Missing base price for trial" });

      const session = await stripe.checkout.sessions.create({
        mode: "subscription",
        customer: customerId,
        line_items: [{ price: monthlyPrice, quantity: 1 }],
        subscription_data: {
          trial_period_days: 3,
          metadata: { supabase_user_id: user.id, plan: "trial" }
        },
        allow_promotion_codes: false,
        success_url: `${process.env.DASHBOARD_URL}?checkout=success`,
        cancel_url: `${process.env.SUBSCRIBE_URL}?checkout=cancel`,
        metadata: { supabase_user_id: user.id, plan: "trial" }
      });

      return res.status(200).json({ id: session.id, url: session.url });
    }

    // === Paid flow ============================================================
    const priceId =
      rawPriceId ||
      (plan === "monthly" ? process.env.PRICE_ID_MONTHLY
       : plan === "yearly" ? process.env.PRICE_ID_YEARLY
       : null);
    if (!priceId) return res.status(400).json({ error: "Missing or invalid plan/priceId" });

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
