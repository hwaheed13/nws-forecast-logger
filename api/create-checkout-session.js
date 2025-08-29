// /api/create-checkout-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  try {
    const { priceId: rawPriceId, plan, supabaseAccessToken } = req.body || {};
    if (!supabaseAccessToken) return res.status(400).json({ error: "Missing supabaseAccessToken" });

    // Map plan -> price, and allow-list any explicit priceId coming from client
    const priceFromPlan =
      plan === "monthly" ? process.env.PRICE_ID_MONTHLY :
      plan === "yearly"  ? process.env.PRICE_ID_YEARLY  : null;

    const ALLOWED_PRICES = new Set([process.env.PRICE_ID_MONTHLY, process.env.PRICE_ID_YEARLY]);
    const priceId = rawPriceId ? (ALLOWED_PRICES.has(rawPriceId) ? rawPriceId : null) : priceFromPlan;

    if (!priceId) return res.status(400).json({ error: "Missing or invalid plan/priceId" });

    // User-scoped client (reads session)
    const supabaseUser = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL,          // <- use NEXT_PUBLIC_* here
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,     // <-
      { global: { headers: { Authorization: `Bearer ${supabaseAccessToken}` } } }
    );

    const { data: { user }, error: userErr } = await supabaseUser.auth.getUser();
    if (userErr || !user) return res.status(401).json({ error: "Not authenticated" });

    // Admin client (service role) — server-only
    if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
      console.error("Missing SUPABASE_SERVICE_ROLE_KEY");
      return res.status(500).json({ error: "Server misconfigured" });
    }
    const supabaseAdmin = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    const { data: profile } = await supabaseAdmin
      .from("profiles")
      .select("id, email, stripe_customer_id")
      .eq("id", user.id)
      .single();

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
