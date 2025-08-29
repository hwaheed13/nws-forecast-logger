// /api/create-checkout-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  try {
    const { plan, supabaseAccessToken } = req.body; // client sends session access_token
    if (!plan || !supabaseAccessToken) return res.status(400).json({ error: "Missing plan or token" });

    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      { global: { headers: { Authorization: `Bearer ${supabaseAccessToken}` } } }
    );

    const { data: { user }, error: userErr } = await supabase.auth.getUser();
    if (userErr || !user) return res.status(401).json({ error: "Not authenticated" });

    // find (or create) stripe customer
    const admin = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
    const { data: profile } = await admin.from("profiles").select("*").eq("id", user.id).single();

    let customerId = profile?.stripe_customer_id;
    if (!customerId) {
      const customer = await stripe.customers.create({
        email: user.email,
        metadata: { supabase_user_id: user.id }
      });
      customerId = customer.id;
      await admin.from("profiles").upsert({ id: user.id, stripe_customer_id: customerId, email: user.email });
    }

    const priceId = plan === "monthly" ? process.env.PRICE_ID_MONTHLY
                 : plan === "yearly"  ? process.env.PRICE_ID_YEARLY
                 : null;
    if (!priceId) return res.status(400).json({ error: "Invalid plan" });

    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      customer: customerId,
      line_items: [{ price: priceId, quantity: 1 }],
      success_url: `${process.env.DASHBOARD_URL}?checkout=success`,
      cancel_url: `${process.env.SUBSCRIBE_URL}?checkout=cancel`,
      metadata: { supabase_user_id: user.id, plan },
      allow_promotion_codes: true
    });

    return res.status(200).json({ id: session.id, url: session.url });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "server_error" });
  }
}
