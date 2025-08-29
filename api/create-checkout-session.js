// /api/create-checkout-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, {
  apiVersion: "2023-10-16",
});

export default async function handler(req, res) {
  // (Optional) CORS — helps if you ever call from a different origin or via preflight
  if (req.method === "OPTIONS") {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    return res.status(200).end();
  }
  res.setHeader("Access-Control-Allow-Origin", "*");

  if (req.method !== "POST") return res.status(405).end();

  try {
    // Be tolerant: sometimes req.body is a string
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { priceId: rawPriceId, plan, supabaseAccessToken } = body;

    // Helpful logs while debugging (remove later)
    console.log("BODY:", body);
    console.log("plan:", plan, "rawPriceId:", rawPriceId ? "[present]" : null);

    if (!supabaseAccessToken) {
      return res.status(400).json({ error: "Missing supabaseAccessToken" });
    }

    // Prefer explicit priceId; fall back to envs via plan
    const priceId =
      rawPriceId ||
      (plan === "monthly"
        ? process.env.PRICE_ID_MONTHLY
        : plan === "yearly"
        ? process.env.PRICE_ID_YEARLY
        : null);

    if (!priceId) {
      console.error("Resolved priceId is empty. plan:", plan, "rawPriceId:", rawPriceId);
      return res.status(400).json({ error: "Missing or invalid plan/priceId" });
    }

    // User-scoped supabase client (auth)
    const supabaseUser = createClient(
      process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL,
      process.env.SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      { global: { headers: { Authorization: `Bearer ${supabaseAccessToken}` } } }
    );

    const {
      data: { user },
      error: userErr,
    } = await supabaseUser.auth.getUser();
    if (userErr || !user) return res.status(401).json({ error: "Not authenticated" });

    // Admin supabase (service role — server only!)
    const supabaseAdmin = createClient(
      process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    // Fetch profile (may not exist yet)
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
