// /api/create-checkout-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

// Allowed frontends
const ALLOW = new Set([
  "https://app.dailydewpoint.com",
  "http://localhost:3000",
]);

// Where Stripe should redirect after checkout
const DASHBOARD_ORIGIN =
  process.env.DASHBOARD_ORIGIN || "https://app.dailydewpoint.com";

export default async function handler(req, res) {
  // ---- CORS / preflight
  const origin = req.headers.origin || "";
  const allow = ALLOW.has(origin) ? origin : "https://app.dailydewpoint.com";
  res.setHeader("Access-Control-Allow-Origin", allow);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", req.headers["access-control-request-headers"] || "Content-Type, Authorization");
  if (req.method === "OPTIONS") return res.status(204).end();

  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { priceId: rawPriceId, plan, supabaseAccessToken } = body;

    if (!supabaseAccessToken) return res.status(400).json({ error: "Missing supabaseAccessToken" });

    const priceId =
      rawPriceId ||
      (plan === "monthly"
        ? process.env.PRICE_ID_MONTHLY
        : plan === "yearly"
        ? process.env.PRICE_ID_YEARLY
        : null);

    if (!priceId) return res.status(400).json({ error: "Missing or invalid plan/priceId" });

    // Verify caller as the logged-in user
    const supabaseUser = createClient(
      process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL,
      process.env.SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      { global: { headers: { Authorization: `Bearer ${supabaseAccessToken}` } } }
    );
    const { data: { user }, error: userErr } = await supabaseUser.auth.getUser();
    if (userErr || !user) return res.status(401).json({ error: "Not authenticated" });

    // Admin client to read/update profile
    const supabaseAdmin = createClient(
      process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    const { data: profile } = await supabaseAdmin
      .from("profiles")
      .select("id,email,stripe_customer_id")
      .eq("id", user.id)
      .single();

    let customerId = profile?.stripe_customer_id;

    if (!customerId) {
      const customer = await stripe.customers.create({
        email: user.email ?? profile?.email ?? undefined,
        metadata: { supabase_user_id: user.id },
      });
      customerId = customer.id;

      await supabaseAdmin
        .from("profiles")
        .upsert(
          { id: user.id, email: user.email ?? profile?.email ?? null, stripe_customer_id: customerId },
          { onConflict: "id" }
        );
    } else {
      // keep metadata in sync
      await stripe.customers.update(customerId, { metadata: { supabase_user_id: user.id } });
    }

    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      customer: customerId,
      line_items: [{ price: priceId, quantity: 1 }],
      allow_promotion_codes: true,
      success_url: `${DASHBOARD_ORIGIN}/subscribe.html?success=1`,
      cancel_url:  `${DASHBOARD_ORIGIN}/subscribe.html?canceled=1`,
      metadata: { supabase_user_id: user.id, plan: plan || "by_price_id" },
    });

    return res.status(200).json({ id: session.id, url: session.url });
  } catch (e) {
    console.error("create-checkout-session error:", e);
    return res.status(500).json({ error: "server_error" });
  }
}
