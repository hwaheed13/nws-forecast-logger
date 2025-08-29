// /api/create-portal-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

export default async function handler(req, res) {
  // Optional CORS (fine to keep even if same-origin)
  if (req.method === "OPTIONS") {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    return res.status(200).end();
  }
  res.setHeader("Access-Control-Allow-Origin", "*");

  if (req.method !== "POST") return res.status(405).end();

  try {
    // Be tolerant if body is a string
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { supabaseAccessToken, returnTo } = body;
    if (!supabaseAccessToken) return res.status(400).json({ error: "Missing token" });

    // Prefer server-only names; fall back to NEXT_PUBLIC if that’s what you set
    const SUPABASE_URL  = process.env.SUPABASE_URL  || process.env.NEXT_PUBLIC_SUPABASE_URL;
    const SUPABASE_ANON = process.env.SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

    // User-scoped (auth)
    const supabaseUser = createClient(SUPABASE_URL, SUPABASE_ANON, {
      global: { headers: { Authorization: `Bearer ${supabaseAccessToken}` } }
    });
    const { data: { user } } = await supabaseUser.auth.getUser();
    if (!user) return res.status(401).json({ error: "Not authenticated" });

    // Admin (service role)
    const supabaseAdmin = createClient(SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
    const { data: profile } = await supabaseAdmin
      .from("profiles")
      .select("id,email,stripe_customer_id")
      .eq("id", user.id)
      .single();

    let customerId = profile?.stripe_customer_id;

    // If missing, try to find by email in Stripe; otherwise create, then persist
    if (!customerId) {
      const email = user.email ?? profile?.email ?? undefined;

      if (email) {
        const list = await stripe.customers.list({ email, limit: 1 });
        if (list.data[0]) customerId = list.data[0].id;
      }

      if (!customerId) {
        const c = await stripe.customers.create({
          email,
          metadata: { supabase_user_id: user.id }
        });
        customerId = c.id;
      }

      await supabaseAdmin.from("profiles").upsert({
        id: user.id,
        email: email ?? null,
        stripe_customer_id: customerId
      });
    }

    // Create portal session
    const portal = await stripe.billingPortal.sessions.create({
      customer: customerId,
      return_url: returnTo || process.env.DASHBOARD_URL || "https://waheedweather.dewdropventures.com/index.html"
    });

    return res.status(200).json({ url: portal.url });
  } catch (e) {
    console.error("create-portal-session error:", e);
    return res.status(500).json({ error: "server_error" });
  }
}
