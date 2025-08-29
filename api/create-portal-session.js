// /api/create-portal-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  try {
    const { supabaseAccessToken } = req.body;
    if (!supabaseAccessToken) return res.status(400).json({ error: "Missing token" });

    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      { global: { headers: { Authorization: `Bearer ${supabaseAccessToken}` } } }
    );
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return res.status(401).json({ error: "Not authenticated" });

    const admin = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
    const { data: profile } = await admin.from("profiles").select("stripe_customer_id").eq("id", user.id).single();
    if (!profile?.stripe_customer_id) return res.status(400).json({ error: "No Stripe customer" });

    const portal = await stripe.billingPortal.sessions.create({
      customer: profile.stripe_customer_id,
      return_url: process.env.DASHBOARD_URL
    });

    return res.status(200).json({ url: portal.url });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "server_error" });
  }
}
