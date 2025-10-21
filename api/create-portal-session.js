// /api/create-portal-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

// --- Strict env checks (fail loud + clear)
const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY;
if (!STRIPE_SECRET_KEY) throw new Error("Missing env var: STRIPE_SECRET_KEY");

const SUPABASE_URL  = process.env.SUPABASE_URL  || process.env.NEXT_PUBLIC_SUPABASE_URL;
if (!SUPABASE_URL)  throw new Error("Missing env var: SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL)");

const SERVICE_ROLE  = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_ROLE;
if (!SERVICE_ROLE)  throw new Error("Missing env var: SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_SERVICE_ROLE)");

// Optional, only affects return_url; won’t crash if absent:
const DASHBOARD_ORIGIN = process.env.DASHBOARD_ORIGIN || process.env.DASHBOARD_URL || "https://app.dailydewpoint.com";

const stripe = new Stripe(STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

// defense-in-depth: only allow returns to your domains
function safeReturnUrl(raw) {
  try {
    if (!raw) return null;
    const u = new URL(raw);
    if (u.hostname === "app.dailydewpoint.com" || u.hostname === "dailydewpoint.com") return u.toString();
    return null;
  } catch { return null; }
}

export default async function handler(req, res) {
  // Mark the version so we know this code is live
  res.setHeader("x-ddp-portal-route", "v4");

  // ---- CORS / preflight (same-origin fetch from your app, safe)
  const origin = req.headers.origin || "";
  const allow = origin && (origin === "https://app.dailydewpoint.com" || origin === "http://localhost:3000")
    ? origin
    : "https://app.dailydewpoint.com";
  res.setHeader("Access-Control-Allow-Origin", allow);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", req.headers["access-control-request-headers"] || "Content-Type, Authorization");
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    // Parse body safely
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { supabaseAccessToken, returnTo } = body;
    if (!supabaseAccessToken) return res.status(400).json({ error: "Missing token" });

    // --- Supabase admin client (deterministic auth)
    const admin = createClient(SUPABASE_URL, SERVICE_ROLE);

    // 1) Verify the token / resolve the user (admin is more robust than anon+headers)
    const { data: userRes, error: userErr } = await admin.auth.getUser(supabaseAccessToken);
    if (userErr || !userRes?.user) return res.status(401).json({ error: "Not authenticated" });
    const user = userRes.user;

    // 2) Get the profile if it exists (don’t crash if it doesn’t)
    const { data: profile } = await admin
      .from("profiles")
      .select("id,email,stripe_customer_id")
      .eq("id", user.id)
      .maybeSingle();

    // 3) Ensure there is a Stripe customer (search by email -> create if missing)
    let customerId = profile?.stripe_customer_id || null;

    // Preferred email source order
    const email = user.email ?? profile?.email ?? undefined;

    if (!customerId) {
      if (email) {
        // Try to reuse an existing Stripe customer for this email
        const list = await stripe.customers.list({ email, limit: 1 });
        if (list.data[0]) customerId = list.data[0].id;
      }
      if (!customerId) {
        const c = await stripe.customers.create({
          email, // may be undefined – Stripe allows that
          metadata: { supabase_user_id: user.id },
        });
        customerId = c.id;
      }
      // Best-effort: persist in profiles (don’t crash if table missing or RLS blocks it)
      await admin
        .from("profiles")
        .upsert({ id: user.id, email: email ?? null, stripe_customer_id: customerId }, { onConflict: "id" })
        .then(() => {})
        .catch(() => {});
    } else {
      // Keep metadata in sync (best-effort)
      await stripe.customers.update(customerId, { metadata: { supabase_user_id: user.id } }).catch(() => {});
    }

    const return_url = safeReturnUrl(returnTo) || `${DASHBOARD_ORIGIN}/account`;

    // 4) Create the portal session
    const session = await stripe.billingPortal.sessions.create({
      customer: customerId,
      return_url,
    });

    return res.status(200).json({ url: session.url });
  } catch (e) {
    const msg = e?.message || String(e);
    console.error("create-portal-session error:", msg);
    return res.status(500).json({ error: msg });
  }
}
