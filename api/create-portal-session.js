// /api/create-portal-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

// Lazy Stripe init so missing envs throw inside try/catch
function getStripe() {
  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) throw new Error("Missing env var: STRIPE_SECRET_KEY");
  return new Stripe(key, { apiVersion: "2023-10-16" });
}

// Allow-listed origins (same-origin calls from your app)
const ALLOW = new Set(["https://app.dailydewpoint.com", "http://localhost:3000"]);

// Where the Stripe Billing Portal "Return" button should land
const DASHBOARD_ORIGIN =
  process.env.DASHBOARD_ORIGIN || process.env.DASHBOARD_URL || "https://app.dailydewpoint.com";

// Only allow return URLs on your domains
function safeReturnUrl(raw) {
  try {
    if (!raw) return null;
    const u = new URL(raw);
    if (u.hostname === "app.dailydewpoint.com" || u.hostname === "dailydewpoint.com") return u.toString();
    return null;
  } catch {
    return null;
  }
}

export default async function handler(req, res) {
  // Marker header so you can confirm the new code is live in Network → Headers
  res.setHeader("x-ddp-portal-route", "v5");

  // CORS / preflight (safe for same-origin fetch)
  const origin = req.headers.origin || "";
  const allow = ALLOW.has(origin) ? origin : "https://app.dailydewpoint.com";
  res.setHeader("Access-Control-Allow-Origin", allow);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    req.headers["access-control-request-headers"] || "Content-Type, Authorization"
  );
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    // --- Strict env checks (inside try so errors are returned as JSON)
    const SUPABASE_URL = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
    const SERVICE_ROLE = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_ROLE;
    if (!SUPABASE_URL) throw new Error("Missing env var: SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL)");
    if (!SERVICE_ROLE) throw new Error("Missing env var: SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_SERVICE_ROLE)");

    const stripe = getStripe();
    const admin = createClient(SUPABASE_URL, SERVICE_ROLE);

    // Parse JSON body safely
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { supabaseAccessToken, returnTo } = body;
    if (!supabaseAccessToken) return res.status(400).json({ error: "Missing token" });

    // 1) Verify the user via admin (most robust)
    const { data: userRes, error: userErr } = await admin.auth.getUser(supabaseAccessToken);
    if (userErr || !userRes?.user) return res.status(401).json({ error: "Not authenticated" });
    const user = userRes.user;

    // 2) Fetch profile (don’t throw if absent)
    const { data: profile } = await admin
      .from("profiles")
      .select("id,email,stripe_customer_id")
      .eq("id", user.id)
      .maybeSingle();

    // 3) Ensure Stripe customer exists (reuse by email → create if needed)
    let customerId = profile?.stripe_customer_id || null;
    const email = user.email ?? profile?.email ?? undefined;

    if (!customerId) {
      if (email) {
        const list = await stripe.customers.list({ email, limit: 1 });
        if (list.data[0]) customerId = list.data[0].id;
      }
      if (!customerId) {
        const c = await stripe.customers.create({
          email, // may be undefined
          metadata: { supabase_user_id: user.id },
        });
        customerId = c.id;
      }
      // Best-effort persist (don’t fail the request if this write has issues)
      try {
        await admin
          .from("profiles")
          .upsert({ id: user.id, email: email ?? null, stripe_customer_id: customerId }, { onConflict: "id" });
      } catch {}
    } else {
      // Keep metadata in sync (best-effort)
      try {
        await stripe.customers.update(customerId, { metadata: { supabase_user_id: user.id } });
      } catch {}
    }

    // 4) Create portal session
    const return_url = safeReturnUrl(returnTo) || `${DASHBOARD_ORIGIN}/account`;
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
