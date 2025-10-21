// /api/create-portal-session.js
export default async function handler(req, res) {
  // marker so you can confirm this version is live (Network → Headers)
  res.setHeader("x-ddp-portal-route", "v7");

  // CORS / preflight (same-origin safe)
  const origin = req.headers.origin || "";
  const allow =
    origin === "https://app.dailydewpoint.com" || origin === "http://localhost:3000"
      ? origin
      : "https://app.dailydewpoint.com";
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
    // ---- dynamic imports to avoid ESM/CJS quirks during cold starts
    const { default: Stripe } = await import("stripe");
    const { createClient } = await import("@supabase/supabase-js");

    // ---- env (validated inside try so errors return JSON)
    const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY;
    if (!STRIPE_SECRET_KEY) throw new Error("Missing env var: STRIPE_SECRET_KEY");

    const SUPABASE_URL = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
    if (!SUPABASE_URL) throw new Error("Missing env var: SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL)");

    const SERVICE_ROLE = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_ROLE;
    if (!SERVICE_ROLE) throw new Error("Missing env var: SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_SERVICE_ROLE)");

    // prefer DASHBOARD_ORIGIN, then DASHBOARD_URL, then default to app domain
    const DASHBOARD_ORIGIN =
      process.env.DASHBOARD_ORIGIN || process.env.DASHBOARD_URL || "https://app.dailydewpoint.com";

    // ---- parse body safely
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { supabaseAccessToken, returnTo } = body;
    if (!supabaseAccessToken) return res.status(400).json({ error: "Missing token" });

    // ---- init clients (after envs are verified)
    const stripe = new Stripe(STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });
    const admin = createClient(SUPABASE_URL, SERVICE_ROLE);

    // ---- verify user using admin (robust)
    const { data: userRes, error: userErr } = await admin.auth.getUser(supabaseAccessToken);
    if (userErr || !userRes?.user) return res.status(401).json({ error: "Not authenticated" });
    const user = userRes.user;

    // ---- fetch profile (don’t throw if missing)
    const { data: profile } = await admin
      .from("profiles")
      .select("id,email,stripe_customer_id")
      .eq("id", user.id)
      .maybeSingle();

    // ---- ensure Stripe customer
    let customerId = profile?.stripe_customer_id || null;
    const email = user.email ?? profile?.email ?? undefined;

    if (!customerId) {
      if (email) {
        const list = await stripe.customers.list({ email, limit: 1 });
        if (list.data[0]) customerId = list.data[0].id;
      }
      if (!customerId) {
        const c = await stripe.customers.create({
          email, // allowed to be undefined
          metadata: { supabase_user_id: user.id },
        });
        customerId = c.id;
      }
      // best-effort persist (don’t fail request if blocked)
      try {
        await admin
          .from("profiles")
          .upsert({ id: user.id, email: email ?? null, stripe_customer_id: customerId }, { onConflict: "id" });
      } catch {}
    } else {
      // keep metadata in sync (best-effort)
      try {
        await stripe.customers.update(customerId, { metadata: { supabase_user_id: user.id } });
      } catch {}
    }

    // ---- only allow your domains for overrides
    const safeReturn = (() => {
      try {
        if (!returnTo) return null;
        const u = new URL(returnTo);
        return (u.hostname === "app.dailydewpoint.com" || u.hostname === "dailydewpoint.com") ? u.toString() : null;
      } catch {
        return null;
      }
    })();

    // ✅ send users back to the site root (home)
    const return_url = safeReturn || `${DASHBOARD_ORIGIN}/`;

    // ---- create Billing Portal session
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
