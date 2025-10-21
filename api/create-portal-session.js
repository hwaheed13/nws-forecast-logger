// /api/create-portal-session.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

const ALLOW = new Set([
  "https://app.dailydewpoint.com",
  "http://localhost:3000",
]);

// default where the Billing Portal's "Return" button goes
const DASHBOARD_ORIGIN =
  process.env.DASHBOARD_ORIGIN || "https://app.dailydewpoint.com";

// defense-in-depth: only allow returns to your domains
function safeReturnUrl(raw) {
  try {
    if (!raw) return null;
    const u = new URL(raw);
    if (u.hostname === "app.dailydewpoint.com" || u.hostname === "dailydewpoint.com") {
      return u.toString();
    }
    return null;
  } catch {
    return null;
  }
}

export default async function handler(req, res) {
  // ---- CORS / preflight
  const origin = req.headers.origin || "";
  const allow = ALLOW.has(origin) ? origin : "https://app.dailydewpoint.com";
  res.setHeader("Access-Control-Allow-Origin", allow);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", req.headers["access-control-request-headers"] || "Content-Type, Authorization");
  if (req.method === "OPTIONS") return res.status(204).end();

  if (req.method !== "POST") return res.status(405).end();

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { supabaseAccessToken, returnTo } = body;
    if (!supabaseAccessToken) return res.status(400).json({ error: "Missing token" });

    const SUPABASE_URL  = process.env.SUPABASE_URL  || process.env.NEXT_PUBLIC_SUPABASE_URL;
    const SUPABASE_ANON = process.env.SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

    // verify user
    const supabaseUser = createClient(SUPABASE_URL, SUPABASE_ANON, {
      global: { headers: { Authorization: `Bearer ${supabaseAccessToken}` } }
    });
    const { data: { user } } = await supabaseUser.auth.getUser();
    if (!user) return res.status(401).json({ error: "Not authenticated" });

    const supabaseAdmin = createClient(SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
    const { data: profile } = await supabaseAdmin
      .from("profiles")
      .select("id,email,stripe_customer_id")
      .eq("id", user.id)
      .single();

    let customerId = profile?.stripe_customer_id;

    if (!customerId) {
      // best-effort: find by email before creating
      const email = user.email ?? profile?.email ?? undefined;
      if (email) {
        const list = await stripe.customers.list({ email, limit: 1 });
        if (list.data[0]) customerId = list.data[0].id;
      }
      if (!customerId) {
        const c = await stripe.customers.create({
          email: user.email ?? profile?.email ?? undefined,
          metadata: { supabase_user_id: user.id }
        });
        customerId = c.id;
      }
      await supabaseAdmin.from("profiles").upsert({
        id: user.id, email: user.email ?? profile?.email ?? null, stripe_customer_id: customerId
      });
    } else {
      await stripe.customers.update(customerId, { metadata: { supabase_user_id: user.id } });
    }

    const return_url = safeReturnUrl(returnTo) || `${DASHBOARD_ORIGIN}/`;

    const portal = await stripe.billingPortal.sessions.create({
      customer: customerId,
      return_url,
    });

    return res.status(200).json({ url: portal.url });
  } catch (e) {
    console.error("create-portal-session error:", e);
    return res.status(500).json({ error: "server_error" });
  }
}
