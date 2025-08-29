// /api/stripe-webhook.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

// Stripe requires the raw body for signature verification
export const config = { api: { bodyParser: false } };

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, {
  apiVersion: "2023-10-16",
});

// Read raw body into a Buffer
function readRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  // 1) Verify Stripe signature
  let event;
  try {
    const rawBody = await readRawBody(req);
    const sig = req.headers["stripe-signature"];
    event = stripe.webhooks.constructEvent(
      rawBody,
      sig,
      process.env.STRIPE_WEBHOOK_SECRET
    );
  } catch (err) {
    console.error("Webhook signature verify failed:", err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  // 2) Supabase admin client (server-only key)
  const SUPABASE_URL =
    process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAdmin = createClient(
    SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
  );

  try {
    // === Handle Checkout completed (subscription) ===
    if (event.type === "checkout.session.completed") {
      const session = event.data.object;

      // Only care if this was a subscription checkout
      if (session.mode === "subscription" && session.subscription && session.customer) {
        const [subscription, customer] = await Promise.all([
          stripe.subscriptions.retrieve(session.subscription),
          stripe.customers.retrieve(session.customer),
        ]);

        // Prefer session metadata; fall back to customer metadata
        const supabase_user_id =
          session.metadata?.supabase_user_id ||
          customer?.metadata?.supabase_user_id ||
          null;

        // Treat trialing as active for your access gate
        const normalizedStatus =
          subscription.status === "active" || subscription.status === "trialing"
            ? "active"
            : subscription.status;

        if (supabase_user_id) {
          const { error: upsertErr } = await supabaseAdmin.from("profiles").upsert({
            id: supabase_user_id,
            email: session.customer_details?.email ?? null,
            stripe_customer_id: customer.id,
            stripe_subscription_id: subscription.id,
            plan: session.metadata?.plan ?? null,
            subscription_status: normalizedStatus,
            current_period_end: subscription.current_period_end
              ? new Date(subscription.current_period_end * 1000).toISOString()
              : null,
          });

          if (upsertErr) {
            console.error("profiles upsert failed:", upsertErr);
            // Return 500 so Stripe retries
            return res.status(500).json({ error: "db_upsert_failed" });
          }
        } else {
          console.warn(
            "No supabase_user_id found in session/customer metadata for checkout.session.completed"
          );
        }
      }
    }

    // === Keep subscription state in sync ===
    if (
      event.type === "customer.subscription.created" ||
      event.type === "customer.subscription.updated" ||
      event.type === "customer.subscription.deleted"
    ) {
      const sub = event.data.object;

      const normalizedStatus =
        sub.status === "active" || sub.status === "trialing" ? "active" : sub.status;

      // Find profile by stored stripe_customer_id
      const { data: prof, error: findErr } = await supabaseAdmin
        .from("profiles")
        .select("id")
        .eq("stripe_customer_id", sub.customer)
        .maybeSingle();

      if (findErr) {
        console.error("profiles find-by-customer failed:", findErr);
        return res.status(500).json({ error: "db_find_failed" });
      }

      if (prof?.id) {
        const { error: updateErr } = await supabaseAdmin
          .from("profiles")
          .update({
            stripe_subscription_id: sub.id,
            subscription_status: normalizedStatus,
            current_period_end: sub.current_period_end
              ? new Date(sub.current_period_end * 1000).toISOString()
              : null,
          })
          .eq("id", prof.id);

        if (updateErr) {
          console.error("profiles update failed:", updateErr);
          return res.status(500).json({ error: "db_update_failed" });
        }
      } else {
        // Not fatal; perhaps the checkout handler hasn't run yet
        console.warn(
          "No profile found for stripe_customer_id in subscription event:",
          sub.customer
        );
      }
    }

    // Acknowledge success
    return res.json({ received: true });
  } catch (e) {
    console.error("Webhook handler error:", e);
    return res.status(500).end();
  }
}
