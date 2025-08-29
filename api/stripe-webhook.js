// /api/stripe-webhook.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

export const config = { api: { bodyParser: false } }; // Stripe needs raw body

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

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

  let event;
  try {
    const rawBody = await readRawBody(req);
    const sig = req.headers["stripe-signature"];
    event = stripe.webhooks.constructEvent(rawBody, sig, process.env.STRIPE_WEBHOOK_SECRET);
  } catch (err) {
    console.error("Webhook signature verify failed.", err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  try {
    // ✅ server-only env names
    const SUPABASE_URL = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
    const admin = createClient(SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

    if (event.type === "checkout.session.completed") {
      const session = event.data.object;

      // In subscription mode, session.subscription is set
      if (session.subscription && session.customer) {
        const [subscription, customer] = await Promise.all([
          stripe.subscriptions.retrieve(session.subscription),
          stripe.customers.retrieve(session.customer),
        ]);

        const supabase_user_id =
          session.metadata?.supabase_user_id ||
          (customer?.metadata?.supabase_user_id ?? null);

        // Normalize status: treat trialing as active for your gate
        const normalizedStatus =
          subscription.status === "active" || subscription.status === "trialing"
            ? "active"
            : subscription.status;

        if (supabase_user_id) {
          await admin.from("profiles").upsert({
            id: supabase_user_id,
            email: session.customer_details?.email ?? null,
            stripe_customer_id: customer.id,
            stripe_subscription_id: subscription.id,
            plan: session.metadata?.plan ?? null,
            subscription_status: normalizedStatus,
            current_period_end: subscription.current_period_end
              ? new Date(subscription.current_period_end * 1000).toISOString()
              : null,
            updated_at: new Date().toISOString(),
          });
        }
      }
    }

    if (
      event.type === "customer.subscription.created" ||
      event.type === "customer.subscription.updated" ||
      event.type === "customer.subscription.deleted"
    ) {
      const sub = event.data.object;

      const normalizedStatus =
        sub.status === "active" || sub.status === "trialing" ? "active" : sub.status;

      // Find profile by stripe_customer_id (we saved it earlier)
      const { data: prof } = await admin
        .from("profiles")
        .select("id")
        .eq("stripe_customer_id", sub.customer)
        .single();

      if (prof?.id) {
        await admin
          .from("profiles")
          .update({
            stripe_subscription_id: sub.id,
            subscription_status: normalizedStatus,
            current_period_end: sub.current_period_end
              ? new Date(sub.current_period_end * 1000).toISOString()
              : null,
            updated_at: new Date().toISOString(),
          })
          .eq("id", prof.id);
      }
    }

    return res.json({ received: true });
  } catch (e) {
    console.error("Webhook handler error", e);
    return res.status(500).end();
  }
}
