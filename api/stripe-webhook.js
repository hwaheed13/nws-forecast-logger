// /api/stripe-webhook.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

export const config = { api: { bodyParser: false } }; // Stripe needs raw body

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

function readRawBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => { data += chunk; });
    req.on("end", () => resolve(Buffer.from(data, "utf8")));
    req.on("error", reject);
  });
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  const rawBody = await readRawBody(req);
  const sig = req.headers["stripe-signature"];

  let event;
  try {
    event = stripe.webhooks.constructEvent(rawBody, sig, process.env.STRIPE_WEBHOOK_SECRET);
  } catch (err) {
    console.error("Webhook signature verify failed.", err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  try {
    const admin = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

    if (event.type === "checkout.session.completed") {
      const session = event.data.object;

      const subscription = await stripe.subscriptions.retrieve(session.subscription);
      const customer = await stripe.customers.retrieve(session.customer);

      const supabase_user_id =
        session.metadata?.supabase_user_id ||
        (customer?.metadata?.supabase_user_id ?? null);

      if (supabase_user_id) {
        const plan = session.metadata?.plan ?? null;
        await admin.from("profiles").upsert({
          id: supabase_user_id,
          stripe_customer_id: customer.id,
          stripe_subscription_id: subscription.id,
          plan,
          subscription_status: subscription.status === "active" || subscription.status === "trialing" ? "active" : subscription.status,
          current_period_end: new Date(subscription.current_period_end * 1000).toISOString()
        });
      }
    }

    if (event.type === "customer.subscription.created" ||
        event.type === "customer.subscription.updated" ||
        event.type === "customer.subscription.deleted") {
      const sub = event.data.object;

      // Find user by Stripe customer id — we stored it earlier
      // If you prefer, also keep supabase_user_id in subscription metadata on creation.
      let supabaseUserId = null;
      // Try using the customer->email path to find the profile if needed
      // but the usual is to look up by stripe_customer_id:
      const admin = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
      const { data: prof } = await admin.from("profiles").select("id").eq("stripe_customer_id", sub.customer).single();

      if (prof?.id) supabaseUserId = prof.id;

      if (supabaseUserId) {
        await admin.from("profiles").update({
          stripe_subscription_id: sub.id,
          subscription_status: sub.status === "active" || sub.status === "trialing" ? "active" : sub.status,
          current_period_end: new Date(sub.current_period_end * 1000).toISOString()
        }).eq("id", supabaseUserId);
      }
    }

    return res.json({ received: true });
  } catch (e) {
    console.error("Webhook handler error", e);
    return res.status(500).end();
  }
}
