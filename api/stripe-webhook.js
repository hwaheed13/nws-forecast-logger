// /api/stripe-webhook.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

export const config = { api: { bodyParser: false } }; // Stripe needs raw body

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2024-06-20" });

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SERVICE_ROLE = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_ROLE;
const admin = createClient(SUPABASE_URL, SERVICE_ROLE);

function readRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function normStatus(s) {
  return (s === "active" || s === "trialing") ? s : "inactive";
}

async function upsertByCustomer(customerId, patch) {
  // try by stripe_customer_id
  const { data: prof } = await admin
    .from("profiles")
    .select("id")
    .eq("stripe_customer_id", customerId)
    .maybeSingle();

  let id = prof?.id;
  if (!id) {
    const cust = await stripe.customers.retrieve(customerId);
    id = cust?.metadata?.supabase_user_id || null;
    if (id) {
      await admin
        .from("profiles")
        .upsert(
          { id, stripe_customer_id: customerId, updated_at: new Date().toISOString() },
          { onConflict: "id" }
        );
    }
  }
  if (!id) return null;

  await admin.from("profiles").update(patch).eq("id", id);
  return id;
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  let event;
  try {
    const raw = await readRawBody(req);
    const sig = req.headers["stripe-signature"];
    event = stripe.webhooks.constructEvent(raw, sig, process.env.STRIPE_WEBHOOK_SECRET);
  } catch (err) {
    console.error("Webhook signature verify failed:", err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  try {
    switch (event.type) {
      case "checkout.session.completed": {
        const session = event.data.object;
        const subscriptionId = session.subscription;
        const customerId = session.customer;
        if (!subscriptionId || !customerId) break;

        const [sub, cust] = await Promise.all([
          stripe.subscriptions.retrieve(subscriptionId),
          stripe.customers.retrieve(customerId),
        ]);

        const status = normStatus(sub.status);
        const isTrial = sub.status === "trialing" || session.metadata?.ddp_trial === "true";

        const price = sub.items?.data?.[0]?.price;
        const plan_interval = price?.recurring?.interval || null;
        const plan_amount = typeof price?.unit_amount === "number" ? price.unit_amount : null;

        const patch = {
          email: session.customer_details?.email ?? null,
          stripe_customer_id: customerId,
          stripe_subscription_id: subscriptionId,
          subscription_status: status,
          plan: session.metadata?.plan || (plan_interval === "year" ? "yearly" : "monthly"),
          current_period_end: sub.current_period_end
            ? new Date(sub.current_period_end * 1000).toISOString()
            : null,
          updated_at: new Date().toISOString(),
        };

        // Mark trial usage (your column names)
        if (isTrial) {
          patch.free_trial_used = true;
          patch.trial_subscription_id = subscriptionId;
          patch.trial_started_at = new Date().toISOString();
        }

        await upsertByCustomer(customerId, patch);
        break;
      }

      case "customer.subscription.created":
      case "customer.subscription.updated":
      case "customer.subscription.deleted": {
        const sub = event.data.object;
        const customerId = sub.customer;

        const status = normStatus(sub.status);
        const price = sub.items?.data?.[0]?.price;
        const plan_interval = price?.recurring?.interval || null;
        const plan_amount = typeof price?.unit_amount === "number" ? price.unit_amount : null;

        const patch = {
          stripe_subscription_id: sub.id,
          subscription_status: status,
          plan: plan_interval === "year" ? "yearly" : "monthly",
          current_period_end: sub.current_period_end
            ? new Date(sub.current_period_end * 1000).toISOString()
            : null,
          updated_at: new Date().toISOString(),
        };

        if (sub.status === "trialing") {
          patch.free_trial_used = true;
          patch.trial_subscription_id = sub.id;
          patch.trial_started_at = new Date().toISOString();
        }

        await upsertByCustomer(customerId, patch);
        break;
      }

      case "customer.subscription.trial_will_end": {
        const sub = event.data.object;
        console.log("Trial will end soon:", sub.id);
        // Optional: send email/notification here.
        break;
      }

      default:
        // ignore
        break;
    }

    return res.json({ received: true });
  } catch (e) {
    console.error("Webhook handler error", e);
    return res.status(500).end();
  }
}
