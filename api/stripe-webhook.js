// pages/api/stripe-webhook.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

export const config = { api: { bodyParser: false } }; // Stripe needs raw body

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, {
  apiVersion: "2024-06-20",
});

// Support either env name to avoid mismatch
const SERVICE_ROLE =
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_ROLE;

const admin = createClient(process.env.SUPABASE_URL, SERVICE_ROLE);

function readRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function normalizeStatus(stripeStatus) {
  // Stripe: incomplete | incomplete_expired | trialing | active | past_due | canceled | unpaid | paused
  if (stripeStatus === "active" || stripeStatus === "trialing") return stripeStatus;
  return "inactive";
}

async function upsertProfileBasic({
  id, // supabase user id
  email,
  stripe_customer_id,
  stripe_subscription_id,
  subscription_status,
  current_period_end,
  plan_interval,
  plan_amount,
}) {
  const payload = {
    id,
    email: email ?? null,
    stripe_customer_id: stripe_customer_id ?? null,
    stripe_subscription_id: stripe_subscription_id ?? null,
    subscription_status: subscription_status ?? "inactive",
    current_period_end: current_period_end ?? null, // ISO string or null
    plan_interval: plan_interval ?? null,           // "month" | "year" | null
    plan_amount: typeof plan_amount === "number" ? plan_amount : null, // cents
    updated_at: new Date().toISOString(),
  };

  const { error } = await admin.from("profiles").upsert(payload, { onConflict: "id" });
  if (error) {
    console.error("profiles upsert error:", {
      message: error.message,
      details: error.details,
      hint: error.hint,
      code: error.code,
    });
    throw new Error("DB upsert failed");
  }
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  let event;
  try {
    const rawBody = await readRawBody(req);
    const sig = req.headers["stripe-signature"];
    event = stripe.webhooks.constructEvent(rawBody, sig, process.env.STRIPE_WEBHOOK_SECRET);
  } catch (err) {
    console.error("Webhook signature verify failed:", err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  try {
    switch (event.type) {
      case "checkout.session.completed": {
        // Fires after successful Checkout
        const session = event.data.object;

        // Only continue if we got a sub & customer
        if (!session.subscription || !session.customer) break;

        const [subscription, customer] = await Promise.all([
          stripe.subscriptions.retrieve(session.subscription),
          stripe.customers.retrieve(session.customer),
        ]);

        const status = normalizeStatus(subscription.status);
        const item = subscription.items?.data?.[0];
        const price = item?.price;
        const plan_interval = price?.recurring?.interval ?? null;
        const plan_amount = typeof price?.unit_amount === "number" ? price.unit_amount : null;

        // Prefer metadata on session; fall back to customer metadata
        const supabase_user_id =
          session.metadata?.supabase_user_id ||
          customer?.metadata?.supabase_user_id ||
          null;

        if (!supabase_user_id) break;

        await upsertProfileBasic({
          id: supabase_user_id,
          email: session.customer_details?.email ?? customer?.email ?? null,
          stripe_customer_id: customer.id,
          stripe_subscription_id: subscription.id,
          subscription_status: status,
          current_period_end: subscription.current_period_end
            ? new Date(subscription.current_period_end * 1000).toISOString()
            : null,
          plan_interval,
          plan_amount,
        });

        break;
      }

      case "customer.subscription.created":
      case "customer.subscription.updated":
      case "customer.subscription.deleted": {
        const sub = event.data.object;

        const status = normalizeStatus(sub.status);
        const item = sub.items?.data?.[0];
        const price = item?.price;
        const plan_interval = price?.recurring?.interval ?? null;
        const plan_amount = typeof price?.unit_amount === "number" ? price.unit_amount : null;

        // Try to locate the user via stripe_customer_id stored in profiles
        const { data: profByCust } = await admin
          .from("profiles")
          .select("id,email")
          .eq("stripe_customer_id", sub.customer)
          .maybeSingle();

        let supabase_user_id = profByCust?.id;
        let email = profByCust?.email ?? null;

        if (!supabase_user_id) {
          // Fallback: read from customer metadata
          const cust = await stripe.customers.retrieve(sub.customer);
          supabase_user_id = cust?.metadata?.supabase_user_id ?? null;
          email = email ?? cust?.email ?? null;

          // Ensure we persist the stripe_customer_id if found now
          if (supabase_user_id) {
            await admin
              .from("profiles")
              .upsert(
                {
                  id: supabase_user_id,
                  email,
                  stripe_customer_id: sub.customer,
                  updated_at: new Date().toISOString(),
                },
                { onConflict: "id" }
              );
          }
        }

        if (supabase_user_id) {
          await upsertProfileBasic({
            id: supabase_user_id,
            email,
            stripe_customer_id: sub.customer,
            stripe_subscription_id: sub.id,
            subscription_status: status,
            current_period_end: sub.current_period_end
              ? new Date(sub.current_period_end * 1000).toISOString()
              : null,
            plan_interval,
            plan_amount,
          });
        }

        break;
      }

      // Optional: you can use this to send your own “trial ending soon” emails.
      case "customer.subscription.trial_will_end": {
        const sub = event.data.object;
        console.log("Trial will end soon for subscription:", sub.id);
        break;
      }

      default:
        // Ignore events that aren't relevant to profile updates
        break;
    }

    return res.json({ received: true });
  } catch (e) {
    console.error("Webhook handler error", e);
    return res.status(500).end();
  }
}
