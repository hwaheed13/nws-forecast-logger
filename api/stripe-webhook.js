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
    console.error("Webhook signature verify failed:", err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  const SUPABASE_URL = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const admin = createClient(SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

  try {
    if (event.type === "checkout.session.completed") {
      const session = event.data.object;

      if (session.mode === "subscription" && session.subscription && session.customer) {
        const [subscription, customer] = await Promise.all([
          stripe.subscriptions.retrieve(session.subscription),
          stripe.customers.retrieve(session.customer),
        ]);

        const supabase_user_id =
          session.metadata?.supabase_user_id ||
          customer?.metadata?.supabase_user_id ||
          null;

        const normalizedStatus =
          subscription.status === "active" || subscription.status === "trialing" ? "active" : subscription.status;

        if (supabase_user_id) {
          // Upsert core profile fields
          const { error: upsertErr } = await admin.from("profiles").upsert({
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
          if (upsertErr) {
            console.error("profiles upsert failed:", upsertErr);
            return res.status(500).json({ error: "db_upsert_failed" });
          }

          // Trial extras: auto-cancel at period end + mark used
          if (session.metadata?.plan === "trial") {
            try {
              if (subscription.current_period_end) {
                await stripe.subscriptions.update(subscription.id, {
                  cancel_at: subscription.current_period_end,
                });
              }
              await admin.from("profiles").update({ free_trial_used: true }).eq("id", supabase_user_id);
            } catch (err) {
              console.error("trial post-processing failed:", err);
              // Not fatal for acknowledgment; Stripe will keep the sub active if this fails
            }
          } else {
            // Any successful paid signup should also count as "trial used"
            const { error: markErr } = await admin
              .from("profiles")
              .update({ free_trial_used: true })
              .eq("id", supabase_user_id);
            if (markErr) console.error("mark free_trial_used on paid signup failed:", markErr);
          }
        }
      }
    }

    if (
      event.type === "customer.subscription.created" ||
      event.type === "customer.subscription.updated" ||
      event.type === "customer.subscription.deleted"
    ) {
      const sub = event.data.object;
      const normalizedStatus = (sub.status === "active" || sub.status === "trialing") ? "active" : sub.status;

      const { data: prof, error: findErr } = await admin
        .from("profiles")
        .select("id")
        .eq("stripe_customer_id", sub.customer)
        .maybeSingle();
      if (findErr) {
        console.error("profiles find-by-customer failed:", findErr);
        return res.status(500).json({ error: "db_find_failed" });
      }
      if (prof?.id) {
        const { error: updateErr } = await admin
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
        if (updateErr) {
          console.error("profiles update failed:", updateErr);
          return res.status(500).json({ error: "db_update_failed" });
        }
      }
    }

    return res.json({ received: true });
  } catch (e) {
    console.error("Webhook handler error", e);
    return res.status(500).end();
  }
}
