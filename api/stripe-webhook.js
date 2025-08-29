// /api/stripe-webhook.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

// Stripe needs raw body
export const config = { api: { bodyParser: false } };

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
  const supabaseAdmin = createClient(SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

  try {
    // ── Checkout completed → upsert profile + handle trial auto-cancel ────────
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
          subscription.status === "active" || subscription.status === "trialing"
            ? "active"
            : subscription.status;

        if (supabase_user_id) {
          // Upsert main subscription state
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
            return res.status(500).json({ error: "db_upsert_failed" });
          }

          // If this was a trial, mark used + schedule cancel at trial end
          if (session.metadata?.plan === "trial") {
            // mark the trial as used
            const { error: trialErr } = await supabaseAdmin
              .from("profiles")
              .update({ free_trial_used: true })
              .eq("id", supabase_user_id);
            if (trialErr) console.error("mark free_trial_used failed:", trialErr);

            // Cancel at the end of trial so they aren't charged automatically
            const trialEnd = subscription.trial_end || subscription.current_period_end;
            if (trialEnd) {
              try {
                await stripe.subscriptions.update(subscription.id, {
                  cancel_at: trialEnd,
                });
              } catch (e) {
                console.error("Could not schedule cancel_at for trial:", e);
              }
            }
          }
        } else {
          console.warn("No supabase_user_id on checkout.session.completed");
        }
      }
    }

    // ── Keep subscription state in sync ────────────────────────────────────────
    if (
      event.type === "customer.subscription.created" ||
      event.type === "customer.subscription.updated" ||
      event.type === "customer.subscription.deleted"
    ) {
      const sub = event.data.object;
      const normalizedStatus =
        sub.status === "active" || sub.status === "trialing" ? "active" : sub.status;

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
        console.warn("No profile found for stripe_customer_id:", sub.customer);
      }
    }

    return res.json({ received: true });
  } catch (e) {
    console.error("Webhook handler error:", e);
    return res.status(500).end();
  }
}
