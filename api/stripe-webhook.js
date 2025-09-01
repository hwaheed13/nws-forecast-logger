// /api/stripe-webhook.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

// Stripe needs the raw body for signature verification on Vercel
export const config = { api: { bodyParser: false } };

// Keep apiVersion aligned with your project (you were on 2023-10-16)
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

// ---- helpers ---------------------------------------------------------------
function readRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

// Normalize Stripe subscription status -> profiles.subscription_status
function normalizedStatusFromStripe(status) {
  // allow free trial access
  return status === "active" || status === "trialing" ? status : "inactive";
}

// pull basic plan info from price
function planFromPrice(price) {
  // You’ve been storing "monthly"/"yearly" — use interval if available
  const interval = price?.recurring?.interval; // "month" | "year" | undefined
  if (interval === "month") return "monthly";
  if (interval === "year") return "yearly";
  return null;
}

function tsOrNull(unix) {
  return typeof unix === "number" ? new Date(unix * 1000).toISOString() : null;
}

// ---- handler ---------------------------------------------------------------
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

  // Server-only Supabase client (Service Role bypasses RLS)
  const admin = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

  try {
    switch (event.type) {
      // Fired after Checkout completes (first time a user subscribes or starts trial)
      case "checkout.session.completed": {
        const session = event.data.object;

        if (session.subscription && session.customer) {
          const [subscription, customer] = await Promise.all([
            stripe.subscriptions.retrieve(session.subscription),
            stripe.customers.retrieve(session.customer),
          ]);

          // Try to get your Supabase user id from metadata (preferred) or customer metadata
          const supabase_user_id =
            session.metadata?.supabase_user_id ||
            customer?.metadata?.supabase_user_id ||
            null;

          // Read status + plan details
          const statusNorm = normalizedStatusFromStripe(subscription.status);
          const item = subscription.items?.data?.[0];
          const price = item?.price ?? null;
          const plan = planFromPrice(price);
          const periodEnd = tsOrNull(subscription.current_period_end);

          // Trial fields (if trialing phase exists)
          const trialStart = tsOrNull(subscription.trial_start);
          const trialEnd   = tsOrNull(subscription.trial_end);
          const onTrial = subscription.status === "trialing";

          if (supabase_user_id) {
            const payload = {
              id: supabase_user_id,
              email: session.customer_details?.email ?? null,
              stripe_customer_id: customer.id,
              stripe_subscription_id: subscription.id,
              subscription_status: statusNorm,      // "active" | "trialing" | "inactive"
              plan,                                  // "monthly" | "yearly" | null
              current_period_end: periodEnd,
              // trial bookkeeping
              free_trial_used: onTrial ? true : (session.metadata?.ddp_trial === "true" ? true : undefined),
              free_trial_started_at: trialStart,
              free_trial_ends_at: trialEnd,
              // also keep a generic "trial_used" if your schema has it
              trial_used: onTrial ? true : undefined,
              trial_subscription_id: onTrial ? subscription.id : undefined,
              trial_started_at: trialStart,
              updated_at: new Date().toISOString(),
            };

            // Upsert on profiles (id PK)
            const { error } = await admin.from("profiles").upsert(payload, { onConflict: "id" });
            if (error) {
              console.error("profiles upsert error (checkout.session.completed):", {
                message: error.message, details: error.details, hint: error.hint, code: error.code
              });
              return res.status(500).json({ error: "DB upsert failed" });
            }
          }
        }
        break;
      }

      // Subscription life cycle changes (status flips, renewals, cancellations, trial→active, etc.)
      case "customer.subscription.created":
      case "customer.subscription.updated":
      case "customer.subscription.deleted": {
        const sub = event.data.object;

        const statusNorm = normalizedStatusFromStripe(sub.status);
        const item = sub.items?.data?.[0];
        const price = item?.price ?? null;

        const plan = planFromPrice(price);
        const periodEnd = tsOrNull(sub.current_period_end);
        const trialStart = tsOrNull(sub.trial_start);
        const trialEnd   = tsOrNull(sub.trial_end);
        const onTrial = sub.status === "trialing";

        // Find profile by stripe_customer_id…
        const { data: profByCust, error: findErr } = await admin
          .from("profiles")
          .select("id")
          .eq("stripe_customer_id", sub.customer)
          .maybeSingle();

        let targetId = profByCust?.id;

        // …or fall back to customer metadata if needed.
        if (!targetId) {
          const cust = await stripe.customers.retrieve(sub.customer);
          targetId = cust?.metadata?.supabase_user_id ?? null;

          // If we found a Supabase id here, ensure stripe_customer_id is persisted once
          if (targetId) {
            await admin
              .from("profiles")
              .upsert(
                {
                  id: targetId,
                  stripe_customer_id: sub.customer,
                  updated_at: new Date().toISOString(),
                },
                { onConflict: "id" }
              );
          }
        }

        if (targetId) {
          const patch = {
            stripe_subscription_id: sub.id,
            subscription_status: statusNorm, // preserves "trialing"
            plan,                             // "monthly" | "yearly" | null
            current_period_end: periodEnd,
            updated_at: new Date().toISOString(),
          };

          // Trial bookkeeping: keep a record of the trial window if present
          if (trialStart) patch.free_trial_started_at = trialStart;
          if (trialEnd)   patch.free_trial_ends_at = trialEnd;

          // Mark that a trial has been used (so you can prevent multiple trials)
          if (onTrial) {
            patch.free_trial_used = true;
            patch.trial_used = true; // if you still keep this legacy column
            patch.trial_subscription_id = sub.id;
            patch.trial_started_at = trialStart;
          }

          const { error } = await admin
            .from("profiles")
            .update(patch)
            .eq("id", targetId);

          if (error) {
            console.error("profiles update error (subscription.*):", {
              message: error.message, details: error.details, hint: error.hint, code: error.code
            });
            return res.status(500).json({ error: "DB update failed" });
          }
        }
        break;
      }

      // Optional — Stripe can send this ~3 days before the trial ends (config dependent)
      case "customer.subscription.trial_will_end": {
        const sub = event.data.object;
        console.log("Trial will end soon for subscription:", sub.id);
        // You could email the user here via your own system if desired.
        break;
      }

      default:
        // ignore others
        break;
    }

    return res.json({ received: true });
  } catch (e) {
    console.error("Webhook handler error", e);
    return res.status(500).end();
  }
}
