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
// (You chose to allow only 'active'/'trialing'; everything else becomes 'inactive')
function normalizedStatusFromStripe(status) {
  return status === "active" || status === "trialing" ? status : "inactive";
}

// plan: "monthly" | "yearly" | null
function planFromPrice(price) {
  const interval = price?.recurring?.interval; // "month" | "year" | undefined
  if (interval === "month") return "monthly";
  if (interval === "year") return "yearly";
  return null;
}

function tsOrNull(unix) {
  return typeof unix === "number" ? new Date(unix * 1000).toISOString() : null;
}

// Resolve your Supabase user id from subscription/customer metadata
async function resolveSupabaseUserIdFromSub(sub) {
  if (sub?.metadata?.supabase_user_id) return sub.metadata.supabase_user_id;
  if (sub?.customer) {
    const cust = await stripe.customers.retrieve(sub.customer);
    if (cust?.metadata?.supabase_user_id) return cust.metadata.supabase_user_id;
  }
  return null;
}

// Apply subscription -> profiles patch per your schema
async function applySubscriptionToProfile(admin, sub, opts = {}) {
  const statusNorm = normalizedStatusFromStripe(sub.status);
  const item = sub.items?.data?.[0];
  const price = item?.price ?? null;

  const plan = planFromPrice(price);
  const periodEnd = tsOrNull(sub.current_period_end);
  const trialStart = tsOrNull(sub.trial_start);
  const trialEnd   = tsOrNull(sub.trial_end);
  const onTrial = sub.status === "trialing";

  // Find Supabase user id
  let targetId = await resolveSupabaseUserIdFromSub(sub);

  // Fallback: match by stripe_customer_id if we’ve stored it previously
  if (!targetId && sub.customer) {
    const { data: profByCust } = await admin
      .from("profiles")
      .select("id")
      .eq("stripe_customer_id", sub.customer)
      .maybeSingle();
    targetId = profByCust?.id || null;
  }

  if (!targetId) return; // cannot map; ignore safely

  // Ensure stripe_customer_id is persisted (one-time upsert)
  await admin
    .from("profiles")
    .upsert(
      { id: targetId, stripe_customer_id: sub.customer, updated_at: new Date().toISOString() },
      { onConflict: "id" }
    );

  const patch = {
    stripe_subscription_id: sub.id,
    subscription_status: statusNorm, // 'active' | 'trialing' | 'inactive'
    plan,                             // 'monthly' | 'yearly' | null
    current_period_end: periodEnd,
    updated_at: new Date().toISOString(),
  };

  // Trial bookkeeping (write when trial exists)
  if (trialStart) {
    patch.free_trial_started_at = trialStart;
    patch.trial_started_at = trialStart; // you keep both
  }
  if (trialEnd) {
    patch.free_trial_ends_at = trialEnd;
  }
  if (onTrial) {
    patch.free_trial_used = true;   // prevents repeat trials
    patch.trial_used = true;        // legacy mirror
    patch.trial_subscription_id = sub.id;
  }

  // Optional email update if provided via opts
  if (opts.email) patch.email = opts.email;

  await admin.from("profiles").update(patch).eq("id", targetId);
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
      // Fired after Checkout completes (first-time subscribe or re-subscribe)
      case "checkout.session.completed": {
        const session = event.data.object;

        if (session.subscription && session.customer) {
          // Retrieve fresh subscription (status may be 'trialing', 'active', or 'incomplete' pre-payment)
          const [subscription, customer] = await Promise.all([
            stripe.subscriptions.retrieve(session.subscription),
            stripe.customers.retrieve(session.customer),
          ]);

          await applySubscriptionToProfile(admin, subscription, {
            email: session.customer_details?.email ?? null,
          });
        }
        break;
      }

      // Some Checkouts with SCA/async auth confirm after redirect
      case "checkout.session.async_payment_succeeded": {
        const session = event.data.object;
        if (session.subscription) {
          const sub = await stripe.subscriptions.retrieve(session.subscription);
          await applySubscriptionToProfile(admin, sub, {
            email: session.customer_details?.email ?? null,
          });
        }
        break;
      }

      // Subscription lifecycle (status flips, renewals, cancellations, trial→active, etc.)
      case "customer.subscription.created":
      case "customer.subscription.updated":
      case "customer.subscription.deleted": {
        const sub = event.data.object;
        await applySubscriptionToProfile(admin, sub);
        break;
      }

      // When an invoice is paid (including the first invoice), ensure period_end & status are fresh
      case "invoice.payment_succeeded": {
        const invoice = event.data.object;
        if (invoice.subscription) {
          const sub = await stripe.subscriptions.retrieve(invoice.subscription);
          await applySubscriptionToProfile(admin, sub);
        }
        break;
      }

      // Optional — notify before trial ends (no DB change required)
      case "customer.subscription.trial_will_end": {
        const sub = event.data.object;
        console.log("Trial will end soon for subscription:", sub.id);
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
