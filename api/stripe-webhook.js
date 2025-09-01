// /api/stripe-webhook.js
import Stripe from "stripe";
import { createClient } from "@supabase/supabase-js";

export const config = { api: { bodyParser: false } }; // Stripe needs raw body

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: "2023-10-16" });

// read raw body for webhook signature verification
function readRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

// map Stripe status → your profile.status
function normalizeStatus(s) {
  return (s === "active" || s === "trialing") ? s : "inactive";
}

// build the profile patch from a Stripe subscription object
function profilePatchFromSubscription(sub, sessionMeta = {}) {
  const status = normalizeStatus(sub.status);

  // Price/interval (first item)
  const item = sub.items?.data?.[0];
  const price = item?.price;
  const plan_interval = price?.recurring?.interval ?? null;
  const plan_amount = typeof price?.unit_amount === "number" ? price.unit_amount : null;

  // Trial fields (Unix → ISO)
  const trial_started = sub.trial_start ? new Date(sub.trial_start * 1000).toISOString() : null;
  const trial_ends    = sub.trial_end   ? new Date(sub.trial_end   * 1000).toISOString() : null;

  // Period end (use Stripe’s current_period_end)
  const current_period_end = sub.current_period_end
    ? new Date(sub.current_period_end * 1000).toISOString()
    : null;

  // If we can infer this subscription came from a free-trial checkout, mark it.
  const trialFlag = (sessionMeta.ddp_trial === "true") || (sub.trial_end && Date.now() < sub.trial_end * 1000);

  return {
    subscription_status: status,
    stripe_subscription_id: sub.id,
    // optional plan label (keep your column if you want it)
    plan: sessionMeta.plan ?? (plan_interval === "year" ? "yearly" : "monthly"),
    plan_interval,
    plan_amount,
    current_period_end,

    // your custom trial tracking fields
    free_trial_used: trialFlag || undefined,  // set true once; leave untouched otherwise
    free_trial_started_at: trial_started,
    free_trial_ends_at: trial_ends,

    // legacy fields you had (keep if you still use them)
    trial_used: trialFlag || undefined,
    trial_subscription_id: trialFlag ? sub.id : undefined,
    trial_started_at: trial_started,

    updated_at: new Date().toISOString(),
  };
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

  const admin = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

  try {
    switch (event.type) {
      case "checkout.session.completed": {
        const session = event.data.object;

        if (!session.subscription || !session.customer) break;

        const [subscription, customer] = await Promise.all([
          stripe.subscriptions.retrieve(session.subscription),
          stripe.customers.retrieve(session.customer),
        ]);

        // Prefer explicit user id from metadata; fall back to customer metadata.
        const supabase_user_id =
          session.metadata?.supabase_user_id ||
          customer?.metadata?.supabase_user_id ||
          null;

        if (!supabase_user_id) break;

        const patch = profilePatchFromSubscription(subscription, session.metadata || {});
        // Keep email and stripe ids in sync, too.
        const payload = {
          id: supabase_user_id,
          email: session.customer_details?.email ?? null,
          stripe_customer_id: customer.id,
          ...patch,
        };

        const { error } = await admin.from("profiles").upsert(payload, { onConflict: "id" });
        if (error) {
          console.error("profiles upsert error (checkout.session.completed):", error);
          return res.status(500).json({ error: "DB upsert failed" });
        }
        break;
      }

      case "customer.subscription.created":
      case "customer.subscription.updated":
      case "customer.subscription.deleted": {
        const sub = event.data.object;

        // find profile by stripe_customer_id
        const { data: profByCust, error: findErr } = await admin
          .from("profiles")
          .select("id")
          .eq("stripe_customer_id", sub.customer)
          .maybeSingle();

        let targetId = profByCust?.id;
        let sessionMeta = {};

        if (!targetId) {
          // fall back to customer metadata
          const cust = await stripe.customers.retrieve(sub.customer);
          targetId = cust?.metadata?.supabase_user_id ?? null;
        }

        if (!targetId) break;

        const patch = profilePatchFromSubscription(sub, sessionMeta);

        const { error } = await admin
          .from("profiles")
          .update(patch)
          .eq("id", targetId);

        if (error) {
          console.error("profiles update error (subscription.*):", error);
          return res.status(500).json({ error: "DB update failed" });
        }
        break;
      }

      default:
        // ignore other events
        break;
    }

    return res.json({ received: true });
  } catch (e) {
    console.error("Webhook handler error", e);
    return res.status(500).end();
  }
}
