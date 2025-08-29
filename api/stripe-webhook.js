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
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  let event;
  try {
    const rawBody = await readRawBody(req);
    const sig = req.headers["stripe-signature"];
    event = stripe.webhooks.constructEvent(rawBody, sig, process.env.STRIPE_WEBHOOK_SECRET);
  } catch (err) {
    console.error("Webhook signature verify failed.", err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  // Server-only Supabase client (Service Role bypasses RLS)
  const admin = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

  try {
    switch (event.type) {
      case "checkout.session.completed": {
        const session = event.data.object;

        if (session.subscription && session.customer) {
          const [subscription, customer] = await Promise.all([
            stripe.subscriptions.retrieve(session.subscription),
            stripe.customers.retrieve(session.customer),
          ]);

          const supabase_user_id =
            session.metadata?.supabase_user_id ||
            customer?.metadata?.supabase_user_id ||
            null;

          // Keep true status, but map any non-active/trialing to 'inactive'
          const normalizedStatus =
            subscription.status === "active" || subscription.status === "trialing"
              ? subscription.status
              : "inactive";

          if (supabase_user_id) {
            const payload = {
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
            };

        const { error } = await admin.from("profiles").upsert(payload, { onConflict: "id" });
        if (error) {
          console.error("profiles upsert error (checkout.session.completed):", {
            message: error.message,
            details: error.details,
            hint: error.hint,
            code: error.code
          });
          return res.status(500).json({ error: "DB upsert failed" });
        }

          }
        }
        break;
      }

      case "customer.subscription.created":
      case "customer.subscription.updated":
      case "customer.subscription.deleted": {
        const sub = event.data.object;

        const normalizedStatus =
          sub.status === "active" || sub.status === "trialing" ? sub.status : "inactive";

        // Try by stripe_customer_id first
        const { data: profByCust } = await admin
          .from("profiles")
          .select("id")
          .eq("stripe_customer_id", sub.customer)
          .maybeSingle();

        let targetId = profByCust?.id;

        // Fallback: use customer.metadata.supabase_user_id
        if (!targetId) {
          const cust = await stripe.customers.retrieve(sub.customer);
          targetId = cust?.metadata?.supabase_user_id ?? null;

          // If we found a Supabase id here, also ensure stripe_customer_id is set
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
          const { error } = await admin
            .from("profiles")
            .update({
              stripe_subscription_id: sub.id,
              subscription_status: normalizedStatus,
              current_period_end: sub.current_period_end
                ? new Date(sub.current_period_end * 1000).toISOString()
                : null,
              updated_at: new Date().toISOString(),
            })
            .eq("id", targetId);

         if (error) {
      console.error("profiles update error (subscription.*):", {
        message: error.message,
        details: error.details,
        hint: error.hint,
        code: error.code
      });
      return res.status(500).json({ error: "DB update failed" });
}

        }
        break;
      }

      default:
        // Not relevant for profile updates
        break;
    }

    return res.json({ received: true });
  } catch (e) {
    console.error("Webhook handler error", e);
    return res.status(500).end();
  }
}
