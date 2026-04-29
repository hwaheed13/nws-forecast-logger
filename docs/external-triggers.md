# External Triggers for forecast-frequent

GitHub Actions throttles `cron: "*/10"` to roughly hourly cadence under
repo load. The `forecast-frequent` workflow now also accepts a
`repository_dispatch` event with `event_type=forecast-frequent-tick`, so
an external scheduler can fire it reliably every 10 minutes.

## Setup

1. **Mint a fine-scoped GitHub PAT** with `repo` (or `actions:write`)
   scope. Save as `GH_DISPATCH_TOKEN`.

2. **Pick a scheduler**. Free options:
   - [cron-job.org](https://cron-job.org) — UI-based, every 1 min minimum.
   - Cloudflare Workers Cron — set in `wrangler.toml`:
     ```toml
     [triggers]
     crons = ["*/10 * * * *"]
     ```

3. **Configure the request** (POST with curl):
   ```bash
   curl -X POST \
     -H "Authorization: Bearer $GH_DISPATCH_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/repos/hwaheed13/nws-forecast-logger/dispatches \
     -d '{"event_type":"forecast-frequent-tick"}'
   ```

4. **Verify** the run shows up under
   `github.com/.../actions/workflows/forecast-frequent.yml`
   with trigger `repository_dispatch`.

## Why not just trust the cron?

The GitHub-managed cron runner is best-effort, not best-cadence. Under
load the scheduler can drop or delay invocations by 30+ minutes. We
intentionally keep the schedule cron in the workflow as a fallback — if
the external trigger fails, you still get hourly runs.

## Concurrency

The workflow already has `concurrency.cancel-in-progress: true`, so
overlapping triggers (cron + dispatch firing within 10s of each other)
won't stack.
