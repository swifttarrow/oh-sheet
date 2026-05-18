# Railway Migration Runbook (YouTube-only)

Minimum-viable lift-and-shift to keep the public site online by deploying
only the services that the YouTube → TuneChat path needs. Everything else
(transcribe / arrange / decomposer / assembler / humanize / refine
workers and the engraver service) is **not deployed on Railway**.

The TuneChat fast-path branches in `backend/jobs/runner.py` cause
`title_lookup` jobs (YouTube URL / song title search) to short-circuit
to TuneChat at the ingest stage. Artifact URLs returned by TuneChat are
proxied back to the client via `/v1/artifacts/{id}/{kind}` —
`backend/api/routes/artifacts.py` already has the proxy + SSRF guard.

## Service map

| Railway service     | Source       | Start command                                                                                       | Public?    |
| ------------------- | ------------ | --------------------------------------------------------------------------------------------------- | ---------- |
| `orchestrator`      | GHCR image   | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1`                                  | Yes (TLS)  |
| `worker-ingest`     | GHCR image   | `celery -A backend.workers.celery_app worker -Q ingest -c 1 --loglevel=warning`                     | No         |
| `redis`             | Railway plugin | (managed)                                                                                         | No         |

Both Railway services pull the same image:
`ghcr.io/<owner>/<repo>/app:latest`. Only the start command differs.

## One-time setup (Railway dashboard)

1. **Create a new Railway project** — call it `oh-sheet` (or attach to
   the existing one alongside TuneChat).
2. **Add the Redis plugin** to the project. Railway will inject
   `REDIS_URL` automatically; we re-map it via `OHSHEET_REDIS_URL`.
3. **Create the `orchestrator` service**
   - Source: Docker image → `ghcr.io/<owner>/<repo>/app:latest`
   - Start command: as above
   - Public networking: enabled, port `8000` (Railway sets `$PORT`)
   - Healthcheck path: `/v1/health`
   - Env vars: see the table below
4. **Create the `worker-ingest` service**
   - Source: same image, same Docker repo
   - Start command: as above
   - Public networking: **disabled**
   - Env vars: same TuneChat + Redis values; cookies optional
5. **Pull-image credentials** — if the GHCR image is private, add a
   Docker Hub style auth pair under Project → Settings → Image Registries
   pointing at `ghcr.io` with a PAT scoped `read:packages`.
6. **Custom domain** — point your existing oh-sheet domain at the
   orchestrator service's Railway-provided CNAME. Railway issues the TLS
   cert automatically (Caddy is no longer needed).

## Env vars (set in both services unless noted)

| Var                             | Value                                            | Notes |
| ------------------------------- | ------------------------------------------------ | ----- |
| `OHSHEET_REDIS_URL`             | `${{ Redis.REDIS_URL }}`                         | Reference Railway's Redis plugin |
| `OHSHEET_BLOB_ROOT`             | `/tmp/blob`                                      | Ephemeral; no volume mount on YouTube path |
| `OHSHEET_TUNECHAT_ENABLED`      | `true`                                           | **Critical** — without this the YouTube short-circuit never fires |
| `OHSHEET_TUNECHAT_URL`          | `https://tunechat.raqdrobinson.com`              | Or whatever your prod TuneChat is |
| `OHSHEET_TUNECHAT_API_KEY`      | (secret)                                         | Match the value TuneChat is checking |
| `OHSHEET_TUNECHAT_TIMEOUT_SEC`  | `900`                                            | Long enough for 10-min audio |
| `OHSHEET_SCORE_PIPELINE`        | `arrange`                                        | Orchestrator only |
| `OHSHEET_YOUTUBE_ONLY_MODE`     | `true`                                           | Refuses audio/MIDI uploads at the API with a friendly 400. Flip to `false` to revive uploads when the ML workers are re-deployed somewhere. |
| `OHSHEET_YOUTUBE_CACHE_ENABLED` | `true` (default)                                 | Re-submitted YouTube URLs short-circuit to the prior job. Kill switch — set to `false` to force every submit to re-run the full pipeline. |
| `OHSHEET_YOUTUBE_CACHE_TTL_SEC` | (unset → 30 days)                                | Cap exposure to stale TuneChat artifact URLs. Tune lower if TuneChat purges artifacts faster than 30 days. |
| `OHSHEET_ANTHROPIC_API_KEY`     | (unset)                                          | Refine worker not deployed — leave empty |
| `OHSHEET_ENGRAVER_SERVICE_URL`  | `http://disabled.local`                          | Required by config but unused on YouTube path |
| `OHSHEET_YTDLP_COOKIES_PATH`    | (unset)                                          | Skip cookies for v0 — yt-dlp runs anonymously, may 429 |

## Deploying

1. Open GitHub → Actions → **Deploy to Railway (YouTube-only)** →
   **Run workflow**. This builds the `app` image and pushes to GHCR.
2. (Optional) Add these GitHub repo secrets so the workflow also
   triggers a Railway redeploy:
   - `RAILWAY_TOKEN`
   - `RAILWAY_SERVICE_ORCHESTRATOR`
   - `RAILWAY_SERVICE_WORKER_INGEST`
3. If the secrets aren't set, click **Deploy** on each Railway service
   manually after the workflow finishes.

## Known limitations (v0)

- **No YouTube cookies**: yt-dlp runs anonymously. Heavy traffic may
  hit YouTube's bot-detection 429. To fix: write a small entrypoint
  wrapper that materializes a `OHSHEET_YTDLP_COOKIES` secret to disk
  before exec'ing the original command.
- **Non-YouTube paths refuse at the API**: with `OHSHEET_YOUTUBE_ONLY_MODE=true`,
  `audio_upload` / `midi_upload` submissions return a friendly 400
  ("Audio and MIDI uploads are temporarily disabled — please paste a
  YouTube link instead"). The underlying pipeline code still
  exercises in CI; flip the flag back to `false` if/when the ML
  workers are re-deployed somewhere. See commit `4b0d655` for the
  feature flag rationale.
- **Job state lost on restart**: `JobManager` is in-process. Container
  restart wipes in-flight jobs. Acceptable for a single-host demo;
  needs Redis/Postgres backing before scaling out. **Mitigation**:
  the YouTube job cache (`OHSHEET_YOUTUBE_CACHE_ENABLED=true`) means
  re-submitted URLs reuse the prior `job_id` from Redis, so a restart
  during in-flight jobs forces a re-run but doesn't lose completed
  jobs — the cache serves as durable read-side for successes.
- **Cache namespace**: the YouTube cache stores entries under
  `ohsheet:cache:youtube:<sha-prefix>` in the same Redis db as
  Celery's broker/result keys. If you add a second cache later, pick
  a sibling namespace (`ohsheet:cache:<feature>:`) to avoid collision.
- **One image, two services**: simpler to operate but you pay for two
  containers' worth of the ~2 GB `app` image. Cost is acceptable while
  Railway is the only environment.

## Cutting over from GCP

1. Run the new workflow once — confirm the `:latest` tag exists on GHCR.
2. Provision Railway services per the steps above.
3. Run the **Verify after deploy** checklist below against the
   Railway-provided URL (something like
   `https://oh-sheet-orchestrator-production.up.railway.app`).
4. Update DNS to point oh-sheet's prod domain at the Railway CNAME.
5. Re-run the verification checklist against the prod domain.
6. Tear down the GCE VM (`gcloud compute instances delete ...`) and the
   Artifact Registry repo only after the new domain has been stable
   for at least 24 hours.
7. Delete `.github/workflows/deploy.yml` (the GCP one) and rename
   `deploy-railway.yml` → `deploy.yml`, switching its trigger from
   `workflow_dispatch` to `push: { branches: [main] }`.

## Verify after deploy

Run these against the Railway URL (or final prod domain). Replace
`$URL` with the actual base URL. Anything that fails → check the
service logs in Railway before proceeding.

```bash
URL="https://your-railway-url"

# 1. Health: orchestrator is up.
curl -fsS "$URL/v1/health"
# Expect: 200, JSON body { "status": "ok", ... }

# 2. Non-YouTube guard fires (proves OHSHEET_YOUTUBE_ONLY_MODE=true).
curl -fsS -X POST "$URL/v1/jobs" \
  -H 'Content-Type: application/json' \
  -d '{"audio": {"uri": "file:///nope.wav", "format": "wav",
       "sample_rate": 44100, "duration_sec": 1.0, "channels": 2,
       "content_hash": "0000000000000000000000000000000000000000000000000000000000000000"}}'
# Expect: 400 with detail containing "youtube" (case-insensitive).

# 3. YouTube job dispatches (proves worker-ingest is consuming the
#    `ingest` queue + TuneChat client is reachable).
curl -fsS -X POST "$URL/v1/jobs" \
  -H 'Content-Type: application/json' \
  -d '{"title": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
# Expect: 202, JSON body with job_id. Note it.

# 4. Job runs to completion (depends on TuneChat; allow up to ~5 min).
JOB_ID="<from step 3>"
curl -fsS "$URL/v1/jobs/$JOB_ID" | jq .status
# Expect: "succeeded" within ~5 min.

# 5. Artifacts proxy works (proves the TuneChat URL proxy +
#    SSRF allowlist are wired correctly).
curl -fsSI "$URL/v1/artifacts/$JOB_ID/pdf"
# Expect: 200, Content-Type: application/pdf, non-zero Content-Length.

# 6. Cache hits on re-submit (proves the YouTube cache is connected
#    to Redis and the route's short-circuit is firing).
curl -fsS -X POST "$URL/v1/jobs" \
  -H 'Content-Type: application/json' \
  -d '{"title": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}' \
  | jq .job_id
# Expect: SAME job_id as step 3. Should return in <100ms.
```

If step 6 returns a different `job_id`, the cache isn't wired —
check `OHSHEET_YOUTUBE_CACHE_ENABLED` is unset or `true`, and that
both services point at the same `OHSHEET_REDIS_URL`.

If step 5 returns 502 or 504, the SSRF allowlist might be blocking
the TuneChat URL — check `OHSHEET_TUNECHAT_URL` exactly matches the
host TuneChat returns artifact URLs from.
