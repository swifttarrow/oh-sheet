# Deployment & Release Flow

Oh Sheet uses a two-environment release flow with strict branch protection.

## Branches & environments

| Branch | Environment | URL | Approvals |
|--------|-------------|-----|-----------|
| `qa`   | **QA**   | https://oh-sheet-qa.duckdns.org | 1 |
| `main` | **prod** | https://oh-sheet.duckdns.org | 2 |

## Release flow

```
feature/foo ──► qa ──► main
              (1 approval)  (2 approvals, only from qa)
                 │              │
                 ▼              ▼
              QA deploy     Prod deploy
```

1. **Develop on a feature branch** off `qa`.
2. **Open a PR into `qa`**. CI runs (lint, typecheck, tests). Requires **1 approval** from the team. No direct pushes.
3. **Merge into `qa`** → the deploy workflow builds images and deploys to the **QA VM** automatically.
4. **Open a PR from `qa` into `main`** to release. The `Branch Guard` workflow enforces that the source branch is exactly `qa` — no feature branch can bypass QA. Requires **2 approvals**.
5. **Merge into `main`** → the same deploy workflow builds images and deploys to the **prod VM** automatically.

## How it works

- One `.github/workflows/deploy.yml` runs for both branches. It reads the target environment from `github.ref_name` (`main` → `prod`, anything else → `qa`).
- GitHub Environments (`qa`, `prod`) provide **environment-scoped secrets and variables**, so `VM_HOST`, `PUBLIC_URL`, and `DOMAIN` automatically point at the right VM and domain.
- `Caddyfile` uses the Caddy placeholder `{$DOMAIN}` so a single file serves both environments. The deploy workflow writes `DOMAIN` into `.env` on the VM and `docker-compose.prod.yml` passes it to the caddy container, which Caddy expands at startup. Caddy auto-provisions a Let's Encrypt cert for whichever hostname the env var resolves to.
- `.github/workflows/branch-guard.yml` runs on every PR targeting `main` and **fails the check unless the PR's source branch is `qa`**. Branch protection on `main` requires this check to pass, so feature branches physically cannot merge directly to `main`.
- Slack notifications are prefixed with `[QA]` or `[PROD]` so the shared `#oh-sheet-notifications` channel clearly indicates which environment was deployed.

## Admin setup (one-time — requires repo admin)

The code changes in this repo enable the flow, but branch protection and GitHub Environments require admin rights to configure.

### 1. Create GitHub Environments

In **Settings → Environments**:

**`qa` environment**:
- No required reviewers
- Environment variables:
  - `VM_HOST = 34.169.16.93`
  - `VM_USER = deploy`
  - `PUBLIC_URL = https://oh-sheet-qa.duckdns.org`
  - `DOMAIN = oh-sheet-qa.duckdns.org`
- Environment secrets (copy from repo secrets): `VM_SSH_PRIVATE_KEY`, `WIF_PROVIDER`, `WIF_SERVICE_ACCOUNT`, `SLACK_WEBHOOK_URL`

**`prod` environment**:
- **Required reviewers**: at least 2 team members (optional — the 2-approval rule is also enforced at PR level, but this adds a runtime gate)
- Environment variables:
  - `VM_HOST = 104.196.254.221`
  - `VM_USER = deploy`
  - `PUBLIC_URL = https://oh-sheet.duckdns.org`
  - `DOMAIN = oh-sheet.duckdns.org`
- Environment secrets: same as qa

Once environment-scoped vars/secrets exist, the repo-level `VM_HOST`/`VM_USER` variables can be removed to avoid confusion.

### 2. Branch protection

**`qa` branch**:
- Require a pull request before merging
- Require approvals: **1**
- Dismiss stale pull request approvals when new commits are pushed
- Require status checks to pass: `Backend Lint`, `Backend Typecheck`, `Backend Tests`, `Frontend Lint`
- Require branches to be up to date before merging
- Do not allow bypassing (include admins)

**`main` branch**:
- Require a pull request before merging
- Require approvals: **2**
- Dismiss stale pull request approvals when new commits are pushed
- Require status checks to pass: `Backend Lint`, `Backend Typecheck`, `Backend Tests`, `Frontend Lint`, **`PR to main must come from qa`** (from `branch-guard.yml`)
- Require branches to be up to date before merging
- Do not allow bypassing (include admins)

## Infrastructure

Both VMs live in GCP project `oh-she3t`:

| VM | Zone | Machine | Static IP |
|----|------|---------|-----------|
| `oh-sheet-vm` (prod) | `us-west1-b` | `e2-small` | `104.196.254.221` |
| `oh-sheet-qa-vm` (qa) | `us-west1-b` | `e2-small` | `34.169.16.93` |

Both VMs share the same firewall tag (`oh-sheet-vm`) so the existing rules on ports 22/80/443 apply to both.

## Domains

| Environment | Domain | DNS |
|-------------|--------|-----|
| QA          | `oh-sheet-qa.duckdns.org` | DuckDNS A record → `34.169.16.93` |
| Prod        | `oh-sheet.duckdns.org`    | DuckDNS A record → `104.196.254.221` |

Caddy auto-provisions Let's Encrypt certificates on first deploy for whichever hostname the `DOMAIN` env var resolves to. If you later add a new environment or rename a domain, just update the env-scoped `DOMAIN` variable — no Caddyfile change required.
