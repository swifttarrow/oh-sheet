# Demo Snapshot — 2026-04-20 (working state)

Reference state for demo rollback. Use this if the engraving rework breaks
something before the next demo (2026-04-23).

## Git tags (already pushed)

| Repo | Tag | SHA | What's in it |
|---|---|---|---|
| TuneChat master | `demo-snapshot/2026-04-20` | `967f685` | Post-Apr-20: PR #2+#3 merged, PR #4 reverted |
| oh-sheet main (prod) | `demo-snapshot/2026-04-20-prod` | `5532577` | Merged PRs #86-#96. TuneChat API key synced. |
| oh-sheet qa | `demo-snapshot/2026-04-20-qa` | `e1e427f` | Same + PR #95 engrave fallback fix |

## Railway TuneChat state

**Plan**: Pro
**Resources**: 8 vCPU, 8 GB RAM (can go up to 24 GB per cgroup memory.max)
**Service**: `TuneChat` in project `radiant-curiosity` (production env)
**Public URL**: <https://tunechat.raqdrobinson.com>

### Env vars (non-secret — OK to commit)

```
MKL_NUM_THREADS=4
MSCORE_BIN=mscore3
NUMEXPR_NUM_THREADS=4
OMP_NUM_THREADS=4
PORT=3000
PUBLIC_BASE_URL=https://tunechat.raqdrobinson.com
TUNECHAT_PREVIEW_SECONDS=0
```

### Secrets (set in Railway dashboard only — do not commit)

```
TRANSCRIBE_API_KEY=<64-char hex, ends b9f1c84e>
```

### Critical env-var context

- **`OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`** — without these, torch
  defaults to `nproc`=48 (host cores visible to container), spawning 48
  threads per transkun process that fight for 8 vCPU. Result: 600s cap
  hits on every job. If these env vars disappear, transcribe dies.
- **`TUNECHAT_PREVIEW_SECONDS=0`** — disables preview-clip generation,
  saving 3 of 5 MuseScore subprocess invocations per job (~40-90s per
  job). Re-enabling means slower pipeline + more mscore fragility.

## GCP state (to capture after gcloud auth refresh)

- **qa VM**: `oh-sheet-qa-vm` in `us-west1-b`, external IP `34.169.16.93`
- **prod VM**: `oh-sheet-vm` in `us-west1-b`, external IP `104.196.254.221`
- **Artifact Registry images in use** (tagged by commit SHA):
  - `us-central1-docker.pkg.dev/oh-she3t/oh-sheet/app:5532577...` (prod main)
  - `us-central1-docker.pkg.dev/oh-she3t/oh-sheet/app:e1e427f...` (qa)

### TODO after gcloud auth refresh

```bash
# Snapshot boot disks for point-in-time rollback capability
gcloud compute disks snapshot oh-sheet-qa-vm \
  --snapshot-names=oh-sheet-qa-2026-04-20 \
  --zone=us-west1-b

gcloud compute disks snapshot oh-sheet-vm \
  --snapshot-names=oh-sheet-prod-2026-04-20 \
  --zone=us-west1-b
```

## Rollback runbook

### If engraving rework breaks oh-sheet on qa

1. Trigger a deploy of the pinned commit to qa:
   ```bash
   git push origin demo-snapshot/2026-04-20-qa:qa --force-with-lease
   ```
   Auto-deploys via GitHub Actions. ~12 min.

### If engraving rework breaks TuneChat on Railway

1. Local:
   ```bash
   cd tunechat
   git reset --hard demo-snapshot/2026-04-20
   git push origin master --force-with-lease
   ```
   Railway auto-deploys from master. ~2-4 min.
2. Verify: `curl https://tunechat.raqdrobinson.com/health` should return `{"ok":true}`
3. Verify env vars still set (should persist across redeploys; if wiped,
   re-set from "Railway TuneChat state" section above).

### If a VM is in a bad state

1. Restore the boot disk from snapshot (if created):
   ```bash
   gcloud compute disks create oh-sheet-qa-vm-restore \
     --source-snapshot=oh-sheet-qa-2026-04-20 \
     --zone=us-west1-b
   ```
2. Detach current disk, attach restored disk, or restart the VM.

## Known issues in the snapshot state (NOT fixes — things to work on)

- **MuseScore SMuFL fonts missing on Railway** → PDFs render blank
  (title + measure numbers only, no notation glyphs). OSMD browser
  preview works because it ships its own font. PDF download does not.
- **31% tie rate** in engraved scores due to MuseScore's default MIDI
  import; `simplify.js` exists but is not wired into `pipeline.js`.
- **6 voices assigned per 2 staves** from MuseScore's overlap-based
  voice splitting — visually chaotic output.
- **TuneChat job metadata is in-memory only** → "Job not found" after
  any container restart (demo saw this twice today).

These are candidates for the engraving rework. This document is the
known-good point to roll back to if the rework goes sideways.
