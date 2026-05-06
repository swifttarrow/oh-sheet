# pop_eval_v1 — Phase 3 paid 30-song pop eval set

The hand-curated, paid 30-song pop eval set per
[`docs/research/transcription-improvement-implementation-plan.md`](../../docs/research/transcription-improvement-implementation-plan.md)
§Phase 3 and
[`docs/research/transcription-improvement-strategy.md`](../../docs/research/transcription-improvement-strategy.md)
Part III §3.

This is the release-gate eval set. The 30 songs are split 50/50 between
a tune set (engineer-readable, used for hyperparameter tuning) and a
holdout set (encrypted, used only at release gates).

> **Heads up.** Most slots are placeholders right now. The transcriber
> contract takes 4–8 weeks to deliver. Engineering should keep iterating
> against [`eval/pop_mini_v0/`](../pop_mini_v0/) (the 5-song
> reference-free mini-eval) until at least a handful of pop_eval_v1
> slots are populated.

## What's in the box

| File / dir | Purpose |
|---|---|
| `manifest.yaml` | The 30 song slots — license bucket, genre, intended source. |
| `songs/<slug>/` | Per-song artifact bundle (see below). Empty slots have `.gitkeep` only. |
| `holdout_manifest.yaml.enc` | Encrypted holdout split (15 of 30 slugs). Decryptable only with `OHSHEET_HOLDOUT_KEY`. |
| `../loader.py` | Yields `EvalSong` tuples; skips slots missing artifacts. |
| `../holdout.py` | 50/50 split + Fernet-encryption of the holdout manifest. |
| `../baselines/pop_eval_v1__baseline_<sha>.json` | Snapshotted run against the tune set. |

## Per-song artifact bundle

Once a slot is delivered, `songs/<slug>/` contains:

```
songs/<slug>/
├── source.audio.json          # license metadata + content_hash + sample_rate + duration
├── source.audio.{mp3,wav}     # IFF redistributable (FMA / CC-licensed track)
├── reference.piano_cover.mid  # contract-transcribed MIDI ground truth
├── reference.piano_cover.musicxml  # engraved reference
├── reference.piano_cover.pdf  # PDF of the engraved reference
├── structural.yaml            # human-verified key, time-sig, tempo, sections, chord progression, downbeats
└── notes.md                   # transcriber notes / known difficulties
```

Schema for `source.audio.json`:

```json
{
  "schema_version": 1,
  "content_hash": "sha256:…",
  "sample_rate": 44100,
  "duration_sec": 217.3,
  "format": "mp3",
  "license": "cc-by-3.0",
  "license_bucket": "fma_redistributable",
  "source_url": "https://freemusicarchive.org/...",
  "internal_storage_uri": null
}
```

`internal_storage_uri` is the private bucket pointer for
`commercial_sync_internal` slots (where audio is not committed). The
release-bot fetches the audio from there at eval time.

Schema for `structural.yaml` — directly loadable into the
`HarmonicAnalysis` Pydantic model
(`shared/shared/contracts.py:190-201`) extended with a
`downbeat_sec: list[float]` field and human-verified `sections` /
`chord_progression`. Mirrors the example in strategy doc §3.2.

```yaml
key: "C:major"
time_signature: "4/4"
tempo_bpm: 124.5
tempo_map:
  - {time_sec: 0.0, bpm: 124.5}
  - {time_sec: 60.0, bpm: 122.0}
sections:
  - {name: "intro",   start_sec: 0.0,  end_sec: 12.5,  bars: "1-4"}
  - {name: "verse_1", start_sec: 12.5, end_sec: 28.7,  bars: "5-12"}
  - {name: "chorus",  start_sec: 28.7, end_sec: 44.5,  bars: "13-20"}
chord_progression:
  - {start_sec: 0.0,  label: "C:maj"}
  - {start_sec: 4.2,  label: "F:maj"}
  - {start_sec: 8.4,  label: "G:7"}
downbeat_sec: [0.0, 1.95, 3.90, 5.85]
license: "FMA-commercial / Internal Research Only / SFM Custom $50"
```

## License buckets

| Bucket | Count | Redistribute? | Cost / song |
|---|---|---|---|
| `fma_redistributable` | 10 | ✅ Audio committed alongside the manifest. | $0 |
| `commercial_sync_internal` | 20 | ❌ Audio NOT committed. Pulled from a private bucket via `internal_storage_uri` at eval time. | $50 |

Per-bucket policy is defined in `manifest.yaml`. The loader treats both
identically once `source.audio.json` is present — for sync-licensed
tracks the loader resolves audio from the bucket URI instead of the
local path.

## Tune / holdout split

```bash
# Generate the encrypted holdout manifest from the tune/holdout split.
# Requires OHSHEET_HOLDOUT_KEY (passphrase) in the environment.
OHSHEET_HOLDOUT_KEY=$(uuidgen) python -m eval.holdout init eval/pop_eval_v1/

# List the tune set (no key needed)
python -m eval.holdout list-tune eval/pop_eval_v1/

# List the holdout set (KEY REQUIRED — fails without it)
OHSHEET_HOLDOUT_KEY=… python -m eval.holdout list-holdout eval/pop_eval_v1/
```

The split is deterministic — `init` is idempotent given the same
manifest contents and the same encryption key. Once published as
`pop_eval_v1.0.0`, the split is frozen; bug-fix releases (`v1.1.0`)
reuse the same split unless explicitly re-rolled.

The encryption uses Fernet (AES-128-CBC + HMAC-SHA-256) with a
PBKDF2-HMAC-SHA-256 derivation from the passphrase. The salt is stored
in plaintext in `holdout_manifest.yaml.enc.salt` so anyone with the
passphrase can decrypt; without the passphrase, the encrypted blob is
opaque.

## Versioning

- `pop_eval_v1.0.0` — initial 30-song release
- `pop_eval_v1.1.0` — bug-fix in `structural.yaml` for one or more songs
- `pop_eval_v2.0.0` — additional songs (genre coverage expansion)

Every CI run logs the eval-set version it ran against. Cross-version
comparisons require re-running the older release on the new eval set.

## Snapshotting baselines

```bash
# Tune set baseline — snapshot after each phase that ships transcribe /
# arrange / engrave changes.
python scripts/eval_mini.py eval/pop_eval_v1/ eval/runs/$(date -u +%Y%m%dT%H%M%SZ)__phase3_init/
cp eval/runs/<latest>/aggregate.json \
   eval/baselines/pop_eval_v1__baseline_$(git rev-parse --short HEAD).json
```

The first such baseline is the **first real-pop number** for Oh Sheet
(per Phase 3 acceptance + strategy doc §9). The estimate going in is
`F1 ∈ [0.05, 0.15]`; the actual number sets Phase 4–6 acceptance bars.
