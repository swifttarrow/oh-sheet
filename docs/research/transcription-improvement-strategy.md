# Oh Sheet — Pop-Music Transcription Improvement Strategy

**A multi-team deep research deliverable**

| | |
|---|---|
| **Date** | 2026-04-25 |
| **Repo** | `/Users/ross/oh-sheet/` (branch: `main`, head: `5532577`) |
| **Scope** | End-to-end audio→sheet pipeline (Ingest → Transcribe → Arrange → Humanize → Engrave) |
| **Effort** | 13 specialist Opus 4.7 agents across 2 phases (10 discovery, 3 cross-pollination) |
| **Source corpus** | Codebase audit + ~150 research URLs (papers, repos, leaderboards) cited inline |

---

## How to read this document

This is the **single deliverable** of a 13-agent research session. It is structured for two audiences:

1. **Senior engineers and the product lead** who need an actionable plan: read the **Executive Summary** below (≈8 pages) and **Part II — Improvement Roadmap**.
2. **Implementers** working a specific stage: jump to **Part I — Pipeline Diagnosis** for evidence (every weakness is anchored to `file:line` in the codebase), **Part III — Evaluation Strategy** for how to verify any change, and **Parts IV–V** for the underlying technology survey and codebase deep-dives.

Each part is self-contained, but cross-references the others. Citations to research URLs are inline; citations to the codebase use absolute paths under `/Users/ross/oh-sheet/`.

---

## Table of contents

- [Executive Summary](#executive-summary)
- [Methodology](#methodology)
- [Part I — Pipeline Diagnosis (weaknesses, ranked)](#part-i--pipeline-diagnosis)
- [Part II — Improvement Roadmap (short / medium / long term)](#part-ii--improvement-roadmap)
- [Part III — End-to-End Evaluation Strategy](#part-iii--evaluation-strategy)
- [Part IV — Codebase Deep-Dives (architecture, transcribe, arrange/humanize/engrave)](#part-iv--codebase-deep-dives)
- [Part V — Technology Research (SOTA AMT, source separation, pop→piano, datasets, metrics, emerging)](#part-v--technology-research)

---

## Executive Summary

### The headline finding

**Oh Sheet's pipeline is structurally pessimistic.** Every stage discards information the next stage would benefit from; default configuration knobs throw away the work the previous knobs did; the eval set tests the easiest possible inputs (FluidSynth-rendered piano MIDI, not real pop); and the engraver receives the leanest possible artifact. The 2024–2026 literature has permissively-licensed, drop-in-shaped fixes for almost every individual gap — Demucs/Mel-RoFormer for separation, Kong/Aria-AMT/hFT for piano transcription, AMT-APC/Pop2Piano for pop→piano covers, Beat This! for beat tracking, BTC/ChordFormer for chords, MV2H/musicdiff/CLAP for evaluation. The biggest blockers are not unsolved research; they are **configuration defaults, contract drops, and the absence of a real-pop test corpus**.

### What the numbers say

- **Reported eval baseline** (the team's own benchmark): `mean_f1_no_offset = 0.368`, `mean_f1_with_offset = 0.130` (`/Users/ross/oh-sheet/eval-baseline.json:362-368`).
- **Reported per-role melody F1**: `0.093` (`eval-baseline.json:373`).
- **What that input actually is**: 25 MIDIs from `eval/fixtures/clean_midi/` synthesized to WAV via FluidSynth + TimGM6mb soundfont, then transcribed (`scripts/eval_transcription.py:1-100`).
- **What this measures**: Basic Pitch on its **easiest possible input** (clean stereo synth piano, perfect timing, no vocals/drums/codec). It tells us **nothing** about Pop2Piano (the actual default path — `backend/services/transcribe.py:52-119`, `config.py:244`), nothing about source separation, nothing about arrange/humanize/engrave correctness.
- **Projected end-to-end real-pop note-F1 vs. a hand-made cover**: **~0.05–0.15**, centered around **0.10**. Reasoning detailed in Part I §9.
- **For comparison**: hFT-Transformer reports 97.44% MAESTRO note-F1 ([arXiv:2307.04305](https://arxiv.org/abs/2307.04305)); Kong's piano transcription 96.72% onset / 91.86% pedal-onset ([arXiv:2010.01815](https://arxiv.org/abs/2010.01815)); Mel-RoFormer 17.66 dB instrumental SDR on MUSDB18-HQ ([arXiv:2409.04702](https://arxiv.org/abs/2409.04702)). The headroom between Oh Sheet's actual real-pop performance and 2026 SOTA is roughly **4–6×**.

### The five most damaging weaknesses

These are ranked by *fix-effort × impact*, so the items at the top are also the cheapest to ship.

1. **`score_pipeline = "condense_only"` is the default** (`backend/config.py:562`). The 580-line rules-based arranger and the 700-line melody/bass/chord extraction stack are computed and then **thrown away** — `condense.py` ignores `MidiTrack.instrument` and hand-splits everything at MIDI 60. **Fix: a one-line config change.**
2. **The engraver receives only MIDI bytes.** The entire `ExpressionMap` (dynamics, articulations, tempo_changes), every chord symbol, every section, the key signature, voice numbers, title, and composer are dropped at `backend/services/midi_render.py:36-137`. `EngravedScoreData` literally hard-codes `includes_dynamics=False, includes_pedal_marks=False, includes_chord_symbols=False, includes_fingering=False` (`backend/jobs/runner.py:516-520`). **Fix: pass MusicXML, not MIDI; days of work, no new ML.**
3. **Audio is collapsed to mono 22.05 kHz before any model touches it** (`transcribe_pipeline_pop2piano.py:65`, `transcribe_pipeline_single.py:74`). All stereo information and content above 11 kHz are destroyed upstream of inference. Demucs/Mel-RoFormer/HTDemucs all expect 44.1 kHz stereo; the current code makes proper source separation impossible. **Fix: keep audio at native 44.1 kHz stereo through ingest+separate; let each model wrapper handle its own resampling.**
4. **Hard cap of 2 voices per hand drops legitimate polyphony** (`backend/services/arrange.py:51-52, 226-227`). Pop piano covers routinely use 3–4 voices in the right hand for chord+melody. The literature on hand reachability uses 5 ([Nakamura & Sagayama 2018](https://arxiv.org/abs/1808.05006)). The 2-cap is half what is physically possible. **Fix: a one-line constant change.**
5. **The eval harness is FluidSynth-rendered piano MIDI, not real pop.** Real-pop transcription quality is **unmeasured**. Until that lands, no other improvement can be quantitatively validated. **Fix: hand-curate a 30–50-song internal pop eval set with paid human MIDI references (~$1.5k); 1 sprint of work.**

The full ranked weakness list is in Part I §8 ("Top 10 most damaging weaknesses overall").

### The recommended target architecture (~12 months out)

This is opinionated and license-clean. It threads a single coherent path through 2024–2026 SOTA while reusing Oh Sheet's existing orchestration scaffolding.

```
USER UPLOAD: MP3 / MIDI / YouTube URL
   │
   ▼
┌──────────────────────┐
│  ingest (yt-dlp)     │  → 44.1 kHz STEREO WAV (stop the mono downmix here)
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│   separate (NEW)     │  HTDemucs (default) / Mel-RoFormer (HQ mode)
└──┬──────┬────────┬───┘
   │      │        │
 vocals  drums   instrumental (= bass + other)
                  │
                  ▼
┌─────────────────────────────────┐
│       transcribe (REWORKED)     │
│   ┌──────────────────────────┐  │
│   │ "faithful" mode:         │  │
│   │   Kong (piano stem AMT)  │  │
│   │     + sustain-pedal head │  │
│   │   + Mel-RoFormer melody  │  │
│   │     on vocals stem       │  │
│   │   + Beat This! beat grid │  │
│   │   + Score-HPT velocity   │  │
│   ├──────────────────────────┤  │
│   │ "cover" mode:            │  │
│   │   AMT-APC end-to-end on  │  │
│   │   instrumental.wav       │  │
│   └──────────────────────────┘  │
│ + chord head (BTC / ChordFormer)│
│ + MSAF section segmentation     │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│      arrange (REWORKED)         │
│   - voice cap 4–5 per hand      │
│   - Cluster-and-Separate GNN    │
│     for voice/staff assignment  │
│     (replaces SPLIT_PITCH=60)   │
│   - velocity preserves dynamics │
│   - swing-aware quantize grid   │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│ humanize (OPT-IN, default OFF   │
│  for transcription products)    │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│      engrave (LOCAL)            │
│   PianoScore + Expression →     │
│     music21 → MusicXML          │
│       → Verovio (SVG)           │
│       → LilyPond (PDF)          │
│   includes_dynamics       = true│
│   includes_pedal_marks    = true│
│   includes_chord_symbols  = true│
│   includes_fingering      = true│
└─────────────┬───────────────────┘
              ▼
EngravedOutput { pdf_uri, musicxml_uri, midi_uri,
                 audio_preview_uri, evaluation_report }
```

**License posture:** every chosen model is MIT / Apache-2 / BSD / LGPL. The plan deliberately avoids YourMT3+ (GPL-3.0), SheetSage (CC-BY-NC-SA), MR-MT3 (CC-BY-NC-SA), NoteEM, Banquet, MoisesDB (all NC), and Pop2Piano (no LICENSE file — research-only).

### The 8-milestone shipping sequence (Sprint 0 → Q4)

| Milestone | Tasks | Acceptance |
|---|---|---|
| **Sprint 0** | Build the 30-song pop eval set with paid human transcriptions ($900–1.5k) | Held-out, frozen, versioned. |
| **M1 — Quick wins** (1 sprint) | A1–A13 (config flips, chord-symbol plumbing, autouse-fixture removal, README/code drift) | Synth-piano F1 ≥ 0.4 in CI; real-pop melody F1 ≥ 0.20 (was 0.093); LH note count ≥ 1.5× current. |
| **M2 — Source separation** (1 sprint) | B1: HTDemucs as a first-class stage | A/B real-pop melody F1 lift ≥ +0.05 vs M1. |
| **M3 — Kong + Beat This!** (1 sprint) | B2 (Kong piano AMT with pedal events), B3 (Beat This! replaces madmom) | 30-song eval Note F1 ≥ 0.45; pedal events visible on 80%+ of pop with pedal; Beat F1 ≥ 0.75. |
| **M4 — Local engraver** (2 sprints) | A5 final + B7 (music21 → MusicXML → Verovio + LilyPond) | `pdf_uri` non-null on 95%+ of jobs; pianists rate engraved output ≥ 3.5/5. |
| **M5 — Score-HPT velocity** (1 sprint) | B5 | MAESTRO Note+Off+Vel F1 ≥ 0.85. |
| **M6 — AMT-APC cover mode** (1 sprint) | B6 | A/B preference (≥30 listeners) ≥ 55% on Musicality vs faithful path. |
| **M7 — Eval ladder in CI** (1 sprint) | Tier 2 + Tier 3 + Tier 4 metrics on every PR; nightly Tier 1 + MV2H | <15 min CI runtime; JSON `evaluation_report` artifact per run. |
| **M8 — Voice/staff GNN** (1 sprint) | B8 (Cluster-and-Separate GNN replaces middle-C split) | Voice/staff F1 ≥ 0.80 on a labeled 20-song hand-split eval. |
| **Q3–Q4 research bets** | C1 (self-collected pop+piano-cover corpus + AMT-APC fine-tune), C2 (MR-MT3-equivalent retrain on permissive data), C3 (sight-readability critic), C4 (Anticipatory Music Transformer infill) | Scrape ≥ 4,000 pairs after filtering, label-noise ≤ 10%; classifier predicts Henle level within ±1 grade on 80% of CIPI; subjective preference ≥ 50% vs rule-based arrange. |

### The three numbers to put on every PR

The full evaluation strategy is in Part III, but if you read nothing else: report these three reference-free metrics on every PR:

1. **Tier 2 — chord accuracy** (`mir_eval.chord` `mirex` mode), audio-side anchor against the original input. Captures *recognizability*. Target: ≥ 0.70 on pop. Regression gate: −3 ppt.
2. **Tier 3 — playability fraction** (custom over `music21`: % of chords with span ≤ 14 semitones AND ≤ 5 notes per hand AND no impossible voice crossings). Captures *playability*. Target: ≥ 0.80. Regression gate: −5 ppt.
3. **Tier 4 — CLAP-music cosine** between the original audio and the FluidSynth-resynthesized engraved MIDI. Captures *perceptual similarity*. Target: ≥ 0.55. Regression gate: −0.05 absolute.

These three run in <2 minutes on a 5-song subset, are entirely reference-free, and triangulate the product surface that single F1 cannot. Everything else (Tier 1 mir_eval suite, MV2H, FAD, Tier 5 human studies) sits behind a nightly or release gate.

### What this document does NOT recommend

- **Do not adopt Pop2Piano** (current default at `config.py:244`) for production. Its upstream repo `sweetcocoa/pop2piano` ships with **no LICENSE file**; "all rights reserved" by default makes commercial use risky. AMT-APC ([github.com/misya11p/amt-apc](https://github.com/misya11p/amt-apc), MIT) is the documented MIT-licensed equivalent built on hFT-Transformer.
- **Do not adopt YourMT3+** despite its 96.5% MAESTRO / 74.8% Slakh Multi-F1 numbers. **GPL-3.0** would contaminate Oh Sheet's commercial product unless isolated as a microservice — and the additional engineering cost erases its quality advantage over Kong+AMT-APC.
- **Do not adopt SheetSage** for chord/lead-sheet extraction. Weights are **CC-BY-NC-SA 3.0**.
- **Do not adopt hFT-Transformer directly** despite its leaderboard SOTA. No pedal head, MAESTRO-overfit (drops ~12 F1 ppt cross-domain on MAPS — Edwards et al. 2024 ([arXiv:2402.01424](https://arxiv.org/abs/2402.01424))), no production-ready pip wrapper. AMT-APC is the better packaged descendant.
- **Do not bet on multimodal Audio LLMs as direct transcribers in 2026.** CMI-Bench (2025) shows current Audio LLMs (Gemini, GPT-4o, Qwen2.5-Omni) "drop significantly" on audio→symbolic tasks. Revisit in 6–12 months. (Track separately.)
- **Do not delete the existing arrange/humanize/engrave stages until after M4 lands.** They're flawed but not yet replaceable; the local-engraver and AMT-APC paths both depend on the upstream contract surface those stages already define.

### How much does this cost?

- **Sprint 0 eval set**: $900–1.5k for paid transcriptions (one-off).
- **Q3 self-collected corpus (C1)**: ~$30k all-in ($24k ML eng for 1 month + transcriber budget for spot-checks + cloud).
- **Q1 release-gate human study**: $300–800 per release on Prolific.
- **Engineering**: ~6 engineer-weeks across M1–M3, ~6 more for M4–M7, ~2 for M8. About **3 calendar months of one full-time engineer** to ship M1–M7 (the chunk that delivers the user-visible quality lift).
- **Compute**: container image grows from ~700 MB to ~1.5 GB; per-song CPU latency from 30–60 s to 2–4 min on Cloud Run 2-vCPU (or 30–60 s if a GPU pool is added). Disk: 4 GB peak per worker.

### One-paragraph defense

The biggest question reviewers will ask is: *why threading multiple specialist models (Demucs + Kong + Mel-RoFormer + Beat This! + Score-HPT + AMT-APC + music21/Verovio/LilyPond) instead of a single end-to-end audio→sheet model?* Three reasons. **(1) None exists** — every "audio→sheet" model in 2026 is either a pop-cover generator (Pop2Piano, AMT-APC) or a symbolic transcriber (Kong, hFT, Aria-AMT) plus a separate engraver. The product surface is genuinely a pipeline. **(2) The threaded specialists are all small (10–340 MB checkpoints), all permissively licensed, and all pip-installable** — total integration cost is bounded and verifiable. **(3) Per-stage diagnostics are the *only* way to localize regressions** in a pipeline this complex; a single end-to-end model would shrink the eval ladder back down to a single number, which Part III demonstrates is exactly the failure mode Oh Sheet is in today. The threaded specialist plan is the unique design that simultaneously fixes today's problems, opens the product to a wider quality ceiling, and remains debuggable.

---

## Methodology

This document was produced by 13 specialist Opus 4.7 agents, in two phases:

### Phase 1 — discovery (10 agents in parallel)

| # | Agent | Subagent type | Output |
|---|---|---|---|
| 01 | `codebase-pipeline` | code-explorer | Architecture map, contract walkthrough, ranked weaknesses with file:line evidence. |
| 02 | `codebase-transcribe` | code-explorer | Basic Pitch wiring, every parameter, post-processing, stub fallback. |
| 03 | `codebase-arrange-engrave` | code-explorer | Arrange/humanize/engrave audit; PianoScore field-by-field; engraving library reality check. |
| 04 | `research-amt-sota` | general-purpose | SOTA polyphonic AMT 2024–2026: Onsets&Frames, MT3, MR-MT3, hFT, YourMT3+, Aria-AMT, Basic Pitch comparison. |
| 05 | `research-source-separation` | general-purpose | Demucs v4 / HTDemucs / Mel-RoFormer / BS-RoFormer / MDX-Net / Banquet; SDR numbers; integration patterns. |
| 06 | `research-pop-piano` | general-purpose | Pop2Piano, AMT-APC, PiCoGen2, Etude; license analysis; SheetSage. |
| 07 | `research-piano-models` | general-purpose | Onsets&Frames, Kong, hFT-Transformer, HPPNet-sp, Aria-AMT, Streaming Piano Transcription; OOD-fragility evidence. |
| 08 | `research-eval-metrics` | general-purpose | mir_eval, MV2H, musicdiff, OMR-NED, FAD, CLAP-music, MERT, perceptual-MOS predictors. |
| 09 | `research-datasets` | general-purpose | MAESTRO, MAPS, ASAP, GiantMIDI, Aria-MIDI, POP909, Slakh2100, MUSDB18-HQ, MoisesDB; license matrix. |
| 10 | `research-emerging` | general-purpose | Multimodal LLMs, diffusion AMT, Beat This!, Score-HPT, Mel-RoFormer melody, Cluster-and-Separate GNN, Anticipatory Music Transformer. |

Each agent saved a thorough report to `/tmp/oh-sheet-research/0X-*.md` (≈170–320 lines / 16–31 KB each). All Phase 1 reports are reproduced in **Parts IV and V** of this document.

### Phase 2 — synthesis (3 agents in parallel)

| # | Agent | Reads | Output |
|---|---|---|---|
| 11 | `synth-weaknesses` | All Phase 1 reports | Cross-referenced ranked weakness analysis (Part I). |
| 12 | `synth-improvements` | All Phase 1 reports | Three-horizon improvement roadmap (Part II). |
| 13 | `synth-eval-strategy` | All Phase 1 reports | 5-tier metric ladder + harness design (Part III). |

The synthesis agents were given each other's task descriptions but read only the Phase 1 corpus (not each other's output) to avoid reasoning collusion. The integrated narrative across Parts I/II/III was produced by the orchestrator (this document) using all 13 reports.

### Reproducibility

- Source repo state: branch `main`, head `5532577` (Merge PR #96 from `qa`).
- All 13 source reports are saved at `/tmp/oh-sheet-research/`.
- Every URL cited in the synthesis agents is reproduced inline in **Parts IV and V**.
- Every code citation is `file:line` against the working tree captured at session start (see `gitStatus` in the session header).

---


# Part I — Pipeline Diagnosis

_Synthesized by Phase 2 agent `synth-weaknesses`. Cross-references the Phase 1 codebase audits (Part IV) with the technology research (Part V) to produce a ranked, evidence-backed weakness analysis. Every weakness is anchored to file:line in the codebase AND to the SOTA literature that demonstrates a fix is achievable._

## Oh Sheet — Synthesis: Weaknesses of the Pop→Piano Pipeline vs SOTA

**Author:** Phase-2 research synthesis agent
**Date:** 2026-04-25
**Audience:** senior engineers triaging quality investments
**Inputs:** Phase-1 reports 01–10 in `/tmp/oh-sheet-research/`. Code paths are absolute under `/Users/ross/oh-sheet/`. Research URLs are inline.

---

### 0. Executive read

Oh Sheet's pipeline is a careful but **structurally pessimistic** design: every stage discards information that the next stage needs, the default knobs throw away most of the work the previous knobs did, the eval set tests the easiest possible inputs, and the engraver receives the leanest possible artifact. The literature has 2-5 year-old, permissively-licensed, drop-in-shaped fixes for nearly every individual gap (Demucs/Mel-RoFormer, Kong/Aria-AMT/hFT, AMT-APC/Pop2Piano, Beat-This!, BTC/ChordFormer, MV2H/MOS, MV2H/musicdiff). The biggest blockers are not unsolved research — they are configuration defaults, contract drops, and an absent end-to-end pop test corpus.

The eval-baseline F1 of **0.368 no-offset / 0.130 with-offset** on 25 fluidsynth-rendered MIDIs (`/Users/ross/oh-sheet/eval-baseline.json:362-368`, summarized in `01-codebase-pipeline.md` §8) is the *floor of the easy case*; the per-role melody F1 of **0.093** (`eval-baseline.json:373`) inside that easy case is the alarming number. We project end-to-end real-pop note-F1 vs a hand-made cover at **0.05–0.15** depending on input (§ "Conjecture").

---

### 1. Theme A — Audio acquisition and preprocessing

| # | Severity | Weakness | Code citation | SOTA gap | Fix hint |
|---|---|---|---|---|---|
| A1 | **CATASTROPHIC** | All audio collapsed to mono 22.05 kHz before any model touches it. Stereo pan / >11 kHz content is destroyed upstream. | `/Users/ross/oh-sheet/backend/services/transcribe_pipeline_pop2piano.py:65`; `/Users/ross/oh-sheet/backend/services/transcribe_pipeline_single.py:74`; `/Users/ross/oh-sheet/backend/services/transcribe_pipeline_stems.py:407` | Demucs, Mel-RoFormer, BS-RoFormer, htdemucs all expect **44.1 kHz stereo** (`05-research-source-separation.md` §6.4). Aria-AMT, hFT-Transformer, Kong all train at ≥16 kHz mono with no stereo expectation, but lose nothing from preserving stereo because they downmix internally. The current behavior makes source separation *impossible* — a 22 kHz mono signal is not what any 2024-2026 separator was trained on. | Move resampling to **inside** each model wrapper. Keep the stage I/O at 44.1 kHz stereo. ([Demucs README](https://github.com/facebookresearch/demucs), `05` §6.4) |
| A2 | **HIGH** | No source separation runs by default in the single-mix path; on Pop2Piano path every instrument including drums and vocals is jointly transcribed as piano notes. | `/Users/ross/oh-sheet/backend/services/transcribe.py:52-119` (dispatch), `/Users/ross/oh-sheet/backend/services/transcribe_pipeline_pop2piano.py:65` (mono load) | Mel-RoFormer instrumental SDR ~17.66 dB on MUSDB-HQ (`05` §1, MVSep). Pop2Piano paper itself recommends Demucs preprocessing for non-K-pop ([HF docs](https://huggingface.co/docs/transformers/model_doc/pop2piano)). Whisper-ALT showed +3.6 ppt WER drop with MSS-vocals on dense mixes ([arXiv:2506.15514](https://arxiv.org/abs/2506.15514)). | Insert a Mel-RoFormer or Demucs htdemucs Separate stage between Ingest and Transcribe (`05` §4 Pattern D). Cost: 30-60 s/song on Cloud Run 2 vCPU. |
| A3 | **HIGH** | `audio_preprocess_enabled=False` default — HPSS / RMS normalize / peak limit never run. | `/Users/ross/oh-sheet/backend/config.py:108` | Mobile-AMT shows +14.3 F1 ppt on realistic audio with EQ + reverb + noise + pitch-shift augmentation pipeline ([Kusaka & Maezawa EUSIPCO 2024](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000036.pdf)). Edwards et al. report Kong loses 19.2 F1 ppt to pitch-shift, 10.1 ppt to reverb on MAPS without explicit aug ([arXiv:2402.01424](https://arxiv.org/abs/2402.01424)). | Either enable HPSS+normalize on by default, or train against augmented data; the best path is the second (`07` §6 Phase 3). |
| A4 | **HIGH** | `minimum_frequency` / `maximum_frequency` never passed to Basic Pitch's `predict()`. Sub-bass rumble and kick-drum transients become "notes". | `/Users/ross/oh-sheet/backend/services/transcribe_inference.py:123-129`; `02-codebase-transcribe.md` §3 | Spotify Basic Pitch defaults are `None` (no floor); the model itself was trained from A0 (27.5 Hz) up. Setting `minimum_frequency=32` would suppress kick-as-note artifacts. No SOTA model leaves this unset for full-mix work. | One-line fix: pass `minimum_frequency=settings.minimum_frequency_hz` (default 30 Hz). |
| A5 | **MEDIUM** | yt-dlp re-encode of YouTube AAC → 44.1 kHz WAV stereo introduces one lossy hop before downsample to 22.05 kHz mono in Transcribe. | `/Users/ross/oh-sheet/backend/services/ingest.py:99-101` | Marginal vs A1; combined effect with A1 is "double-decoded mono signal at half-bandwidth into models trained at full bandwidth". Aria-AMT's 100k-hour corpus was YouTube-derived but kept at 16+ kHz mono ([ICLR 2025](https://arxiv.org/abs/2504.15071)) — not 22 kHz mono. | Skip yt-dlp's re-encode and pass the source codec straight to the resampler. |
| A6 | **MEDIUM** | `RemoteAudioFile` contract carries no bit-depth, codec, original-SR-before-yt-dlp-encode, or channel layout. | `/Users/ross/oh-sheet/shared/shared/contracts.py:42-49`; `01-codebase-pipeline.md` §4 | Robust transcribers benefit from knowing the input chain (Edwards et al. 2024 explicit augmentation by codec). Without this metadata, downstream stages can't even decide whether HPSS is appropriate. | Extend the contract to carry codec + channel layout + content hash. |

**Theme A summary.** A1 + A2 together create the "before any model sees the signal, half its frequency content and all its spatial information is gone" problem. This is the single most fixable architectural gap because it does not require model swaps — it requires keeping audio at 44.1 kHz stereo through ingest + separate, then letting each model wrapper handle its own input format. The eval set in `eval/fixtures/clean_midi/` doesn't observe this because every input is fluidsynth-rendered piano MIDI (`01` §8) — the bandwidth and stereo problems literally cannot register in CI.

---

### 2. Theme B — Transcription model choice and configuration

| # | Severity | Weakness | Code citation | SOTA gap | Fix hint |
|---|---|---|---|---|---|
| B1 | **CATASTROPHIC** | Per-role melody F1 = **0.093** on the *easy* eval set (clean piano MIDI rendered via FluidSynth). | `/Users/ross/oh-sheet/eval-baseline.json:373`; `01-codebase-pipeline.md` §8 | hFT-Transformer = 97.44% MAESTRO note-F1 ([arXiv:2307.04305](https://arxiv.org/abs/2307.04305)); Kong = 96.72%; Onsets&Frames = 94.80% (`07` §2). Even Basic Pitch's own claimed MAESTRO note-F1 is ~88% (`04` §2.6). The Oh Sheet melody role's 0.093 is **not** Basic Pitch's quality — it's the quality of the post-Basic-Pitch Viterbi melody-extractor (`/Users/ross/oh-sheet/backend/services/melody_extraction.py`) reading off Basic Pitch's contour matrix, after the audio has been fluidsynth-rendered then mono-downsampled. The melody Viterbi voicing-floor + transition_weight combo is doing most of the damage (`02` §5). | Replace "Basic Pitch contour → Viterbi melody/bass split" with one of: (a) AMT-APC ([github MIT](https://github.com/misya11p/amt-apc)) which directly outputs the dual-staff piano cover; (b) Mel-RoFormer vocals stem → CREPE/RMVPE for the melody track only ([arXiv:2409.04702](https://arxiv.org/abs/2409.04702)); (c) Aria-AMT ([github Apache 2.0](https://github.com/EleutherAI/aria-amt)) for piano-only signal. |
| B2 | **CATASTROPHIC** | Pop2Piano is the actual default path (`pop2piano_enabled=True`, `config.py:244`), but it has **unresolved license status** — the upstream repo `sweetcocoa/pop2piano` does not contain a top-level LICENSE file. | `/Users/ross/oh-sheet/backend/config.py:244`; `06` §1.5 | "All rights reserved" by default makes commercial use risky. Pop2Piano paper notes 8th-note quantization, 4-beat coherence window, K-pop training distribution as known weaknesses (`06` §1.6). On non-K-pop / hip-hop / sparse-piano genres it can output 10-note 4-octave chords ([HN thread cited in `06`](https://news.ycombinator.com/item?id=34205996)). | Either (a) get a written license grant from Choi & Lee, or (b) replace Pop2Piano with **AMT-APC** ([MIT, github.com/misya11p/amt-apc](https://github.com/misya11p/amt-apc)) which builds on hFT-Transformer and is MIT-licensed end-to-end (`06` §2.4). |
| B3 | **HIGH** | Single global `onset_threshold=0.5`, `frame_threshold=0.3` for the full mix in single-mix path. No per-frequency-band thresholds. | `/Users/ross/oh-sheet/backend/config.py:45-46`; `02` §3 | Pop mixes routinely have 12-20 dB SPL differential between drums and quiet melody. SOTA approach: separate first, then per-stem thresholding (Demucs path already does this with vocals=0.6/0.25, bass=0.5/0.3, other=0.5/0.3 at `config.py:70-83`) — but only when the Demucs path is taken, which Pop2Piano default disables. | Move to the Demucs+per-stem thresholds path as default; consult Mobile-AMT's per-frequency augmentation ([EUSIPCO 2024](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000036.pdf)). |
| B4 | **HIGH** | Pitch bends are read by the Pop2Piano wrapper, then **discarded** at MIDI rebuild. `multiple_pitch_bends=False` to Basic Pitch. | `/Users/ross/oh-sheet/backend/services/transcribe_pop2piano.py:114` (`pitch_bends=[]`); `/Users/ross/oh-sheet/backend/services/transcribe_midi.py:50-54`; `02` §3 | Bent guitar / vocal notes / sliding bass quantize to nearest semitone. Pop2Piano paper does emit pitch bends; AMT-APC and Aria-AMT carry them through. Loss is silent — the contract `Note` doesn't even have a `bends` field (`/Users/ross/oh-sheet/shared/shared/contracts.py:157-161`; `01` §4). | Extend `Note` with `bends: list[PitchBend]`; pass `multiple_pitch_bends=True`. |
| B5 | **HIGH** | `vocals_enabled` only kicks in for Demucs path; on Pop2Piano default and on single-mix fallback every legato vocal phrase becomes a noisy run of polyphonic pitch detections. | `/Users/ross/oh-sheet/backend/services/transcribe_pipeline_stems.py:128-153` | Mel-RoFormer vocals SDR 12.98 + Whisper-ALT lyric-aware path ([arXiv:2506.15514](https://arxiv.org/abs/2506.15514)) — alternatively CREPE/RMVPE on a vocal stem give vocal-melody F1 in the 0.6-0.8 range vs 0.09 here (`05` §1, `06` §2.3). | Make Demucs+CREPE-on-vocals the **default** vocal path. |
| B6 | **HIGH** | Stub fallback emits hard-coded C-E-G-C arpeggio on any failure, masking real pipeline issues. | `/Users/ross/oh-sheet/backend/services/transcribe_result.py:249-278` | Silent failure mode. End-users get "a transcription" that is actually a placeholder. Eval can't see this because the eval harness shorts out at the inference step (`02` §6). | Replace stub with an explicit error; surface "transcription failed" to the user. |
| B7 | **MEDIUM** | Velocity is a linear `int(round(127*amplitude))` mapping with no perceptual loudness curve. | `/Users/ross/oh-sheet/backend/services/transcribe_result.py:46-47`; `02` §11 | Score-HPT ([arXiv:2508.07757](https://arxiv.org/abs/2508.07757)) achieves SOTA velocity refinement with 1M extra params (`10` §6). Even hFT-Transformer's velocity F1 collapses cross-domain (89.48 MAESTRO → 48.20 MAPS, `07` §3.3) — meaning *no* current AMT velocity is robust without domain-specific calibration. | Add a Score-HPT-style velocity refinement post-pass. |
| B8 | **MEDIUM** | `cleanup_chords_octave_amp_ratio = 0.5` and `cleanup_octave_amp_ratio = 0.6` heuristics drop legitimate octave doublings — the codebase comment in `config.py:139-140` itself acknowledges "real chord octave doublings are common." | `/Users/ross/oh-sheet/backend/services/transcription_cleanup.py:1-30, 53-62`; `/Users/ross/oh-sheet/backend/config.py:139-147` | Static thresholds can't separate intentional octave from artifact. Modern AMTs (hFT, Aria-AMT) emit them directly without a cleanup stage. | Replace with model-side suppression (per-track octave-doubling probability) or remove the cleanup entirely once a stronger transcriber is plugged in. |
| B9 | **MEDIUM** | `cleanup_energy_gate_max_sustain_sec = 2.0` — sustained whole-notes longer than 2 s are truncated. | `/Users/ross/oh-sheet/backend/config.py:144-147`; `01` §3 | A held pad / whole note in a ballad routinely exceeds 2 s. Aria-AMT and hFT both emit notes > 4 s freely. | Remove the cap, or compute sustain length from the audio's RMS envelope decay rather than a wall-clock threshold. |
| B10 | **MEDIUM** | `pop2piano_enabled=True` runs first by priority, then **Basic Pitch's contour matrix is also fed to a Viterbi split with `contour=None`** (the Pop2Piano path doesn't produce a contour). | `/Users/ross/oh-sheet/backend/services/transcribe_pipeline_pop2piano.py:73-74`; `01` §13 row 5 | Comment says "extractors handle this gracefully" but they actually skip back-fill (their main quality lever). The result is the worst of both worlds: Pop2Piano's known weak melody fidelity (PiCoGen MCA 0.25, Pop2Piano 0.17 — `06` §2.1) without the Viterbi melody recovery. | Either remove the post-Pop2Piano Viterbi entirely, or switch to AMT-APC which doesn't need the post-pass. |

**Theme B summary.** The transcribe stage is doing *too many things* — three pipelines glued together, each invalidating the others' assumptions. SOTA in 2024-26 has converged on either (a) "one trained model + good MSS upstream" (Aria-AMT, hFT) or (b) "one trained pop-cover model" (AMT-APC, Pop2Piano). The current architecture sits in the worst spot: paying integration cost for all three while getting the quality of none.

---

### 3. Theme C — Polyphony, voicing, and arrangement

| # | Severity | Weakness | Code citation | SOTA gap | Fix hint |
|---|---|---|---|---|---|
| C1 | **CATASTROPHIC** | `score_pipeline = "condense_only"` is the default — completely bypasses the 580-LOC arrange.py and uses condense.py, which **ignores `MidiTrack.instrument`** (melody/bass/chord routing) and hand-splits everything at MIDI 60. | `/Users/ross/oh-sheet/backend/config.py:562`; `/Users/ross/oh-sheet/backend/services/condense.py:1-9, 86-103`; `01` §13 row 2; `03` §1 | Every other pop-piano system uses melody-aware staff assignment (Pop2Piano emits two tracks; AMT-APC emits two staves directly; AccoMontage / GETMusic treat melody and accompaniment separately; `06` §2). This default makes the 700-LOC melody/bass/chord pipeline computed-then-discarded. | Set default `score_pipeline="rules"` or `"hf_midi_identity"` (the latter is also a stub — `03` §1 — so really set to `"rules"`). Or remove condense_only entirely. |
| C2 | **CATASTROPHIC** | Hard cap of **2 voices per hand** in arrange.py; excess polyphony silently `continue`'d in the loop. | `/Users/ross/oh-sheet/backend/services/arrange.py:51-52, 226-227` (`continue # exceeds polyphony — drop`); `01` §3, `03` §1 | Pop piano covers routinely have 3-4 voices in the right hand for chord+melody. Practical hand-reachability is ~5 simultaneous notes ([Nakamura & Sagayama 2018, arXiv:1808.05006](https://arxiv.org/abs/1808.05006)); the literature on playability scoring uses 5 (`08` §3.5). The 2-cap is half of what is physically possible. | Bump `MAX_VOICES_RH = MAX_VOICES_LH = 5`; backed by Cluster-and-Separate GNN ([arXiv:2407.21030](https://arxiv.org/abs/2407.21030)) for principled voice assignment (`10` §10). |
| C3 | **HIGH** | `arrange_simplify_min_velocity = 55` drops every quiet note before engrave; LH bass and inner voices from BP routinely sit at velocity 30-50. | `/Users/ross/oh-sheet/backend/config.py:219`; `/Users/ross/oh-sheet/backend/services/arrange_simplify.py:99`; `01` §3, `03` §1 | The codebase's own tuning history note (`config.py:215-222`) says "LH untouched because bass notes from Basic Pitch rarely dip below 40" — i.e. they tuned this on a small number of takes where bass was loud, and it kills LH on most pop. SOTA velocity processing (Score-HPT, `10` §6) refines velocities upward; it doesn't *threshold* them. | Set min_velocity=20 or remove the threshold; let a velocity-refinement stage handle dynamics. |
| C4 | **HIGH** | Velocity globally remapped to [35, 120] mean ~75. Destroys dynamic range information from transcription before humanize sees it. | `/Users/ross/oh-sheet/backend/services/arrange.py:365-390`; `03` §1 | Humanize then *adds* its own velocity offsets via `sin(progress · π)` (`/Users/ross/oh-sheet/backend/services/humanize.py:81-87`) — synthetic phrase shape unrelated to the audio's actual loudness. Net effect: original dynamics squashed, then overlaid with a sine wave. MV2H velocity-component (`08` §3.1) on this output would be near-random. | Remove the global velocity remap; let humanize / Score-HPT operate on transcription-native velocities. |
| C5 | **HIGH** | Hand split is at MIDI 60 (middle C) for any non-MELODY/BASS track. No hand-position / reachability analysis. | `/Users/ross/oh-sheet/backend/services/arrange.py:43, 135-156`; `01` §3, `03` §1 | A tenor-range vocal cover (typical pop hook is C4-G4, sitting *exactly* at the split) gets shredded across hands. Cluster-and-Separate GNN ([arXiv:2407.21030](https://arxiv.org/abs/2407.21030)) and the Hierarchical Audio-to-Score IJCAI 2024 paper (`10` §10) both train models specifically for voice/staff assignment. Statistical Piano Reduction ([Nakamura & Sagayama 2018](https://arxiv.org/abs/1808.05006)) gives the classical recipe. | Replace with a learned voice/staff GNN, or at minimum a beat-windowed sliding-pitch-median split (e.g., split at the running median of melody pitch over an 8-beat window). |
| C6 | **HIGH** | Track-level `confidence < 0.35` → silent track drop. Pop2Piano emits one track with arbitrary confidence — entire piece can vanish. | `/Users/ross/oh-sheet/backend/services/arrange.py:53, 142-146`; `01` §13 row 11 | Confidence as `clamp(mean(amplitudes), 0.1, 1.0)` is a noise floor proxy, not transcription confidence. AMT-APC, hFT, Aria-AMT all emit per-note posterior probabilities; aggregation should be note-frequency-weighted, not mean-amplitude. | Remove the threshold; use per-note pruning instead of per-track. |
| C7 | **MEDIUM** | Quantization grid restricted to `[0.167, 0.25, 0.333, 0.5]`. No 1/32, no swung grid, no tempo-aware piecewise grid. | `/Users/ross/oh-sheet/backend/config.py:443-445`; `/Users/ross/oh-sheet/backend/services/arrange.py:77-108`; `01` §13 row 9 | Pop with 16th-note hi-hat patterns at 180+ BPM (most modern dance) gets smeared. Shuffle/swing (most rock'n'roll, blues, jazz-influenced pop) is misnotated. Beat-This! ([github.com/CPJKU/beat_this, ISMIR 2024](https://github.com/CPJKU/beat_this), `10` §7) explicitly handles swing in beat tracking. Pop2Piano paper acknowledges 8th-note-only quantization as a known weakness (`06` §1.6). | Add 1/32 to the candidate set; detect swing ratio from beat tracker and emit triplet feel where appropriate. |
| C8 | **MEDIUM** | Same-pitch dedup keeps loudest within tolerance — kills tremolos and trills. | `/Users/ross/oh-sheet/backend/services/arrange.py:178-191`; `01` §3 | Repeated-note ornaments are common in pop melodies (cf. Adele "Hello" verse). hFT and Aria-AMT preserve them. | Tighten the tolerance (currently 0.05 s? — verify `arrange.py`); or detect repeated-note clusters as a separate pre-pass. |
| C9 | **MEDIUM** | `transform.py` is a literal passthrough stub (`return score`). When `condense_only` (default), it is skipped entirely. | `/Users/ross/oh-sheet/backend/services/transform.py:11-15`; `/Users/ross/oh-sheet/shared/shared/contracts.py:417-419` | Planned home for "voicing, register, or style transforms." Without it, no place exists for register-aware reduction (LH octave-down for low pitches, RH octave-up for high pitches), block-chord inference, or fingering hints. | Either implement (with AccoMontage-3 or Anticipatory Music Transformer, `06` §2.5, `10` §8) or remove the stage from the contract. |

**Theme C summary.** The arrange/condense stack is two parallel, partly-overlapping implementations with a default (`condense_only`) that throws away the smarter one's output. The 2-voice cap is the most architecturally damaging single number in the codebase — it makes correct pop transcription **representationally impossible** before any other quality work matters.

---

### 4. Theme D — Symbolic structure inference

| # | Severity | Weakness | Code citation | SOTA gap | Fix hint |
|---|---|---|---|---|---|
| D1 | **HIGH** | Key estimator is **major/minor only** — no modal detection (Dorian, Mixolydian, etc.). | `/Users/ross/oh-sheet/backend/services/key_estimation.py:39-41`; `01` §13 row 8, `02` §11 | "Riders on the Storm" is Dorian, "Stairway to Heaven" verses are Aeolian, half of pop-rock has modal moments. Modern key detectors (madmom, ChordFormer-derived) handle modes; mir_eval.key supports modal labels (`08` §2.1). | Use a learned key detector (e.g. madmom's `KeyClassificationProcessor` or MERT-fine-tuned head, `10` §5). |
| D2 | **HIGH** | Time-signature detector picks 3/4 or 4/4 only. 6/8 ballads, 7/8, 12/8 shuffles all become 4/4. | `/Users/ross/oh-sheet/backend/services/key_estimation.py:38-44`; `02` §11 | "Hey Jude" coda is 4/4 but verse is 4/4-feels-like-12/8. "Bohemian Rhapsody" goes 4/4 → 6/8 → 4/4. Beat-This! ([ISMIR 2024](https://github.com/CPJKU/beat_this), `10` §7) infers meter; the Time-Signature-Detection survey ([PMC 8512143](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512143/)) documents standard approaches. | Replace with Beat-This! + downbeat tracker; emit confidence over a meter set including 6/8, 12/8, 3/4, 4/4, and 7/8. |
| D3 | **HIGH** | `sections` are always `[]` from transcribe. Only the Anthropic-backed `RefineService` LLM can fill them. Without `OHSHEET_ANTHROPIC_API_KEY`, dynamics list is permanently empty (humanize depends on sections). | `/Users/ross/oh-sheet/backend/services/transcribe_result.py:188`; `/Users/ross/oh-sheet/backend/services/humanize.py:101-114`; `03` §2 | MSAF ([Nieto & Bello 2015](https://github.com/urinieto/msaf)) does pop-section segmentation as a self-contained Python library at F1≈0.6-0.7 (`08` §2.6). MERT and music foundation models can emit structure heads (`10` §5). | Wire MSAF as the default sections producer; let RefineService overwrite if the LLM is available. |
| D4 | **HIGH** | Chord symbols are detected by `chord_recognition.py` (24-template HMM-smoothed) and propagated as `ScoreChordEvent` through arrange — but **silently dropped at engrave** because `midi_render.py` doesn't include them in MIDI. | `/Users/ross/oh-sheet/backend/services/chord_recognition.py`; `/Users/ross/oh-sheet/backend/services/midi_render.py`; `03` §3 (engrave audit table) | ChordFormer 2025 SOTA: 84.7% Root, 84.1% MajMin, 83.6% MIREX ([arXiv:2502.11840](https://arxiv.org/html/2502.11840v1), `06` §3.1). BTC + GPT-4o CoT ([arXiv:2509.18700](https://arxiv.org/abs/2509.18700)) adds 1-2.77% on top (`10` §7). Loss is not at chord recognition — it's at the engrave handoff. | Add chord-symbol track to the MusicXML envelope; teach the remote engraver to emit them. Or: emit chord symbols directly in the MIDI as text events. |
| D5 | **MEDIUM** | Chord recognition has **no inversion tracking** — explicit comment: "the root we return is the template root, not necessarily the lowest note." | `/Users/ross/oh-sheet/backend/services/chord_recognition.py:31`; `01` §12 | Inversions are pop-music-defining (e.g., "Let It Be" piano: I-V/3-vi-IV). Modern chord recognizers (BTC, ChordFormer) detect inversion. mir_eval.chord supports inversion in `tetrads` comparison (`08` §2.5). | Adopt BTC or ChordFormer; emit slash-chord notation. |
| D6 | **MEDIUM** | No downbeat extraction. Beat tracker is madmom DBN (good) but downbeats are not separately exposed in the contract. | `02` §8 (table); `/Users/ross/oh-sheet/backend/services/audio_timing.py:99-148` | Beat-This! emits both beats and downbeats with one model ([github](https://github.com/CPJKU/beat_this), `10` §7). Madmom has a `DBNDownBeatTrackingProcessor` not used. Without downbeats, bar lines in the engraved score have to be re-derived by the remote engraver from the meter alone. | Add `downbeats` to `HarmonicAnalysis`; pass through. |
| D7 | **LOW** | `tempo_map` clamps BPM to [30, 300]. | `/Users/ross/oh-sheet/backend/services/audio_timing.py:43-44`; `02` §11 | Reasonable for pop. Non-issue except for very slow ambient (<30 BPM ballad outros) or DnB at 170+ feel-like-85. | Leave alone; document. |

**Theme D summary.** Most structural metadata is *computed* somewhere in the pipeline and then *not delivered* to the engraver. The fixes are not "compute better" — they are "preserve the channel" (D3, D4, D6). MV2H's harmony component (`08` §3.1) is unscoreable on the current output because the chord channel is dropped.

---

### 5. Theme E — Engraving and output

| # | Severity | Weakness | Code citation | SOTA gap | Fix hint |
|---|---|---|---|---|---|
| E1 | **CATASTROPHIC** | Engrave is an external HTTP black box that receives **only MIDI bytes**. The entire `ExpressionMap` (dynamics, articulations, tempo_changes) is dropped. Only `pedal_events` survive as CC64/66/67. Chord symbols, sections, key signature, voice numbers, repeats, title, composer, tempo marking, lyrics — all dropped. | `/Users/ross/oh-sheet/backend/services/midi_render.py:36-137, 99-117`; `/Users/ross/oh-sheet/backend/services/ml_engraver_client.py:107-141`; `/Users/ross/oh-sheet/backend/jobs/runner.py:431-536, 514-528`; `03` §3 | This is the single largest *quality multiplier loss* in the pipeline. Even if every upstream stage were SOTA, the engraver receives the lossiest possible representation. MusicXML supports all of this; pretty_midi to MIDI does not. The team's own comment at `runner.py:299-302` acknowledges TuneChat (tcalgo + MuseScore) "produces much cleaner scores than Basic Pitch + music21" — i.e., they know the local path is worse. | Pass MusicXML to the engraver, not MIDI. Or: include all metadata in MIDI text events (Marker meta, Lyric meta, Cue Point meta) and teach the engraver to read them. |
| E2 | **CATASTROPHIC** | `EngravedScoreData` hard-codes `includes_dynamics=False, includes_pedal_marks=False, includes_fingering=False, includes_chord_symbols=False`. | `/Users/ross/oh-sheet/backend/jobs/runner.py:516-520`; `03` §3 (engrave audit) | These are explicit acknowledgement that no expression data is in the output. The contract literally says "the score has none of the things sheet music is for." | Compute these flags from what was actually included; never hard-code False. |
| E3 | **HIGH** | `pdf_uri = None` always. PDF output retired. Frontend renders MusicXML client-side via OSMD/VexFlow; OSMD config has `drawTitle: false, drawComposer: false`. | `/Users/ross/oh-sheet/backend/jobs/runner.py:524`; `03` §3 | The README claims "music21 → MusicXML, LilyPond → PDF" (`03` §3 audit; README says so). No PDF is ever produced. The "publication-quality engraving" claim is structurally false. Verovio, LilyPond, MuseScore CLI all offer permissively-licensed local PDF rendering (`03` §3 engraving library audit). | Add a Verovio/LilyPond/MuseScore-CLI PDF rendering pass after the engraver returns MusicXML. |
| E4 | **HIGH** | `_looks_like_stub` rejects responses < 500 bytes. If the upstream ML service is itself a stub returning skeleton MusicXML, jobs hard-fail rather than degrade gracefully. | `/Users/ross/oh-sheet/backend/services/ml_engraver_client.py:93-104`; `03` §3 | No local fallback exists at all (no music21, no LilyPond, no Verovio in `backend/services/` — `03` §3 library audit). An outage of `oh-sheet-ml-pipeline` makes the entire system non-functional. | Add a local Verovio fallback (~ 5 MB binary, MIT). |
| E5 | **HIGH** | No key signature in output MIDI even though `metadata.key` is detected and refined. | `/Users/ross/oh-sheet/backend/services/midi_render.py:36-137`; `03` §3 | Engraver must guess key from accidentals. mido / pretty_midi both support `KeySignature` events; one line of code is missing. | Emit `KeySignature` from `metadata.key` in `render_midi_bytes`. |
| E6 | **HIGH** | Voice number per `ScoreNote.voice` (1 vs 2) is not transmitted. | `/Users/ross/oh-sheet/backend/services/midi_render.py`; `03` §3 PianoScore audit | Voice/staff information has to be re-derived by the engraver from pitch range alone — exactly the failure mode the Cluster-and-Separate GNN ([arXiv:2407.21030](https://arxiv.org/abs/2407.21030)) was designed to fix (`10` §10). | Use MIDI track-per-voice (track 0 = RH-voice-1, track 1 = RH-voice-2, track 2 = LH-voice-1, track 3 = LH-voice-2); the engraver can read this. |
| E7 | **MEDIUM** | README ↔ code drift: README claims `engrave.py # music21 → MusicXML, LilyPond → PDF`, `POST /v1/stages/engrave`, "music21, LilyPond, pretty_midi" — none of which exist. | `/Users/ross/oh-sheet/README.md` lines 127, 248, 303 (per `03` §3); compare to `/Users/ross/oh-sheet/backend/services/` (no `engrave.py`) | New contributors will reasonably believe the engrave stage uses music21 + LilyPond. The lie surface is large. | Update README to describe the actual external HTTP architecture. |
| E8 | **MEDIUM** | The decomposer / assembler microservices are stub-only dead code: `STEP_TO_TASK` at `runner.py:49-57` has no `decomposer` / `assembler` entries; both microservices return shape-correct stubs (4-note C/E/G/C and 2-note score). | `/Users/ross/oh-sheet/svc-decomposer/decomposer/tasks.py:32-75`; `/Users/ross/oh-sheet/svc-assembler/assembler/tasks.py:29-69`; `01` §11, §13 row 17 | Misleading code surface for any contributor. | Either route real work to them, or delete. |

**Theme E summary.** The engrave hop is where most of the upstream quality is destroyed. Fixing this is structurally cheaper than fixing the transcribe stage because it doesn't require new models — it requires preserving channels that already exist. Compare the engraver-input information content today vs MusicXML-supported information content; the gap is the entire `ExpressionMap` plus chord symbols plus key signature plus voicing.

---

### 6. Theme F — Evaluation and feedback loops

| # | Severity | Weakness | Code citation | SOTA gap | Fix hint |
|---|---|---|---|---|---|
| F1 | **CATASTROPHIC** | Eval harness is **fluidsynth-rendered MIDI**, not real pop audio. F1=0.368 is the *easy* case. | `/Users/ross/oh-sheet/scripts/eval_transcription.py:1-100`; `/Users/ross/oh-sheet/eval-baseline.json:362`; `01` §8 | Aria-MIDI was deliberately built on YouTube audio because Disklavier-style data doesn't generalize to real-world recordings ([arXiv:2504.15071](https://arxiv.org/abs/2504.15071), `09` §1, `07` §3.2). Edwards et al. 2024 ([arXiv:2402.01424](https://arxiv.org/abs/2402.01424)) show standard MAESTRO-trained models drop 19.2 F1 ppt under pitch-shift, 10.1 ppt under reverb on MAPS — and Oh Sheet's eval doesn't even have MAPS-level realism. | Build a 30-50-song hand-curated pop eval set per `09` §3 / `08` §5.4. Cost ~$1.5k for transcribers. |
| F2 | **HIGH** | `skip_real_transcription` is autouse=True — every pipeline integration test runs against the 4-note stub fallback. | `/Users/ross/oh-sheet/tests/conftest.py:46-68`; `01` §9, `02` §6 | CI proves orchestration but says nothing about transcription quality. Configuration regressions in BP / Pop2Piano wiring are invisible. | Make `skip_real_transcription` opt-in (`@pytest.mark.usefixtures`). Add a marked `@pytest.mark.transcription_e2e` slow-suite that runs real BP on 3 fixture clips. |
| F3 | **HIGH** | No tests for humanize. No tests inspect MusicXML payload structure. ML engraver always faked with 100-byte stub. | `/Users/ross/oh-sheet/tests/`; `03` §3 test coverage gaps | The fake stubs every test uses (`conftest.py:24-27` `_FAKE_ML_MUSICXML`) is itself **< 500 bytes** so `_looks_like_stub` would reject it in production — but in tests the stub bypasses the check. Tests are not testing what production sees. | Add MusicXML structural assertions (via lxml + music21 parse); use `pytest.mark.engraver_e2e` for real upstream calls. |
| F4 | **HIGH** | No eval covers Pop2Piano, condense, arrange, humanize, or engrave correctness. Only Basic Pitch on synthesized clean piano. | `01` §8 | MV2H ([github.com/apmcleod/MV2H](https://github.com/apmcleod/MV2H)) provides a 5-component score (multi-pitch, voice, meter, value, harmony) for symbolic transcription quality, exactly suited to evaluating arrange+engrave (`08` §3.1). musicdiff for MusicXML quality (`08` §3.3). FAD/CLAP for re-synth perceptual similarity (`08` §4). None used. | Adopt the metric ladder in `08` §"Recommended metric ladder": chord-acc + playability + CLAP-music cosine for CI; MV2H + FAD + Tier-5 MOS for releases. |
| F5 | **MEDIUM** | One MIDI in the eval set fails to synthesize at all. | `/Users/ross/oh-sheet/eval-baseline.json:76`; `01` §8 | Silent reduction in eval coverage. | Triage and fix the synth failure or replace the file. |
| F6 | **MEDIUM** | `overall_confidence` floor of 0.1 hides total failures — a stub-fallback transcription returns confidence ≥ 0.1 indistinguishable from a real low-quality one. | `/Users/ross/oh-sheet/backend/services/transcribe_result.py:236-237`; `02` §12 | No SOTA model emits confidence in this band; confidence reporting in mir_eval-aligned formats uses note-level posterior (`08` §1). | Surface stub-fallback distinctly; don't conflate with low-confidence real transcription. |
| F7 | **MEDIUM** | No tracking of which model version / config produced which eval result. `eval-baseline.json` is a single file. | `/Users/ross/oh-sheet/eval-baseline.json` | Cannot do A/B regression analysis between releases. Aria-MIDI paper, hFT paper, Mobile-AMT paper all version their evaluation alongside the model checkpoint. | Add a `model_version`, `config_hash`, and `dataset_version` to each eval row. |

**Theme F summary.** The eval harness gives a number that looks like progress but measures the *easiest* slice of inputs. Real pop music has never been measured. Until F1 lands, no other improvement can be quantitatively validated.

---

### 7. Theme G — Architecture, engineering, license

| # | Severity | Weakness | Code citation | SOTA gap | Fix hint |
|---|---|---|---|---|---|
| G1 | **HIGH** | The runner runs sequentially even though Celery is set up — `for step in plan` with `await` on each (`runner.py:257-545`). A slow transcribe blocks the whole job. | `/Users/ross/oh-sheet/backend/jobs/runner.py:257-552`; `01` §13 row 10 | Production AMT pipelines (Aria-MIDI build, Pop2Piano CI) parallelize ingest+separate+transcribe. Demucs and Basic Pitch can run concurrently on separate stems — there's no dependency. | Express the plan as a DAG; let independent stages run concurrently via `asyncio.gather`. |
| G2 | **HIGH** | LocalBlobStore only — `transcribe_audio.py:30-33` raises on non-`file://` URIs. S3 / GCS deployments need code changes. | `/Users/ross/oh-sheet/backend/services/transcribe_audio.py:30-33`; `/Users/ross/oh-sheet/shared/shared/storage/`; `01` §6 | Cloud Run + GCS is the deployment target per `CLAUDE.md`. Without S3/GCS, model weights, blob caches, and result artifacts can't be shared across replicas. | Implement an S3/GCS BlobStore backend; the Protocol exists at `storage/base.py:12-26`. |
| G3 | **HIGH** | Pop2Piano upstream repo has no LICENSE file. Sheet Sage models are CC BY-NC-SA 3.0. NoteEM, MR-MT3, Banquet, RWC, MoisesDB, MUSDB18, MAESTRO are research-only. | `/Users/ross/oh-sheet/backend/services/transcribe_pop2piano.py`; `06` §1.5; `07` §6 hard-no list; `09` §7 | Commercial use is implicitly blocked across multiple stages (Pop2Piano default + any plausible better model: Sheet Sage / MR-MT3 / YourMT3+ are GPL or NC). Aria-AMT (Apache 2.0), AMT-APC (MIT), Kong (Apache/MIT), hFT (MIT) are the safe candidates. | Pin every external model + dataset to a known commercial-friendly license; remove or sandbox Pop2Piano. |
| G4 | **MEDIUM** | Stage timeouts default to 600 s; Pop2Piano on long audio can blow this with no partial result. | `/Users/ross/oh-sheet/backend/config.py:34`; `01` §13 row 16 | Aria-AMT batched is 131× real-time on H100; Pop2Piano single-batch on T4 is ~1× real-time. A 6-minute song touches the timeout. | Per-stage timeouts; chunked transcription with partial result emit. |
| G5 | **MEDIUM** | Decomposer / assembler microservices are dead code (G2 of `01`). | `/Users/ross/oh-sheet/svc-decomposer/decomposer/tasks.py:32-75`; `/Users/ross/oh-sheet/svc-assembler/assembler/tasks.py:29-69`; `01` §11 | Misleading. The engrave stage retired its Celery worker in favor of inline HTTP (`docker-compose.yml` line 113 comment). | Delete or wire to real implementations. |
| G6 | **MEDIUM** | 645-line `config.py` with 80+ transcription knobs implies extensive empirical tuning, yet eval F1 is 0.368. The configuration surface has been **over-tuned to a non-pop test fixture**. | `/Users/ross/oh-sheet/backend/config.py`; `01` §10 | Most modern AMT papers expose 3-5 knobs (Basic Pitch's own `predict()` exposes 9). The amount of tuning here is a smell. | Audit which knobs actually moved eval F1; delete the rest. |

**Theme G summary.** Engineering debt is real but not the bottleneck. Fix Themes A-E first; G is mostly cleanup to enable scaling.

---

### 8. Top 10 most damaging weaknesses overall

Ranked by "magnitude of quality loss × number of inputs affected × difficulty for downstream stages to recover."

| Rank | Weakness | Theme | Severity | Approximate fix effort | Impact if fixed |
|---|---|---|---|---|---|
| 1 | **`condense_only` default discards melody/bass/chord routing.** Hand-splits all tracks at MIDI 60 even when melody/bass were correctly identified upstream. | C1 | CATASTROPHIC | 1-line config change | Recovers 700 LOC of upstream intelligence; immediately enables hand-aware engraving. |
| 2 | **Engrave receives only MIDI; entire ExpressionMap + chord symbols + sections + key signature dropped.** Hard-coded `includes_*=False`. | E1, E2 | CATASTROPHIC | Days (MusicXML serialization in `midi_render.py`) | Restores everything upstream computed but currently discarded. Enables actual sheet music. |
| 3 | **Audio collapsed to mono 22.05 kHz before any model sees it.** Stereo and >11 kHz lost. | A1, A2 | CATASTROPHIC | Days (refactor sample-rate plumbing; add separate stage) | Enables Demucs/Mel-RoFormer to work at all; improves all downstream models simultaneously. |
| 4 | **2-voice cap per hand silently drops legitimate polyphony.** | C2 | CATASTROPHIC | 1-line + downstream propagation | Pop chord+melody RH demands 3-4 voices; without this no faithful pop arrangement is possible. |
| 5 | **Eval is on FluidSynth-rendered piano MIDI, not real pop.** No real-pop measurement exists. F1 = 0.368 is the easy case; real-pop F1 is unmeasured (likely <0.15). | F1, F4 | HIGH | Weeks (curate 30-50 song corpus, $1-2k transcription cost) | Without this, every other improvement is theoretical. |
| 6 | **`arrange_simplify_min_velocity = 55` drops every quiet note + global velocity remap to [35,120].** | C3, C4 | HIGH | 2-line config change | LH bass and inner voices return; dynamics survive into humanize. |
| 7 | **Per-role melody F1 = 0.093 driven by Viterbi melody-extraction off Basic Pitch contour rather than a melody-aware model.** | B1 | HIGH (and the fix has SOTA candidates: AMT-APC, Aria-AMT, Mel-RoFormer→CREPE) | Weeks (model integration) | 10-20× improvement in melody fidelity is the literature-supported expectation. |
| 8 | **Pop2Piano default has unresolved license; runs as primary path despite being commercial-risky.** | B2, G3 | HIGH | 1-line config change + AMT-APC integration | Removes legal risk; AMT-APC ([github MIT](https://github.com/misya11p/amt-apc)) is the documented MIT-licensed equivalent. |
| 9 | **Chord symbols + key signature + sections detected then dropped at engrave.** | D3, D4, E5 | HIGH | Days (MIDI text events or MusicXML envelope) | Engraved score becomes recognizable as the song. |
| 10 | **`skip_real_transcription` autouse=True; no test runs real Basic Pitch / Pop2Piano.** | F2 | HIGH | Hours (mark fixture opt-in; add slow suite) | All other improvements are now CI-protectable. |

These ten changes are *not* model swaps; they are configuration changes, channel preservations, and one eval-corpus build. They are the highest-leverage 1-2 sprints of work in the codebase.

---

### 9. Conjecture about real-world performance

#### Question
The eval-baseline F1 of **0.368 no-offset / 0.130 with-offset** comes from FluidSynth-rendered clean piano MIDI from `eval/fixtures/clean_midi/` — the easiest possible inputs (clean stereo synth piano, perfect timing, no vocals/drums/codec, single instrument). Per-role melody F1 inside that easy case is **0.093**. What would real-pop end-to-end note-F1 (vs a hand-made cover) be?

#### Reasoning chain

**Step 1: Locate the F1=0.368 in the literature's coordinate system.**
Basic Pitch's own claimed MAESTRO note-F1 is ~88% (`04` §2.6, author-reported). On synthesized piano MIDI we should expect *roughly comparable* numbers if the pipeline were doing nothing wrong post-Basic-Pitch. We are seeing **0.368** — a ~50-point gap. That gap is "the cost of arrange + cleanup + condense + role-routing + (default) condense_only" applied to Basic Pitch's output. In other words: even on Basic Pitch's easiest input distribution, the post-processing destroys >50 ppt of F1.

**Step 2: Apply known degradations from clean MIDI to real pop audio.**
- Mono 22.05 kHz collapse (Theme A): Edwards et al. 2024 ([arXiv:2402.01424](https://arxiv.org/abs/2402.01424)) shows MAESTRO-trained models drop 19.2 ppt under pitch-shift, 10.1 ppt under reverb. A YouTube MP3 has both, plus codec, plus mastering EQ. **Expected drop: 15-25 ppt** in raw transcription quality.
- No source separation (A2): drums + vocals injected as polyphonic notes. Whisper-ALT on dense mixes ([arXiv:2506.15514](https://arxiv.org/abs/2506.15514)) shows ~3.6 ppt WER improvement *with* MSS. AMT-on-pop-with-vocals literature is rougher; Pop2Piano's own subjective MOS on non-K-pop is ~3.0/5 vs human 3.77 (`06` §1.4) — a ~0.8/5 gap that translates to ~15 ppt in objective F1. **Expected drop: 10-20 ppt** beyond clean piano.
- Pop2Piano vs PiCoGen Melody Chroma Accuracy: 0.25 vs 0.17 (`06` §2.1) — note these are *cover-generation* MCAs, not transcription F1. The real-faithful comparator (AMT-APC) reports "more accurate than any existing models" but no concrete F1 (`06` §2.4). Transcription F1 of Pop2Piano on pop is unmeasured publicly.

**Step 3: Apply the Oh-Sheet-specific multiplier on top.**
The default `condense_only` (Theme C1) is doing post-processing that has already destroyed >50 ppt on the easy case. Real-pop input is a worse case. The same percentage degradation does *not* apply linearly to a smaller starting F1; we should expect the floor.

#### Estimate

Putting it together for a real-pop input vs a hand-made cover (the comparator the user implicitly wants):

- **Note-F1 (no offset) on melody role**: 0.093 (current easy-case) × {real-pop degradation factor 0.4-0.7} ≈ **0.04-0.08**.
- **Note-F1 (no offset) overall**: 0.368 (current easy-case) × {real-pop degradation factor 0.3-0.5} ≈ **0.10-0.18**.
- **Note-F1 (with offset)**: 0.130 × similar factor ≈ **0.04-0.07**.
- **Chord-progression accuracy (`mir_eval.chord` mirex scheme)** — chord recognition itself is pop-tuned and works on the audio not the transcribed MIDI, so this is ~**0.5-0.7** (decent), but per Theme D4, **the chord channel doesn't reach the engraved output anyway**, so its contribution to user-perceived quality is zero today.
- **Playability fraction (Tier 3 of `08`)**: limited mostly by the 2-voice cap (Theme C2); ~**0.7-0.9** because 2-voice content is by construction playable.

#### Single-number projection

**End-to-end real-pop note-F1 vs a hand-made piano cover: ~0.05-0.15**, centered around **0.10**.

For comparison, public papers on the "audio→piano cover" task report objective MCA in the 0.17-0.25 range (PiCoGen, Pop2Piano subjective quality `06` §2.1) — but those are *cover-generation* metrics, not transcription F1, and they include the model emitting plausible-but-wrong notes. A pure-transcription comparator on pop audio is essentially absent from the literature. The closest published number is Mobile-AMT's "+14.3 F1 pts on realistic audio" ([EUSIPCO 2024](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000036.pdf)) — reported on noise+reverb-augmented MAESTRO, not real pop, but it suggests a ~70%-of-MAESTRO-clean ceiling for current SOTA on realistic audio: **~0.65-0.70 F1 on realistic-but-clean piano**, and lower for full-mix pop where vocal/drum polyphony is the real obstacle.

So the ratio between "what's feasible with 2026 SOTA" and "what Oh Sheet actually delivers" is **roughly 4-6×**. This is the headroom.

#### Uncertainty drivers

1. **The 0.05-0.15 range is wide because no real-pop benchmark exists for this pipeline.** F1 was never run on a real pop song through the full Oh Sheet pipeline against a hand-made cover. Until F1 (Theme F) is built, this estimate is the best the literature supports.
2. **Pop2Piano vs Demucs+BP path balance is unknown.** If most jobs hit the Pop2Piano path, the per-piece variance is high (it works well on K-pop, badly on hip-hop / sparse arrangements). If most hit the Demucs path, average is more consistent but lower mean.
3. **Eval tolerance window (50 ms onset, 50 ms offset) is generous.** Tightening to 20 ms (perceptual MIDI) would cut these numbers by another ~30-50%.
4. **"Hand-made cover" as comparator is itself non-canonical.** A given pop song has many valid covers. F1 between two pianists' covers of the same song is itself often 0.4-0.6. This sets an upper bound on what a transcription system could ever score against any single cover.

#### Recommendation flowing from this estimate

Even if all 10 of §8's fixes were applied perfectly, real-pop end-to-end F1 vs a single hand-made cover would top out at **~0.40-0.55** (matching the inter-cover F1 ceiling) — and only after replacing the transcription core with AMT-APC or Aria-AMT (Theme B1, B2). The *user-perceived* quality, however, is gated more by Themes E (engrave) and C (arrange) than by the raw transcription number, because it's what reaches the rendered PDF/MusicXML that the user sees. **Fix Themes E + C + F1 first, then validate on real pop, then decide whether the model swap (B) is worth the complexity.**

---

### 10. Reference index

#### Code paths cited (all under `/Users/ross/oh-sheet/`)
- `backend/config.py` (645 LOC, lines 34, 45-47, 70-83, 108, 119-147, 154-156, 166-171, 187-192, 199-205, 215-222, 244, 293, 412-413, 438, 443-445, 457-461, 562, 608)
- `backend/contracts.py` (re-exports `shared/shared/contracts.py`)
- `backend/jobs/runner.py:49-57, 170-217, 257-552, 299-302, 303-358, 431-536, 514-528, 524`
- `backend/services/arrange.py:43, 51-52, 53, 56-58, 77-108, 135-156, 142-146, 155, 163-229, 178-191, 226-227, 261-358, 365-390, 427-517, 520-533`
- `backend/services/arrange_simplify.py:99`
- `backend/services/audio_preprocess.py:69, 190-227, 298, 326`
- `backend/services/audio_timing.py:1-29, 43-44, 99-148`
- `backend/services/chord_recognition.py:31`
- `backend/services/condense.py:1-9, 42-103, 86-103, 131-194`
- `backend/services/humanize.py:8, 34-94, 44-59, 48-58, 66-94, 81-87, 81-88, 101-114, 121-155, 162-189, 196-271, 254, 269`
- `backend/services/ingest.py:36, 99-101, 143-265, 276, 323-413`
- `backend/services/key_estimation.py:38-44, 39-44, 60-75`
- `backend/services/melody_extraction.py:53-56`
- `backend/services/midi_render.py:36-137, 99-117`
- `backend/services/ml_engraver_client.py:93-104, 103-104, 107-141, 107-173`
- `backend/services/refine.py:63-130; refine_prompt.py:31`
- `backend/services/transcribe.py:52-119, 84-96, 122-216, 142-186`
- `backend/services/transcribe_audio.py:24-33, 30-33, 31-33`
- `backend/services/transcribe_inference.py:35-58, 39-58, 53-57, 123-129, 144-164`
- `backend/services/transcribe_midi.py:50-54, 74`
- `backend/services/transcribe_pipeline_pop2piano.py:65, 73-74`
- `backend/services/transcribe_pipeline_single.py:73-74, 108-144`
- `backend/services/transcribe_pipeline_stems.py:128-153, 185-203, 230-243, 407, 486-491`
- `backend/services/transcribe_pop2piano.py:114, 120-170`
- `backend/services/transcribe_result.py:45, 46-47, 70-246, 188, 191-193, 236-237, 249-278, 255-264`
- `backend/services/transcription_cleanup.py:1-30, 53-62, 435-514, 460-461`
- `backend/services/transform.py:1-5, 11-15, 14-15`
- `backend/workers/transcribe.py:12, 20`
- `eval-baseline.json:76, 362-368, 373`
- `eval/fixtures/clean_midi/`
- `scripts/eval_transcription.py:1-100, 24-27`
- `shared/shared/contracts.py:42-49, 52-56, 130-143, 157-161, 164-168, 171-176, 185-190, 208-214, 250-265, 268-272, 279-291, 320-324, 339-346, 348-377, 393-422, 400-422, 406, 417-419`
- `shared/shared/storage/base.py:12-26, local.py:33-36`
- `svc-decomposer/decomposer/tasks.py:32-75, 60`
- `svc-assembler/assembler/tasks.py:29-69, 32-34`
- `tests/conftest.py:24-27, 46-68, 81-95`

#### Research URLs cited
- Aria-AMT / Aria-MIDI ICLR 2025: https://arxiv.org/abs/2504.15071 ; https://github.com/EleutherAI/aria-amt
- AMT-APC: https://arxiv.org/abs/2409.14086 ; https://github.com/misya11p/amt-apc
- Anticipatory Music Transformer: https://arxiv.org/abs/2306.08620
- Beat This! ISMIR 2024: https://github.com/CPJKU/beat_this
- BTC chord + LLM CoT: https://arxiv.org/abs/2509.18700
- ChordFormer: https://arxiv.org/html/2502.11840v1
- ChatMusician ACL 2024: https://arxiv.org/abs/2402.16153
- Cluster-and-Separate GNN: https://arxiv.org/html/2407.21030v1
- Demucs (htdemucs): https://github.com/facebookresearch/demucs
- DiffRoll: https://arxiv.org/abs/2210.05148 ; https://github.com/sony/DiffRoll
- Edwards et al. 2024 robust AMT: https://arxiv.org/html/2402.01424v1
- Etude 2025: https://arxiv.org/abs/2509.16522
- FAD: https://github.com/microsoft/fadtk ; https://arxiv.org/abs/2311.01616
- hFT-Transformer: https://arxiv.org/abs/2307.04305 ; https://github.com/sony/hFT-Transformer
- HookTheory HLSD: https://hooktheory.com/theorytab
- Kong (ByteDance): https://arxiv.org/abs/2010.01815 ; https://github.com/bytedance/piano_transcription ; https://pypi.org/project/piano-transcription-inference/
- LAION-CLAP: https://github.com/LAION-AI/CLAP
- Lakh MIDI: https://colinraffel.com/projects/lmd
- MAESTRO v3: https://magenta.tensorflow.org/maestro-wave2midi2wave
- MAPS: https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database
- MERT: https://arxiv.org/abs/2306.00107 ; https://huggingface.co/m-a-p/MERT-v1-330M
- Mel-Band RoFormer: https://arxiv.org/abs/2310.01809 ; https://arxiv.org/abs/2409.04702
- mir_eval: https://mir-eval.readthedocs.io/latest/
- Mobile-AMT EUSIPCO 2024: https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000036.pdf
- MoisesDB: https://github.com/moises-ai/moises-db
- MR-MT3: https://arxiv.org/abs/2403.10024
- MT3: https://arxiv.org/abs/2111.03017 ; https://github.com/magenta/mt3
- MUSDB18-HQ: https://sigsep.github.io/datasets/musdb.html
- MV2H: https://github.com/apmcleod/MV2H ; https://ismir2018.ismir.net/doc/pdfs/148_Paper.pdf
- musicdiff: https://pypi.org/project/musicdiff/
- MSAF: https://github.com/urinieto/msaf
- Music Flamingo (NVIDIA): https://research.nvidia.com/labs/adlr/MF/
- Nakamura & Sagayama 2018 (playability): https://arxiv.org/abs/1808.05006
- NotaGen 2025: https://arxiv.org/abs/2502.18008
- Onsets & Frames: https://arxiv.org/abs/1710.11153
- Pop2Piano: https://arxiv.org/abs/2211.00895 ; https://github.com/sweetcocoa/pop2piano ; https://huggingface.co/docs/transformers/model_doc/pop2piano
- POP909: https://github.com/music-x-lab/POP909-Dataset
- PerceiverTF: https://arxiv.org/abs/2306.10785
- PiCoGen / PiCoGen2: https://arxiv.org/abs/2407.20883 ; https://arxiv.org/abs/2408.01551
- RMVPE: https://arxiv.org/abs/2306.15412
- SCNet: https://arxiv.org/abs/2401.13276
- Score-HPT: https://arxiv.org/abs/2508.07757
- Sheet Sage: https://arxiv.org/abs/2212.01884 ; https://github.com/chrisdonahue/sheetsage
- Slakh2100: https://zenodo.org/records/4599666
- Streaming Piano Transcription (Niikura 2025): https://arxiv.org/html/2503.01362
- Towards Musically Informed Evaluation: https://arxiv.org/html/2406.08454v2
- Whisper-ALT (cascaded MSS+ASR): https://arxiv.org/html/2506.15514v1
- YourMT3+: https://arxiv.org/abs/2407.04822 ; https://github.com/mimbres/YourMT3

---

# Part II — Improvement Roadmap

_Synthesized by Phase 2 agent `synth-improvements`. Translates Phase 1 findings into a three-horizon engineering plan: Section A quick wins (no new ML, ≤1 sprint), Section B architectural changes (pip-installable models, 1–3 sprints), Section C research-bet experiments (1–2 quarters), Section D recommended target stack, Section E risks/costs/dependencies, Section F sequencing recommendation. Includes diff-level hints (file:line + before/after) for the quick wins._

## Oh Sheet — Phase 2 Synthesis: Improvement Roadmap

**Author:** synthesis agent (Phase 2)
**Date:** 2026-04-25
**Inputs:** Reports 01–10 in `/tmp/oh-sheet-research/`
**Audience:** Oh Sheet engineering — concrete, executable plan with diff-level hints, named technologies, expected gains, risks, and effort.

---

### Section A — Quick Wins (≤ 1 sprint, no new ML)

These changes are **config and code-only**. Each one is anchored to a specific finding from Phase 1 reports. Effort estimates assume one engineer; expected gains are qualitative unless a paper number is cited.

#### A1. Stop the default `score_pipeline=condense_only` swap-out
**Anchor:** `01-codebase-pipeline.md` §13 rank 2; `03-codebase-arrange-engrave.md` §"Score pipeline mode".
**Diff hint:** `backend/config.py:562` change `score_pipeline = "condense_only"` → `"arrange"`. Rename the `condense_only` branch to `condense_legacy` and gate it with a debug flag. In `shared/shared/contracts.py:417-419`, remove the silent-swap logic so the orchestration always honors the configured pipeline.
**Before:** `condense_only` is the default → condense.py ignores `MidiTrack.instrument`, splits at MIDI 60, never quantizes. The whole 700+ LOC melody/bass/chords pipeline is computed and then thrown away.
**After:** Real arranger runs by default; melody → RH, bass → LH, chord roles preserved.
**Expected gain:** Removes the largest "computed-then-discarded" loss in the pipeline. Subjectively the LH should stop being a chronological pile of below-middle-C notes and start being a bass-driven hand.
**Risk:** `tests/test_arrange.py` and any snapshot tests pinned to condense output will break. Re-baseline `eval-baseline.json`. **Effort:** 1 day code + 1 day re-baseline.

#### A2. Lift voice cap from 2 to 4 per hand, make it adaptive
**Anchor:** `01-codebase-pipeline.md` §13 rank 4; `03-codebase-arrange-engrave.md` §"Stage 1: ARRANGE".
**Diff hint:** `backend/services/arrange.py:51-52` change `MAX_VOICES_RH = MAX_VOICES_LH = 2` → `4`, expose as `arrange_max_voices_rh`/`lh` settings in `backend/config.py`. In `_resolve_overlaps` (`arrange.py:226-227`), instead of `continue # exceeds polyphony`, log a warning and merge the lowest-velocity overflow into a stem-shared chord rather than dropping it.
**Expected gain:** Pop piano covers routinely use 3–4 RH voices for chord+melody. Today they get silently dropped. Recovering them is a direct readability win.
**Risk:** Engraver may produce dense output that some users find unreadable; pair with a `target_difficulty` knob that downsamples voices for "beginner" mode. **Effort:** 0.5 day.

#### A3. Lower `arrange_simplify_min_velocity`, fix global velocity remap
**Anchor:** `01-codebase-pipeline.md` §13 rank 3; `03-codebase-arrange-engrave.md` §"Key failure modes" (arrange).
**Diff hint:** `backend/config.py:219` change `arrange_simplify_min_velocity = 55` → `25`. In `backend/services/arrange.py:365-390` (`_normalize_velocity`), replace the linear remap-to-[35,120] with a percentile-based remap that **preserves dynamic range relative to the median** (use 5th percentile → 25, 95th percentile → 110, don't clip).
**Expected gain:** Bass guitar fundamentals from BP routinely sit at 30-50; today every one of them is dropped. After this change LH will populate.
**Risk:** Increases note count; engraver may slow. **Effort:** 0.5 day.

#### A4. Bypass `humanize` for `sheet_only` and add a `bypass_humanize` toggle
**Anchor:** `03-codebase-arrange-engrave.md` §"Stage 2: HUMANIZE", §"Key failure modes" (humanize).
The humanize stage applies hard-coded ±5ms downbeat / +3ms backbeat nudges and a sin-curve dynamics shape. For a transcription product these are anti-fidelity.
**Diff hint:** `shared/shared/contracts.py:406` confirms `sheet_only` already skips humanize. Add `bypass_humanize: bool = False` to `PipelineConfig`. Add UI toggle "Faithful timing (no humanization)" in the upload screen. In `backend/jobs/runner.py` add `if config.bypass_humanize: skip("humanize")`.
**Expected gain:** Avoids ±5–8 ms timing perturbation that has no audio basis. For users who want a faithful transcription this is strictly better.
**Risk:** None — opt-in. **Effort:** 0.5 day.

#### A5. Plumb chord_symbols, key signature, dynamics through the engraver via direct MusicXML
**Anchor:** `03-codebase-arrange-engrave.md` §"What `render_midi_bytes` DROPS"; `01-codebase-pipeline.md` §13 rank 6.
**Diff hint:** Replace MIDI-only request to `oh-sheet-ml-pipeline` with a MusicXML-shaped contract. Add a `score_to_musicxml(score, expression)` helper that uses `music21` (currently uninstalled — add to `pyproject.toml`) to write key signature, time signature, chord symbols (from `metadata.chord_symbols`), tempo marking, voice numbers, and dynamics. POST the MusicXML to the ML engraver and let *it* polish layout, not derive structure from naked MIDI. Update `runner.py:516-520` to set `includes_*` flags from the actual content of the produced MusicXML rather than hard-coding `False`.
**Expected gain:** The single biggest engrave-stage lift in the entire roadmap. The ML engraver currently has to re-derive key, voice, and chord structure from notes alone; giving it a structured MusicXML shifts most of the heavy lifting back to the deterministic stages we already wrote.
**Risk:** External engraver may not understand the new XML; verify with the engraver maintainer. **Effort:** 3 days code + 1 day integration test.

#### A6. Remove the C-E-G-C `_stub_result` arpeggio fallback; fail loudly
**Anchor:** `02-codebase-transcribe.md` §6.
**Diff hint:** `backend/services/transcribe_result.py:249-278` — replace `_stub_result` with `raise TranscriptionFailure(reason)`. The orchestrator already has a job-level error path; surface the failure to the user instead of silently returning a fake C-major triad. Update `tests/conftest.py:46-68` to remove the autouse `skip_real_transcription` (see A11).
**Expected gain:** Makes silent failures visible. Today ~every "broken" job ships a C-major arpeggio that downstream stages happily render and the user has no idea anything went wrong.
**Risk:** Visible failure rate temporarily increases (it was always there, just hidden). **Effort:** 0.25 day.

#### A7. Pass a real `minimum_frequency` to Basic Pitch; preserve pitch bends
**Anchor:** `02-codebase-transcribe.md` §3, §12.1, §12.8.
**Diff hint:** `backend/services/transcribe_inference.py:123-129` — pass `minimum_frequency=27.5` (A0) on full-mix and `minimum_frequency=80` on the residual "other" stem; pass `multiple_pitch_bends=True` for vocal/melody passes. Stop dropping pitch bends in `transcribe_midi.py:50-54` — store them on `Note.pitch_bends_cents` (new field on `Note`) so engraver can render glissandi/portamento.
**Expected gain:** Suppresses sub-bass kick-drum-as-note artifacts on the single-mix path; preserves bent guitar/vocal that today quantizes to nearest semitone.
**Risk:** Adding a field to `Note` is a contract bump; gate behind schema-version negotiation. **Effort:** 1 day.

#### A8. Stop downsampling to mono 22 kHz for non-Basic-Pitch paths
**Anchor:** `01-codebase-pipeline.md` §13 rank 1; `02-codebase-transcribe.md` §4.
**Diff hint:** `backend/services/transcribe_pipeline_pop2piano.py:65` and `transcribe_pipeline_single.py:74` — only apply the 22.05 kHz mono downmix on the Basic-Pitch sub-path that requires it. Pop2Piano's spectrogram is built from native-rate stereo; passing it 22 kHz mono is throwing away half the channel info Pop2Piano is trained on. For the Demucs path keep stereo through separation, only mono-mix per-stem before Basic Pitch.
**Expected gain:** Pop is fundamentally stereo (panned vocals, doubled instruments). Restoring stereo to Pop2Piano and Demucs is a free quality win.
**Risk:** Memory / runtime grows ~2×; pin Cloud Run RAM. **Effort:** 1 day.

#### A9. Track-confidence drop floor: replace `< 0.35 → drop` with soft warning
**Anchor:** `01-codebase-pipeline.md` §13 rank 11.
**Diff hint:** `backend/services/arrange.py:53, 142-146` — replace `continue` with `track.confidence = max(track.confidence, 0.05); add_warning("low-confidence track included")`. Pop2Piano emits one track with arbitrary confidence; on noisy songs the entire piece is silently lost today.
**Expected gain:** Fixes "entire song vanishes" failure mode for noisy inputs.
**Risk:** Garbage stays in. Mitigate with explicit `quality.warnings` plumbed to the UI. **Effort:** 0.5 day.

#### A10. Pass `key signature` and `tempo_changes` through `midi_render`
**Anchor:** `03-codebase-arrange-engrave.md` §"What `render_midi_bytes` DROPS".
**Diff hint:** `backend/services/midi_render.py:36-137` — emit a `pretty_midi.KeySignature` from `metadata.key`, plus all `metadata.tempo_map` entries (not just first), plus `expression.tempo_changes` (currently always `[]` — see also A12). This is a no-cost write since the data is already on the Pydantic models.
**Expected gain:** Engraver no longer has to guess key signature from accidentals.
**Risk:** None. **Effort:** 0.5 day.

#### A11. Disable autouse `skip_real_transcription`; add a real-path CI smoke test
**Anchor:** `01-codebase-pipeline.md` §9; `02-codebase-transcribe.md` §9.
**Diff hint:** `tests/conftest.py:46-68` — remove `autouse=True`, expose as opt-in fixture. Add `tests/test_real_transcribe_smoke.py` that runs the actual Basic Pitch path on a 5-second synth-piano clip and asserts non-zero notes, F1 ≥ 0.4 against the synth-input MIDI. Run on PRs.
**Expected gain:** Configuration-level regressions in BP wiring become visible in CI.
**Risk:** CI runs ~1 min slower per PR; cache model weights in CI. **Effort:** 1 day.

#### A12. Fix README/code drift on `engrave.py`, music21, LilyPond
**Anchor:** `03-codebase-arrange-engrave.md` §"What the README claims" vs §"What actually exists".
**Diff hint:** Update `README.md` to remove `engrave.py`, music21, LilyPond mentions until they actually exist. Note the `oh-sheet-ml-pipeline` HTTP dependency. Update `docker-compose.yml` to reflect retired engrave worker. (The first three sentences of A5 above will eventually backfill the music21 claim.)
**Expected gain:** New contributors stop being misled.
**Risk:** None. **Effort:** 0.25 day.

#### A13. Include the `transcription_midi` (raw) in `EngravedOutput`
**Anchor:** `01-codebase-pipeline.md` §13 rank 12.
**Diff hint:** `backend/jobs/runner.py:501-512` — add `transcription_midi_uri` to `EngravedOutput`. The remote engraver currently sees only humanize-perturbed MIDI; surface the raw transcription as a separate artifact for debugging and as a "faithful" download option.
**Expected gain:** Easier user/dev debugging; gives `bypass_humanize` users the actual artifact they want.
**Risk:** Schema bump. **Effort:** 0.5 day.

**Quick-Wins total effort:** ~10 engineer-days (one sprint).

---

### Section B — Architectural Changes (1–3 sprints, pip-installable models)

Each entry is a model swap that doesn't require training. Order is "best ROI first" given Phase 1 findings.

#### B1. Demucs v4 / Mel-RoFormer source separation as a first-class stage
**Model:** `htdemucs` (default) or `mel_band_roformer_vocals` (better quality, slower).
**Repos:** [facebookresearch/demucs](https://github.com/facebookresearch/demucs) (MIT, archived 2025-01 but fork-friendly); [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) for Mel-RoFormer (MIT-via-UVR).
**Anchor:** `05-research-source-separation.md` Pattern D recommendation.
**Integration sketch:** New worker `backend/workers/separate.py`, new stage between `ingest` and `transcribe`. Output `instrumental.wav` (vocals + drums suppressed) plus the per-stem set. Add `audio_stems: dict[str, BlobURI]` to `TranscriptionResult` so downstream stages can reference individual stems. New `PipelineConfig.separator: Literal["htdemucs", "mel_roformer", "off"]`.
**Code stub** (HTDemucs):
```python
## backend/workers/separate.py
sources = apply_model(get_model("htdemucs"), wav, segment=8, overlap=0.10)
drums, bass, other, vocals = sources[0]
instrumental = bass + other  # drop vocals AND drums for Pattern D
torchaudio.save(out_wav, instrumental, model.samplerate)
```
**Expected gain:** Mel-RoFormer instrumental SDR ~17.66 dB; the Mel-RoFormer→AMT cascade in literature gives **+7.5 ppt** COnPOff for vocal melody transcription ([arXiv:2409.04702](https://arxiv.org/html/2409.04702v1)). Pattern D's vocal+drum suppression is the #1 documented intervention against false-positive onsets in pop AMT.
**Footprint:** HTDemucs ~80 MB weights, ~7 GB peak RAM, 1.5× audio CPU. Mel-RoFormer ~340 MB, 2-3× slower. On Cloud Run 2 vCPU: 30–60 s for HTDemucs, 90–180 s for Mel-RoFormer per 3-min song.
**Risk:** Demucs repo archived; pin a fork. SDR ↑ does NOT automatically mean transcription F1 ↑ ([Whisper-ALT 2025](https://arxiv.org/html/2506.15514v1) — separation can trigger AMT hallucinations). Run an A/B before switching default. License-clean (MIT). **Effort:** 1 sprint.

#### B2. ByteDance Kong's `piano_transcription_inference` for piano-stem transcription (with pedal!)
**Model:** Kong et al. 2021, "High-Resolution Piano Transcription with Pedals."
**Repos:** [bytedance/piano_transcription](https://github.com/bytedance/piano_transcription) (Apache-2.0); [pypi piano-transcription-inference](https://pypi.org/project/piano-transcription-inference/) (MIT). [arXiv:2010.01815](https://arxiv.org/abs/2010.01815).
**Anchor:** `07-research-piano-models.md` §3.1; `04-research-amt-sota.md` §2.3.
**Integration sketch:** New transcribe sub-path in `backend/services/transcribe.py:52-119`. Route to Kong when (a) Demucs/Mel-RoFormer says vocal energy < threshold (likely solo piano/cover) OR (b) user uploads with `--source-hint=piano`. Outputs onsets/offsets *and* sustain pedal events — extend `TranscriptionResult` contract to carry `pedal_events: list[PedalEvent]` (which today are only generated heuristically by humanize).
**Code stub:**
```python
from piano_transcription_inference import PianoTranscription, sample_rate
import librosa
audio, _ = librosa.load(stem_path, sr=sample_rate, mono=True)
pt = PianoTranscription(device='cpu')  # 'cuda' if available
pt.transcribe(audio, out_midi_path)  # MIDI includes CC64 pedal events
```
**Expected gain:** MAESTRO Note-onset F1 = **96.72%**, Note+Off F1 = **82.47%**, Pedal-onset F1 = **91.86%**. Kong is **the only mature, pip-installable, commercially licensed transcriber that emits pedal**. Pedal is the difference between "blizzard of staccato eighths" and a readable score.
**Footprint:** ~84 M params, ~330 MB checkpoint (Zenodo auto-download). CPU minutes-per-minute audio; faster-than-real-time on consumer GPU.
**Risk:** Kong is MAESTRO-overfit. Edwards et al. 2024 ([arXiv:2402.01424](https://arxiv.org/html/2402.01424v1)) showed −19.2 F1 on pitch-shift, −10.1 on reverb — meaning real YouTube rips degrade. **Mitigate by gating Kong behind Demucs (B1) so it sees a clean piano-ish stem.** Repo archived Dec 2025 but PyPI pkg actively maintained. **Effort:** 1 sprint.

#### B3. Beat This! transformer beat tracker (replace madmom)
**Model:** Foscarin et al., ISMIR 2024.
**Repo:** [CPJKU/beat_this](https://github.com/CPJKU/beat_this) (license MIT-ish, verify; check README).
**Anchor:** `10-research-emerging.md` §7.
**Integration sketch:** `backend/services/audio_timing.py:99-148` — replace madmom `RNNBeatProcessor + DBNBeatTrackingProcessor` with Beat This! transformer. madmom is unmaintained and tied to legacy NumPy. Beat This! is a single PyTorch model, no DBN postprocessing, accurate cross-genre.
**Expected gain:** Better beat/downbeat on pop. Fixes the "16th-note hi-hat smearing" Phase 1 documents.
**Footprint:** Single small transformer (~100 MB est).
**Risk:** Replacing madmom is a dependency-tree win since madmom pins old numpy/scipy. **Effort:** 2 days.

#### B4. Mel-RoFormer-based melody extraction (replace Viterbi melody)
**Model:** Wang et al. 2024 ([arXiv:2409.04702](https://arxiv.org/abs/2409.04702)).
**Repo:** Same `audio-separator` toolchain as B1.
**Anchor:** `10-research-emerging.md` §7; `05-research-source-separation.md` §4.
**Integration sketch:** Replace `backend/services/melody_extraction.py` Viterbi-over-264-bin-contour with the Mel-RoFormer vocal-melody head. Fed by the *vocal stem* from B1 (currently discarded in Pattern D — restore it as a separate transcribe path).
**Expected gain:** Phase 1's eval-baseline melody F1 = 0.093 — barely above noise. Mel-RoFormer-vocal SOTA on MIR-ST500 / POP909 should lift this 3-4×.
**Footprint:** Reuses B1's Mel-RoFormer; +1 inference pass.
**Risk:** Vocal-only — won't capture instrumental hooks (e.g., guitar lead lines). Keep current Viterbi as fallback.
**Effort:** 1 sprint (depends on B1).

#### B5. Score-HPT velocity refinement (post-AMT velocity polish)
**Model:** Score-HPT, Aug 2025 ([arXiv:2508.07757](https://arxiv.org/abs/2508.07757)).
**Anchor:** `10-research-emerging.md` §6.
**Integration sketch:** Insert a small (~1M param) BiLSTM/Transformer head between transcribe and arrange that re-estimates velocities given onset positions. Plumb back into `Note.velocity` before arrange's `_normalize_velocity` runs.
**Expected gain:** Fixes the linear `round(127*amplitude)` velocity mapping that ignores perceptual loudness. Dynamics are useless today regardless of source path; this is the cheapest fix.
**Footprint:** ~1M params extra.
**Risk:** Code likely needs reimplementation (paper-only at writing).
**Effort:** 1 sprint if reimplemented from paper.

#### B6. AMT-APC (MIT-licensed Pop2Piano-style) as parallel "Cover Mode"
**Model:** Komiya & Fukuhara 2024 ([arXiv:2409.14086](https://arxiv.org/abs/2409.14086)).
**Repo:** [misya11p/amt-apc](https://github.com/misya11p/amt-apc) — **MIT**.
**Anchor:** `06-research-pop-piano.md` §2.4, §4 recommendation; `10-research-emerging.md` §3.
**Integration sketch:** Add a new pipeline variant `pop_cover` (alongside `audio_upload`). Routes raw audio (or Demucs-cleaned) → AMT-APC → **skip arrange entirely** → engrave. Surface as a UI toggle: "Faithful transcription" vs "Piano cover."
**Expected gain:** AMT-APC fine-tunes hFT-Transformer on YouTube piano covers; "reproduces original tracks more accurately than any existing models" per authors. Etude paper subjectively rates the family above straight-transcription pipelines for pop. Direct path to the "this sounds like a real piano cover" experience.
**Footprint:** hFT-Transformer base ~5.5 M params; checkpoint <100 MB. Single-GPU.
**Risk:** Trained mostly on J-pop YouTube covers — Western-pop generalization unaudited. Output may exceed playability constraints; add a hand-span clamp post-process. License is **clean MIT** (vs Pop2Piano which has *no LICENSE file*). **Effort:** 1 sprint (single-line inference script).

#### B7. Replace external HTTP engrave with local Verovio + abjad/music21 + LilyPond
**Anchor:** `03-codebase-arrange-engrave.md` §"What actually exists"; `01-codebase-pipeline.md` §13 rank 6.
**Repos:** [rism-digital/verovio](https://github.com/rism-digital/verovio) (LGPL — link only, don't statically embed); [cuthbertLab/music21](https://github.com/cuthbertLab/music21) (BSD); [Abjad/abjad](https://github.com/Abjad/abjad) (Apache); LilyPond (GPL, separate process).
**Integration sketch:** New `backend/services/engrave_local.py` that takes `(PianoScore, ExpressionMap)` directly (NOT MIDI) and writes MusicXML via `music21` (already proposed in A5). For PDF, shell out to LilyPond via `abjad` formatting, OR render MusicXML→SVG via Verovio via `pyverovio`. Make the external `oh-sheet-ml-pipeline` an optional alternative engraver, not the only path. Drop the inline-from-runner pattern in `backend/jobs/runner.py:431-536` in favor of a real `engrave` Celery worker.
**Expected gain:** No more black-box layout. Eliminates the "PDF is always None" problem. Restores the README-implied music21+LilyPond stack.
**Footprint:** music21 ~50 MB pip; LilyPond ~80 MB binary; Verovio ~20 MB. All commodity.
**Risk:** LilyPond is GPL — invoke as a subprocess on user input, do not statically link. Verovio is LGPL — link dynamically. Engrave latency may grow (LilyPond is single-threaded) — mitigate by running in background and showing OSMD-rendered MusicXML to the user immediately. **Effort:** 2 sprints.

#### B8. Cluster-and-Separate GNN for voice/staff assignment
**Model:** Karystinaios & Widmer 2024 ([arXiv:2407.21030](https://arxiv.org/html/2407.21030v1)).
**Anchor:** `10-research-emerging.md` §10.
**Integration sketch:** Replace the naive `pitch >= 60` middle-C hand split in `backend/services/arrange.py:135-156` with a GNN-based voice/staff assigner. Input: notes with onset, duration, pitch. Output: `voice` and `hand` per note.
**Expected gain:** Solves the "tenor-range vocal lines spanning both staves" failure mode Phase 1 documents. Big readability lift.
**Footprint:** Small GNN (<10 MB est).
**Risk:** Code/weights availability unconfirmed; budget reimplementation. **Effort:** 1 sprint.

#### B-tier combination recommendation:
- **Adopt all of: A1–A13 (quick wins) + B1 (Demucs) + B2 (Kong) + B3 (Beat This!) + B5 (Score-HPT) + B6 (AMT-APC behind a toggle) + B7 (local engraver).** This is the recommended target stack.
- **Defer: B4 (Mel-RoFormer melody — overlaps with B1's vocal stem; revisit only if melody F1 still bad after B6). B8 (GNN voice assignment — defer to Section C as a research bet).**
- **Skip: hFT-Transformer (despite SOTA) — no pedal head, MAESTRO-overfit, awkward integration, no pip wrapper. Save for headlines. Cite `07-research-piano-models.md` §3.3.**
- **Skip: YourMT3+ — GPL-3.0 license blocker per `04-research-amt-sota.md` §2.2.**
- **Skip: SheetSage — CC BY-NC-SA on weights blocks commercial use per `06-research-pop-piano.md` §3.1, `09-research-datasets.md`.**

---

### Section C — Research-Bet Experiments (1–2 quarters, finetuning / data collection)

#### C1. Self-collected Pop+Piano-Cover corpus + AMT-APC fine-tune
**Anchor:** `09-research-datasets.md` §2 "Why pop pairs are scarce" + §5 "Process".
**Hypothesis:** AMT-APC's J-pop-heavy training distribution underperforms on Western pop. Replicating Pop2Piano's automated YouTube scrape (now with the better aligned audfprint/Panako tooling and Aria-AMT for cover transcription) will close the genre gap.
**Plan:**
1. Scrape Billboard Hot-100 + Spotify Top-50 × `"<title>" piano cover` × top-3 by views.
2. Audio-fingerprint match cover ↔ original master using **Panako** ([JorenSix/Panako](https://github.com/JorenSix/Panako)) — robust to pitch/tempo shift.
3. Auto-transcribe cover via **Aria-AMT** ([EleutherAI/aria-amt](https://github.com/EleutherAI/aria-amt), Apache-2.0, robust to YouTube rips per `07-research-piano-models.md` §3.2).
4. DTW-align cover MIDI to original chroma+onset features.
5. Filter via Pop2Piano's published thresholds (chroma accuracy ≥ 0.05, length-mismatch ≤ 15%).
6. Manual spot-check 5%.
**Scale:** 5,000 pairs, ~180 hrs total. Cost: $30k all-in (1 eng-mo + transcriber budget) per `09-research-datasets.md` §5.
**Use:** Fine-tune AMT-APC on the resulting corpus.
**Legal posture:** Training-only fair-use (Authors Guild v. Google, Bartz v. Anthropic 2025); keep corpus internal; never redistribute audio. Release model weights + scrape pipeline (low risk) + a small ~30-song CC-licensed eval set.
**Risk:** DMCA / ToS risk if scraping is aggressive. Quality of pseudo-labels is uncertain. **Effort:** 1 quarter.

#### C2. Train a multi-track decomposer: MR-MT3 / YourMT3+ finetune
**Anchor:** `04-research-amt-sota.md` §2.4 (MR-MT3, Apache-2.0) and §2.2 (YourMT3+, GPL-3.0 — only if license resolves).
**Plan:** Pretrain on Slakh2100 (CC-BY-4.0, 145 hrs synthetic multitrack) + POP909 symbolic + the C1 corpus. Output is per-instrument MIDI tracks (drums, bass, vocals, keys) which then feed a learned arrangement-reduction head.
**Why MR-MT3 not MT3:** instrument-leakage ratio improved from φ=1.65 → 1.05; +18 ppt on Slakh Flat F1.
**Risk:** MR-MT3 is **CC BY-NC-SA 4.0** per `07-research-piano-models.md` §3 — research only. May force us to retrain MT3 from scratch on permissive data. **Effort:** 1 quarter for finetune; 2 quarters for from-scratch retrain.

#### C3. Reinforce arrangement quality with a learned playability+sight-readability critic
**Anchor:** `08-research-eval-metrics.md` Tier 3 §3.5 "Hand reachability"; `10-research-emerging.md` §10 "RubricNet difficulty descriptors".
**Plan:** Train a small classifier (~5M params) that scores any `PianoScore` for (a) hand-span violations, (b) voice-leading smoothness, (c) Henle difficulty grade. Use Nakamura & Sagayama 2018 statistical-piano-reduction features; train on CIPI difficulty dataset + RubricNet labels.
**Use:**
- As a CI gate that fails PRs regressing playability fraction.
- As a **post-arrange refinement loop** that minimally edits the score until it scores ≥ playability threshold (gradient-free local search over voice swaps and octave drops).
**Risk:** Training data may be too classical-piano; needs pop augmentation. **Effort:** 1 quarter.

#### C4. Multimodal LLM in-context arrangement (research stub)
**Anchor:** `10-research-emerging.md` §8 (ChatMusician, MIDI-LLM, Anticipatory Music Transformer), "Not Yet Ready" — multimodal LLMs.
**Plan:** Given a transcribed melody+chords+beat-grid, prompt Claude/Gemini to emit voicings as ABC notation (or directly via MIDI-LLM). Compare to AMT-APC and the rule-based arranger.
**Caveat:** Per `10-research-emerging.md` Music Flamingo / CMI-Bench: audio-to-symbolic from current Audio LLMs "drops significantly". So **only pass symbolic input** (already-transcribed score), not raw audio.
**Best near-term candidate:** Anticipatory Music Transformer ([github.com/jthickstun/anticipation](https://github.com/jthickstun/anticipation)) for *infilling* — given melody+chord skeleton, fill in piano accompaniment voicing. Production-ready.
**Risk:** Quality unproven for pop. **Effort:** 0.5 quarter as research probe.

#### C5. Self-consistency / multi-pass transcription (DiffRoll refinement)
**Anchor:** `10-research-emerging.md` §4.
**Plan:** Run Aria-AMT or Kong N times with different conditioning (audio noise, slight time-shifts) → use DiffRoll ([sony/DiffRoll](https://github.com/sony/DiffRoll)) as a learned ensemble-reconciliation module. Compute per-frame agreement; for low-confidence frames, sample more denoising steps.
**Expected gain:** DiffRoll's paper claims +19 ppt over its discriminative counterpart but that gap closed against newer models; main value is the **flexible compute/accuracy trade-off**.
**Risk:** Latency multiplier. **Effort:** 1 quarter.

#### C6. Active-learning loop: user edits → fine-tuning data
**Anchor:** `10-research-emerging.md` "Not Yet Ready — Active learning loops" cautions about cost.
**Plan:** Add a frontend MusicXML editor (already partially exists via OSMD); log every user edit as `(audio_input, model_output, user_correction)` triples. Quarterly retrain. Pseudo-labeling alternative ([groupmm/onsets_frames_semisup](https://github.com/groupmm/onsets_frames_semisup)) gets most of the benefit cheaper.
**Effort:** 1 quarter (frontend + data plumbing) + ongoing.

---

### Section D — Recommended Target Stack (~12 months out)

#### ASCII diagram

```
           USER UPLOAD: MP3 / MIDI / YouTube URL
                          │
                          ▼
                ┌────────────────────┐
                │  ingest (yt-dlp)   │  → 44.1 kHz STEREO WAV (no mono downmix here)
                └─────────┬──────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │   separate (NEW)     │  HTDemucs default → vocals/drums/bass/other
                │     Pattern D        │  Mel-RoFormer optional ("HQ" mode)
                └──┬──────┬───────┬───┘
                   │      │       │
       vocals  drums   instrumental (=bass+other)
          │       │       │
          ▼       ▼       ▼
                ┌─────────────────────────────────┐
                │       transcribe (NEW)          │
                │  ┌───────────────────────────┐  │
                │  │ pop_cover MODE:           │  │
                │  │   AMT-APC end-to-end on   │  │
                │  │   instrumental.wav        │  │
                │  ├───────────────────────────┤  │
                │  │ faithful MODE:            │  │
                │  │   Kong (piano stem AMT)   │  │
                │  │     + pedal events!       │  │
                │  │   + Mel-RoFormer melody   │  │
                │  │     on vocals stem        │  │
                │  │   + Beat This! bar grid   │  │
                │  │   + Score-HPT velocity    │  │
                │  └───────────────────────────┘  │
                │  + chord head (BTC/ChordFormer) │
                │  + key estimation (Krumhansl)   │
                └─────────────┬───────────────────┘
                              │
                              ▼  TranscriptionResult v4 (adds: pedal_events,
                              │      audio_stems URIs, pitch_bends, sections)
                              │
                ┌─────────────────────────────────┐
                │      arrange (REWORKED)         │
                │   - voice cap 4/hand            │
                │   - Cluster-and-Separate GNN    │
                │     for voice/staff assignment  │
                │     (replaces SPLIT_PITCH=60)   │
                │   - velocity preserves dynamics │
                │   - quantize w/ swing-aware grid│
                └─────────────┬───────────────────┘
                              │
                              ▼
                ┌─────────────────────────────────┐
                │ humanize (OPT-IN, default OFF   │
                │  for transcription products)    │
                └─────────────┬───────────────────┘
                              │
                              ▼
                ┌─────────────────────────────────┐
                │      engrave (LOCAL)            │
                │   PianoScore + Expression →     │
                │   music21 → MusicXML            │
                │   → Verovio (SVG)               │
                │   → LilyPond (PDF)              │
                │   includes_dynamics=true        │
                │   includes_pedal_marks=true     │
                │   includes_chord_symbols=true   │
                └─────────────┬───────────────────┘
                              │
                              ▼
              EngravedOutput: pdf_uri, musicxml_uri, midi_uri,
                              audio_preview_uri, evaluation_report
```

#### Stage-by-stage choices

| Stage | Today | Target | Rationale |
|---|---|---|---|
| ingest | yt-dlp + librosa probe | unchanged + stereo preservation | A8 |
| **separate** | not present | **HTDemucs (default), Mel-RoFormer (HQ)** | B1 — Pattern D from `05-research-source-separation.md` |
| transcribe (faithful) | Pop2Piano default → BP fallback | **Kong piano AMT on bass+other** + **Mel-RoFormer vocal melody** + **Beat This!** beat | B2/B3/B4 — pedal events, robust beats |
| transcribe (cover) | n/a | **AMT-APC** on instrumental.wav | B6 — direct pop→piano, MIT |
| velocity polish | linear `127*amp` | **Score-HPT** | B5 — perceptual loudness |
| arrange | naive middle-C split, 2-voice cap | 4-voice cap + GNN voice/staff + dynamics-preserving normalize | A2/A3 + B8 |
| humanize | rule-based, default ON | rule-based, default OFF for "faithful" | A4 |
| engrave | external HTTP black box | **Local music21 → Verovio (SVG) + LilyPond (PDF)** | B7 — A5 plumbs structured XML |

#### Contract changes vs current Pydantic schema

```python
## shared/shared/contracts.py (target v4.0.0)
class Note(BaseModel):
    pitch: int
    onset_sec: float
    offset_sec: float
    velocity: int
    pitch_bend_cents: list[tuple[float, float]] = []  # NEW (A7)

class PedalEvent(BaseModel):  # NEW
    cc: Literal[64, 66, 67]  # sustain / sostenuto / una corda
    onset_sec: float
    offset_sec: float
    confidence: float

class TranscriptionResult(BaseModel):
    midi_tracks: list[MidiTrack]
    analysis: HarmonicAnalysis
    quality: QualitySignal
    pedal_events: list[PedalEvent] = []  # NEW (B2 from Kong)
    audio_stems: dict[str, BlobURI] = {}  # NEW (B1)
    transcription_midi_uri: BlobURI | None = None  # NEW (A13)

class HarmonicAnalysis(BaseModel):
    key: str
    time_signature: str  # extend to 6/8, 12/8, 7/8 (drop hard 4/4 default)
    tempo_map: list[TempoChange]
    chords: list[RealtimeChordEvent]
    sections: list[Section]  # populate from MSAF/structure analysis (filed C-grade)
    mode: Literal["major", "minor", "dorian", "mixolydian", "phrygian", ...]  # NEW

class EngravedScoreData(BaseModel):
    includes_dynamics: bool  # actually computed, not hard-coded False
    includes_pedal_marks: bool
    includes_fingering: bool
    includes_chord_symbols: bool
    page_count: int  # NEW
    measure_count: int  # NEW
    validation_warnings: list[str] = []  # NEW

class EngravedOutput(BaseModel):
    pdf_uri: BlobURI  # NEW non-null when local engrave runs
    musicxml_uri: BlobURI
    humanized_midi_uri: BlobURI
    transcription_midi_uri: BlobURI | None  # NEW (A13)
    audio_preview_uri: BlobURI | None
    evaluation_report: EvalReport | None  # NEW (auto-runs the metric ladder)
```

#### What can be deleted / simplified

- **`backend/services/condense.py`** — entirely deprecated by A1.
- **`backend/services/transform.py`** — passthrough stub, deletable.
- **`svc-decomposer/`** and **`svc-assembler/`** — dead-code stubs per `01-codebase-pipeline.md` §11. Either wire them up as the real Celery boundaries for transcribe and arrange, or delete.
- **Pop2Piano path** in transcribe — superseded by AMT-APC (B6) which is MIT-licensed where Pop2Piano is *no LICENSE file*. Keep Pop2Piano as a research-only A/B comparator.
- **The "humanize-then-engrave" coupling in `runner.py`** — humanize becomes optional and engrave reads `PianoScore + ExpressionMap` directly.

#### Readability/playability flow (end-to-end argument)

The reason this stack produces readable+playable sheets:
1. **Separation** removes the false-positive onsets that cause "blizzard of staccato eighths" in current Basic Pitch output.
2. **Kong with pedal events** + **Score-HPT velocity** restores damper physics and dynamic shaping — the score breathes instead of being uniformly loud.
3. **Cluster-and-Separate GNN** assigns notes to hands by reachability and voice continuity, not naive middle-C.
4. **Local music21 + LilyPond engraver** receives structured `PianoScore` (key, voice numbers, chord symbols) — not naked MIDI. The engraver renders proper accidentals, beam grouping, dynamics text.
5. **AMT-APC "Cover Mode"** as a parallel path serves users who want musicality > fidelity.

#### License-clean summary
- HTDemucs MIT; Mel-RoFormer code MIT (UVR weights MIT).
- Kong Apache-2.0 / MIT.
- AMT-APC MIT.
- music21 BSD; Verovio LGPL (link only); LilyPond GPL (subprocess only).
- Beat This! MIT-ish (verify).
- All Tier 1 picks **avoid** GPL-3.0 (YourMT3+) and CC-NC (SheetSage, NoteEM, MoisesDB, MR-MT3) — see Section E.

---

### Section E — Risks, Costs, Dependencies

#### E1. Compute / latency budget growth

| Stage | Current | Target | Latency add (3-min song, 2 vCPU CPU) |
|---|---|---|---|
| separate | 0 | HTDemucs | +30–60 s |
| separate | 0 | Mel-RoFormer | +90–180 s |
| transcribe (Kong) | BP ~10 s CPU | Kong + BP | +60–180 s CPU; <30 s GPU |
| transcribe (AMT-APC) | n/a | AMT-APC | seconds on GPU; minutes on CPU |
| velocity (Score-HPT) | n/a | +1M params | +5 s |
| engrave (LilyPond) | external HTTP | local LilyPond subprocess | +15-30 s (single-threaded) |

**Total wall-clock for a 3-min pop song:** today ~30–60 s; target 2–4 min on CPU, 30–60 s on GPU. Plan for: (a) GPU-spot-pool for the heavy paths, (b) progressive UX (show separation done → transcription done → engraving done as each completes).

#### E2. Disk footprint for model weights

| Model | Disk | Container layer? |
|---|---|---|
| HTDemucs `htdemucs` | 80 MB | Yes — bake in |
| Mel-RoFormer (HQ mode) | 340 MB | Yes |
| Kong piano transcription | 330 MB (Zenodo auto-DL) | **Pre-cache in Docker build** |
| AMT-APC (hFT-Transformer base) | <100 MB | Yes |
| Beat This! | ~100 MB est | Yes |
| Score-HPT | <10 MB | Yes |
| LilyPond binary | 80 MB | Yes |
| music21 + Verovio | 70 MB | Yes |

**Total target Docker image:** ~1.5 GB (vs current ~700 MB est). Acceptable on Cloud Run; worth optimizing image layering.

#### E3. License audit

| Component | License | Commercial OK | Risk |
|---|---|---|---|
| HTDemucs | MIT | Yes | Repo archived 2025-01; pin a fork |
| Mel-RoFormer | code MIT, UVR weights mostly MIT | Yes | Per-checkpoint check (`viperx`, `jarredou` weights from UVR community) |
| Kong | Apache-2.0 (training) / MIT (inference pkg) | Yes | None |
| AMT-APC | MIT | Yes | None |
| Beat This! | MIT-ish (verify README) | Likely yes | Verify before integration |
| music21 | BSD | Yes | None |
| Verovio | LGPL | Yes (link dynamic) | Don't statically embed |
| LilyPond | GPL-2 | Yes (subprocess) | Don't link in-process |
| **YourMT3+** | **GPL-3.0** | **NO unless microservice-isolated** | Skip |
| **SheetSage weights** | **CC BY-NC-SA 3.0** | **NO** | Skip |
| **MR-MT3** | **CC BY-NC-SA 4.0** | **NO** | Skip |
| **NoteEM** | **CC BY-NC-SA 4.0** | **NO** | Skip |
| **Banquet** (separator) | CC BY-NC-SA 4.0 | NO | Skip |
| **Pop2Piano weights** | **No LICENSE file** | **Murky** | Treat as research-only |

#### E4. Data-licensing risk

**Self-collected pop/piano-cover corpus (C1):** training-only fair-use (US) per `09-research-datasets.md` §5.
- Keep corpus internal; never redistribute audio.
- Open-source the *scrape pipeline*; release model weights + a small CC-licensed eval set.
- Avoid DMCA circumvention (no rate-limit busting, no paywall bypass).
- Honor robots.txt and ToS.
- Risk: non-zero. Comparable to Pop2Piano's posture; the field has done this.

#### E5. Migration risk — what existing tests break, what new tests are needed

| Test | Status under target stack |
|---|---|
| `tests/conftest.py:46-68 skip_real_transcription autouse` | **Disable** (A11). Replaces with opt-in fixture. |
| `tests/conftest.py:81-95 stub_ml_engraver` | Refactor — replace with stub for *local* engraver (music21+LilyPond) |
| `tests/test_arrange.py` snapshot tests | Re-baseline after A1 (default arrange) |
| `tests/test_arrange_simplify.py` | Update min_velocity threshold (A3) |
| `tests/test_transcribe_pop2piano.py` | Keep but mark as legacy path |
| `tests/test_ml_engraver_*.py` | Re-target at local engraver (B7) |
| **NEW** `tests/test_real_transcribe_smoke.py` | A11 — synth-piano F1 ≥ 0.4 |
| **NEW** `tests/test_separation.py` | B1 — verify HTDemucs returns 4 stems |
| **NEW** `tests/test_kong_transcribe.py` | B2 — assert pedal events in result |
| **NEW** `tests/test_amt_apc_cover.py` | B6 — basic shape correctness |
| **NEW** `tests/test_engrave_local.py` | B7 — MusicXML validates against schema; PDF non-empty |
| **NEW** `eval/scripts/run_eval_ladder.py` | per `08-research-eval-metrics.md` §"Recommended ladder" — Tier 2 + Tier 3 + Tier 4 in CI |

#### E6. Data shape risks
- Schema bump from v3 to v4 affects all consumers; add a migration shim that downgrades v4 → v3 for legacy clients.
- Frontend OSMD/VexFlow rendering needs to handle the new fields (chord symbols, dynamics text) or it'll display nothing (current state).

#### E7. Operational
- Cloud Run cold starts: bake weights into image, warm-start a long-running container per worker type.
- Memory: pin segment sizes for HTDemucs/Mel-RoFormer; budget 4 GB peak per worker.
- Failure modes: separation can fail on extremely short clips (<1 s) or pure-vocal a-capella; gate the new stage behind a duration check.

---

### Section F — Sequencing Recommendation

Each milestone has: anchor IDs from earlier sections, definition of done, measurable acceptance criterion. Numbers like "real-pop F1" assume the new internal eval set described in `08-research-eval-metrics.md` and `09-research-datasets.md` §3 (build a 30-song hand-curated pop eval set ahead of milestone M1 — small one-off engineering work).

#### Sprint 0 — Foundations (1 week)
**Pre-req for everything below.** Build the **30-song internal pop eval set** (per `09-research-datasets.md` §3) — buy human-verified MIDI references at $30/song = $900. This is the precondition for all "real-pop F1" acceptance criteria below.

#### M1 — Quick wins ship (1 sprint, 2 weeks)
**Tasks:** A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13.
**DoD:**
- `score_pipeline=arrange` is default.
- Voice cap 4/hand, velocity threshold 25.
- `humanize` opt-out toggle in UI.
- C-major arpeggio fallback gone.
- BP `minimum_frequency` and pitch-bend on.
- A real-path CI smoke test runs green.
- README/code drift fixed.
- `KeySignature` flows to MIDI.
- Local-engraver MusicXML stub with chord symbols (A5 partial).
**Acceptance:**
- Real-path Basic Pitch CI test passes with F1 ≥ 0.4 on synth-piano.
- 30-song eval set: melody F1 ≥ 0.20 (was 0.093). LH note-count ≥ 1.5× current (was nearly empty due to A3).
- Subjective spot-check on 5 pop songs: dynamics audibly varied (was uniform [35,120]).

#### M2 — Source separation lands (1 sprint, 2 weeks)
**Tasks:** B1.
**DoD:**
- `backend/workers/separate.py` deployed; HTDemucs default.
- New `audio_stems: dict` in `TranscriptionResult`.
- Optional Mel-RoFormer toggle.
- Cache by `sha256(audio_bytes) + model_id` per `05-research-source-separation.md` §5.4.
**Acceptance:**
- A/B on 30-song eval set: real-pop melody F1 lift ≥ +0.05 vs M1.
- Separation latency ≤ 60 s on 2 vCPU for HTDemucs on 3-min songs.

#### M3 — Kong piano AMT + Beat This! (1 sprint, 2 weeks)
**Tasks:** B2, B3.
**DoD:**
- Kong wired in `backend/services/transcribe.py:52-119` as new sub-path.
- `pedal_events` flows from Kong through arrange and engrave.
- `EngravedScoreData.includes_pedal_marks=True` is no longer hard-coded False.
- madmom replaced by Beat This!.
**Acceptance:**
- 30-song eval set Note F1 ≥ 0.45 (vs 0.368 baseline). MAESTRO-clean Note F1 ≥ 0.95 (regression guard).
- Pedal events visible in 80%+ of pop songs containing pedal.
- Beat F1 ≥ 0.75 on the eval set (madmom baseline ~0.65).

#### M4 — Local engraver replaces external HTTP (2 sprints, 4 weeks)
**Tasks:** A5 finished, B7.
**DoD:**
- music21-based MusicXML writer with key, voice, chord symbols, dynamics.
- Verovio for SVG render; LilyPond for PDF.
- `pdf_uri` is no longer always None.
- External `oh-sheet-ml-pipeline` becomes optional alternative engraver.
**Acceptance:**
- Real `pdf_uri` returned on 95%+ of jobs.
- 5 pianists rate engraved output ≥ 3.5/5 on "looks like real sheet music" axis (Tier 5 §5.4 from `08-research-eval-metrics.md`).
- Round-trip metric (audio→engrave→re-synth→re-transcribe) F1 ≥ 0.85 (Tier 4 §4.7).

#### M5 — Score-HPT velocity refinement (1 sprint, 2 weeks)
**Tasks:** B5.
**DoD:** Score-HPT BiLSTM/Transformer post-processor in pipeline.
**Acceptance:** MAESTRO Note+Off+Vel F1 ≥ 0.85 (current Kong-baseline 80.92%; Score-HPT paper claims ≥+5 ppt).

#### M6 — AMT-APC cover mode (1 sprint, 2 weeks)
**Tasks:** B6.
**DoD:**
- New `pop_cover` pipeline variant; UI toggle.
- Output bypasses arrange entirely; engrave receives AMT-APC MIDI directly.
**Acceptance:** A/B preference vote (≥30 listeners) on 30-song eval: AMT-APC cover wins ≥ 55% on "Musicality" against the faithful path. Playability fraction ≥ 0.85 (hand-span ≤ 14 semitones, ≤ 5 notes/hand).

#### M7 — Eval ladder in CI (1 sprint, 2 weeks)
**Tasks:** all of `08-research-eval-metrics.md` §"recommended ladder".
**DoD:**
- CI computes Tier 2 (key, tempo, chord), Tier 3 (playability), Tier 4 (CLAP-music + chroma cosine) on PRs.
- Hard-fail PRs that regress playability fraction by ≥ 0.05.
- Weekly: full Tier 1 + MV2H on 30-song held-out set.
**Acceptance:** Eval pipeline runs end-to-end in <15 min on CI; produces a JSON `evaluation_report` artifact.

#### M8 — Cluster-and-Separate GNN voice/staff (1 sprint, 2 weeks)
**Tasks:** B8.
**DoD:** GNN replaces SPLIT_PITCH=60 in arrange.
**Acceptance:** Voice/staff F1 ≥ 0.80 on a labeled 20-song hand-split eval (manual labels) — beats 0.65 of middle-C heuristic.

#### Q3 — Quarterly research bets (parallel to ongoing M-tasks)
- **C1:** scrape + align 5,000 pop+piano-cover pairs. **Acceptance:** ≥ 4,000 pairs after filtering; manual spot-check label noise rate ≤ 10%.
- **C3:** train playability+sight-readability critic. **Acceptance:** classifier predicts Henle level within ±1 grade on 80% of CIPI test set.
- **C4 probe:** Anticipatory Music Transformer A/B for inner-voice infill. **Acceptance:** subjective preference ≥ 50% vs rule-based arrange.

#### Q4
- **C1 fine-tune:** AMT-APC fine-tune on the C1 corpus. **Acceptance:** real-pop A/B preference ≥ 60% vs zero-shot AMT-APC.
- **C2 multi-track decomposer:** MR-MT3 or YourMT3+-equivalent retrain on Slakh + POP909 + C1 corpus. **Acceptance:** Slakh Multi F1 ≥ 0.65 (matches MR-MT3 baseline).

---

### Anchored citations summary

- Codebase weaknesses: `01-codebase-pipeline.md` §13, `02-codebase-transcribe.md` §12, `03-codebase-arrange-engrave.md` "Ranked failure modes".
- AMT model picks: `04-research-amt-sota.md` §2-3, `07-research-piano-models.md` §3, §6.
- Source separation: `05-research-source-separation.md` Pattern D (§4) + integration sketch (§5).
- Pop-piano models: `06-research-pop-piano.md` §1, §2, §4 (E hybrid recommendation).
- Eval ladder: `08-research-eval-metrics.md` "Recommended ladder" §"summary".
- Datasets / scrape pipeline: `09-research-datasets.md` §2, §5.
- Emerging techniques (Beat This!, Score-HPT, AMT-APC, MERT, GNN voice): `10-research-emerging.md` §3, §6, §7, §10.


---

# Part III — Evaluation Strategy

_Synthesized by Phase 2 agent `synth-eval-strategy`. The 5-tier metric ladder, reference data design, test-harness architecture, CI/release gates, continuous benchmarking, human-in-the-loop protocols, reference-free evaluation deep-dive, and a concrete first-90-days execution plan._

## Oh Sheet — End-to-End Evaluation Strategy

**Author:** Phase-2 synthesis agent
**Date:** 2026-04-25
**Audience:** Oh Sheet engineering + product leadership
**Companion docs:** `/tmp/oh-sheet-research/01-…10-*.md`

---

### 1. Evaluation philosophy

#### 1.1 Why a single F1 number is insufficient

Oh Sheet today reports a single headline metric — `mean_f1_no_offset = 0.368`
on a 25-MIDI subset (`eval-baseline.json:362-368`). This number is
*operationally meaningless* for the actual product:

1. **The F1 number measures Basic Pitch on FluidSynth-rendered piano MIDI**
   (`scripts/eval_transcription.py:22-24`), not on real pop audio. Basic
   Pitch's easiest possible input. It tells us nothing about Pop2Piano (the
   actual default path — `backend/services/transcribe.py:52-119`,
   `config.py:244`), about source separation, or about anything past
   transcribe.
2. **Note-onset F1 ignores most of what makes a piano score useful.** mir_eval
   `precision_recall_f1_overlap` (used at `scripts/eval_transcription.py:502-507`)
   counts a 50 ms onset, ±50 cents pitch hit as success. It is blind to
   voicing, hand assignment, key signature, chord symbols, dynamics,
   articulation, beat alignment, sight-readability, and engraving cleanliness
   — every dimension that determines whether a user can play the result.
   The Ycart 2020 paper (TISMIR, "Investigating the Perceptual Validity of
   Evaluation Metrics for Automatic Piano Music Transcription",
   https://transactions.ismir.net/articles/10.5334/tismir.57) and Simonetta
   2022 (Multimedia Tools and Applications,
   https://link.springer.com/article/10.1007/s11042-022-12476-0) both
   demonstrate empirically that F1 has *only modest* correlation with human
   judgement.
3. **Two correct transcriptions of the same audio can have low cross-F1.**
   Pop covers shift bars, transpose, drop verses, and rephrase melodies —
   the literature consensus (`08-research-eval-metrics.md` §1.2-1.3) is that
   onset+offset F1 is meaningless across covers, and onset-only F1 only
   marginally less so.
4. **F1 cannot localize regressions.** A drop from 0.40 to 0.35 could be a
   transcribe regression, a velocity-flattening change in `arrange_simplify`
   (`backend/services/arrange_simplify.py:99`), an arrange voice-cap regression
   (`backend/services/arrange.py:51-52`), or a downstream condense bug — F1
   alone tells you only that *something* is worse.
5. **Oh Sheet's stated baseline (F1=0.368) crashes against research SOTA.**
   hFT-Transformer reports 97.44% MAESTRO note F1 (Toyama 2023,
   arXiv:2307.04305); even Basic Pitch on its own benchmark hovers near 88%.
   The fact that we report 0.368 against fluidsynth piano renders means
   something is dramatically wrong with how we're using mir_eval or with the
   pipeline itself — and a single number can't tell us which.

#### 1.2 The reference-availability problem

The single most important constraint on Oh Sheet's eval design: **for an
arbitrary pop song the user uploads, there is no canonical "correct" piano
cover**. Three pianists transcribing the same Taylor Swift track will produce
three different MIDIs that all "sound right" — yet may pairwise score F1 < 0.3
under mir_eval. Even when piano covers exist on MuseScore.com or sheetmusicplus,
they are protected derivative works (`09-research-datasets.md` §2) — we cannot
redistribute them, and we cannot assume the user uploaded a song we have a
cover for.

This fact has three consequences that shape every recommendation below:

- **Reference-required metrics** (mir_eval.transcription, MV2H, musicdiff) are
  only usable on a *small curated eval set* whose references we built or
  bought. They cannot run in production, and they cannot run on user
  uploads.
- **Reference-free metrics** (CLAP-music cosine, chroma cosine, playability
  fraction, voice-leading smoothness, structural agreement vs. audio-side
  estimators) must do most of the per-song quality monitoring work in CI and
  in production telemetry.
- **Human evaluation** is the ground truth that automated metrics are
  validated against. Without periodic human-in-the-loop study, all automated
  metrics drift away from what users actually care about.

#### 1.3 The 5-tier metric ladder (recommended)

| Tier | Name | What it measures | Reference required? | When |
|---|---|---|---|---|
| **1** | **Note-level transcription accuracy** | Onset / offset / velocity correctness | Yes (paired MIDI) | Curated 30-50 song eval set; nightly |
| **2** | **Structural fidelity** | Key, tempo, beat/downbeat, time-sig, chord progression — the *bones* of the piece | No (audio-side estimator as anchor) | Every PR; cheap; ~30 s/song |
| **3** | **Arrangement / engraving quality** | Voice leading, hand reachability, polyphony density, sight-readability, MV2H-style score-similarity | Mixed (RF for playability, ref-required for MV2H/musicdiff) | Every PR (RF subset); nightly (full MV2H) |
| **4** | **Perceptual / re-synthesis** | CLAP-music cosine, MERT cosine, chroma cosine vs. original audio, FAD over corpus | Reference-free per-song; FAD set-level | Every PR (per-song); nightly (FAD) |
| **5** | **Human evaluation** | Multi-axis MOS (Faithfulness/Playability/Musicality), pianist rubric, sight-readability tests, A/B win rate vs. previous release | Yes (human raters) | Release gate (every release); quarterly calibration |

The ladder matches Oh Sheet's pipeline shape: Tier 1 isolates *transcribe*
quality; Tier 2 isolates *arrange* / *refine* harmonic structure; Tier 3
isolates *arrange* (voicing, hand split) and *engrave* (MIDI→MusicXML); Tier
4 closes the loop on perceptual fidelity end-to-end without paired data;
Tier 5 anchors automated metrics in human judgement.

---

### 2. Tier-by-tier metric ladder

In each tier I list the recommended metrics with: name, what it measures,
library/code, input/output format, *when* to compute, expected ranges,
gotchas, citation URL. **Reference-required (RR)** vs **reference-free (RF)**
is called out per metric. Bold headings indicate the metrics I recommend
running by default.

#### 2.1 Tier 1 — note-level transcription accuracy

These compare a transcription's MIDI against a paired ground-truth MIDI. They
*do not work* on user uploads — only on a curated eval set with paired
references. They are the gold standard for AMT and what everyone in the
literature reports.

| Metric | Library | Input → Output | RR/RF | Citation |
|---|---|---|---|---|
| **Note-onset F1** (50 ms onset, ±50 cents pitch) | `mir_eval.transcription.precision_recall_f1_overlap(..., offset_ratio=None)` | (ref intervals/pitches), (est intervals/pitches) → P/R/F1 | RR | https://mir-eval.readthedocs.io/latest/api/transcription.html |
| **Note onset+offset F1** (offset_ratio=0.2) | Same call with `offset_ratio=0.2` | Same | RR | Bay 2009 / Raffel 2014, https://colinraffel.com/posters/ismir2014mir_eval.pdf |
| **Note onset+offset+velocity F1** | `mir_eval.transcription_velocity` | Same + velocities (0-127) | RR | https://github.com/mir-evaluation/mir_eval/blob/main/mir_eval/transcription_velocity.py |
| Frame-level multi-pitch F1 | `mir_eval.multipitch` | Pianorolls, 10 ms hop | RR | https://mir-eval.readthedocs.io/latest/api/multipitch.html |
| Onset-only F1 (pitch-agnostic) | `mir_eval.onset` | Onset arrays | RR | https://mir-eval.readthedocs.io/latest/api/onset.html |
| Per-dimension errors (onset MAE, duration MAE, offset MAE on matched notes) | `mir_eval.transcription.match_notes` then numpy | Same | RR | Already implemented at `scripts/eval_transcription.py:518-585` |

**Code already in repo:** `scripts/eval_transcription.py:502-585` computes
all of the above except velocity F1. Per-role breakdown for melody / bass /
chords / piano is at `scripts/eval_transcription.py:722-789`. These
functions are the foundation; we extend them rather than rewrite.

**Expected ranges**:
- MAESTRO solo piano: SOTA 96–98% onset F1 (hFT-Transformer 97.44%).
- MAPS (cross-dataset): ~85%.
- Pop full mix → piano (Oh Sheet's actual job): **far lower**. Real-world
  pop transcription onset F1 on a paired-cover reference is in the
  20–40% range based on the 0.368 baseline; with-offset will be ~10-15%.

**Gotchas**:
- Velocity check at 10% tolerance is meaningless across covers (different
  pianist, different dynamic shaping).
- Pedal makes offset detection ill-defined (sustain releases drift); never
  tighten offset tolerance below 80 ms for piano.
- mir_eval's `precision_recall_f1_overlap` raises on empty inputs — guard
  with the existing zero-row pattern (`scripts/eval_transcription.py:483-500`).
- **Critical:** Tier 1 is only meaningful when reference MIDI is the *same
  arrangement*, ideally with bar-aligned onsets. Across-cover Tier 1 numbers
  are noise.

**When to compute**: Nightly on the curated eval set. Subset on every PR
(see §5).

#### 2.2 Tier 2 — structural fidelity

These measure whether the transcription gets the **musical bones** right:
key, tempo, beat grid, time signature, chord progression. They are *robust
to voicing differences* — a perfectly transcribed pop song and a 4-note
piano reduction of the same song should score similarly. Most are
**reference-free** when measured against the audio-side analysis of the
original input — which is exactly what we want for production telemetry.

| Metric | Library | Input → Output | RR/RF | Citation |
|---|---|---|---|---|
| **Key detection (MIREX weighted score)** | `mir_eval.key.weighted_score` | Estimated key, ground-truth key (string) → score in [0,1] | RF (audio-side anchor) | https://mir-eval.readthedocs.io/latest/api/key.html |
| **Tempo accuracy** (`±4%`, `±8%` octave-relaxed) | `mir_eval.tempo.detection` | Estimated BPM, reference BPM → P-score | RF | https://mir-eval.readthedocs.io/latest/api/tempo.html |
| **Beat F1** (±70 ms) + **CMLt** | `mir_eval.beat.f_measure`, `.cmlc/.cmlt` | Beat times → F1, CMLt | RF (madmom anchor) | https://mir-eval.readthedocs.io/latest/api/beat.html ; Davies 2014, https://archives.ismir.net/ismir2014/paper/000238.pdf |
| Downbeat F1 | `madmom.evaluation.beats.BeatEvaluation` | Downbeat times → F1 | RF | https://madmom.readthedocs.io/ |
| Time-signature accuracy | Custom — exact-match against madmom-derived meter | (3/4, 4/4, 6/8…) → confusion matrix entry | Partial RF | https://pmc.ncbi.nlm.nih.gov/articles/PMC8512143/ |
| **Chord accuracy** (`mirex` mode + `majmin`) | `mir_eval.chord.weighted_accuracy` | Time-aligned label arrays → segment-weighted accuracy | RF (chord-recognize both sides) | https://mir-eval.readthedocs.io/latest/api/chord.html |
| Chord segmentation | `mir_eval.chord.evaluate` returns `seg`, `overseg`, `underseg` | Same | RF | Same |
| Section-boundary F-measure (HR3F at ±3 s, HR.5F at ±0.5 s) | `mir_eval.segment.detection` | Boundary times → F1 | RF (MSAF anchor) | Nieto & Bello 2015, https://ccrma.stanford.edu/~urinieto/MARL/publications/NietoBello-ISMIR2015.pdf |

**Code hook**: New file `eval/tier2_structural.py`. Anchor side = run
`madmom` + `mir_eval.key` + chord-recognize on the *original audio*; estimate
side = read from `TranscriptionResult.analysis` (`shared/shared/contracts.py:185-190`).
This already exists in the contract — `HarmonicAnalysis(key, time_signature,
tempo_map, chords, sections=[])` — so Tier 2 just needs the anchor-side
runner plus a comparator.

**Expected ranges** (pop):
- Key (MIREX weighted): 0.70–0.85 plausible target (research convention says
  pop without modulation tops out around 0.85).
- Tempo P-score: 0.80–0.95; octave errors (half/double) are very common and
  should be reported separately.
- Beat F1 ±70 ms: 0.75–0.90 with madmom anchor.
- Chord (`mirex`): 0.70–0.85 on pop. Inversion-sensitive `tetrads` will sit
  10–15 ppt lower; pick `mirex` and stick to it for headline.
- Section boundaries: human inter-annotator agreement is the ceiling — F1
  ≈ 0.67 at strict, 0.76 at relaxed (SALAMI numbers).

**Gotchas**:
- Tempo octave errors: half-tempo (60 reported as 120) and double-tempo
  (120 reported as 60) are the two failure modes; report `tempo_p_score`
  *and* `tempo_p_score_octave_relaxed` so you can distinguish "wrong
  tempo" from "octave-confused tempo".
- madmom downbeat tracker requires a meter assumption; pop is mostly 4/4
  but 6/8 shuffles and 3/4 ballads will silently fail. Confusion matrix
  beats accuracy.
- Chord recognition is currently major/minor-only in Oh Sheet
  (`backend/services/key_estimation.py:38-44`); this caps chord accuracy
  on modal pop ("Riders on the Storm" is Dorian, scored as wrong). Add a
  modal-aware comparator that scores root+quality only.
- **Sections are always `[]` in the current Oh Sheet output**
  (`backend/services/transcribe_result.py:188`); section eval will report 0
  unless RefineService runs (Anthropic key) or transcribe is upgraded to
  emit sections.

**When**: Every PR, on a 5-song quick-look subset. Full Tier 2 nightly on
30-song corpus.

#### 2.3 Tier 3 — arrangement quality (voicing, playability, engraving)

These ask whether the **sheet music itself** is a competent piano arrangement.
Most are RF — they evaluate the score in isolation.

| Metric | Library | Input → Output | RR/RF | Citation |
|---|---|---|---|---|
| **Hand-reachability / playability fraction** | Custom over `music21.chord` | Per chord: max-pitch − min-pitch ≤ 14 st AND ≤ 5 simultaneous notes; aggregate as fraction | RF | Nakamura & Sagayama 2018, https://arxiv.org/pdf/1808.05006 ; music21 https://music21.org |
| **Voice-leading smoothness** | Custom over Tonnetz / music21 | Mean semitone displacement per voice across consecutive chords | RF | Lerdahl, Tonal Pitch Space; https://en.wikipedia.org/wiki/Tonnetz |
| Polyphony density (notes per beat per hand) | `pretty_midi` + numpy | Beat-bucketed counts → mean, p95, max | RF | Custom (descriptor in RubricNet, https://arxiv.org/html/2509.16913) |
| **Sight-readability score** (RubricNet-style) | RubricNet descriptors implemented over music21 | Score → predicted Henle/ABRSM grade + readability score | RF | Ramoneda 2023, https://arxiv.org/abs/2306.08480 ; https://arxiv.org/html/2509.16913 |
| Texture / density (Couvreur & Lartillot) | Custom over music21 / partitura | Score → 1/2/3-layer density profile | RF | https://hal.science/hal-03631151/file/main.pdf |
| Voice-separation evaluation | MV2H Voice component (Java; Python port) | Score, reference → F1 over voice edges | RR | McLeod 2018, https://github.com/apmcleod/MV2H |
| **MV2H** (Multi-pitch / Voice / Meter / Value / Harmony joint score) | `apmcleod/MV2H` (Java canonical, slow Python port) | Score, reference → 5 sub-scores + composite | RR | McLeod & Steedman 2018, https://ismir2018.ismir.net/doc/pdfs/148_Paper.pdf |
| musicdiff (score-tree edit distance over MusicXML) | `musicdiff` PyPI; depends on music21 ≥9.7 | Score, reference → edit distance + visualization | RR | Foscarin 2019, https://inria.hal.science/hal-02267454v2/document ; https://pypi.org/project/musicdiff/ |
| Cogliati & Duan score-similarity | https://github.com/AndreaCogliati/MetricForScoreSimilarity | Same | RR | Cogliati & Duan 2017, https://archives.ismir.net/ismir2017/paper/000131.pdf |
| OMR-NED (rendered-page edit distance) | https://arxiv.org/abs/2506.10488 | Rendered PNG, reference PNG | RR | ISMIR 2025 |
| Engraving heuristic checks (ledger lines >3, beam violations, voice crossings, enharmonic spelling) | Custom rule check over MusicXML + lxml | Score → list of warnings + scalar | RF | Custom |

**Code hook**: New file `eval/tier3_arrangement.py`. Inputs are the
`PianoScore` (`shared/shared/contracts.py:268-272`) emitted by arrange/condense,
or the MusicXML coming back from the engrave HTTP service. The
`HumanizedPerformance` already has staff/voice info per ExpressiveNote
(`backend/services/humanize.py:196-271`); we can compute Tier 3 metrics
*before* engrave to isolate arrange-stage regressions.

**Critical for Oh Sheet**: the current default `score_pipeline=condense_only`
(`config.py:562`) silently bypasses voice routing. Tier 3 metrics will
expose this directly: condense's middle-C split (`condense.py:86-103`) will
produce poor voice-leading and exceed hand reachability much more often than
arrange's role-aware split. **Tier 3 is the metric that will tell us
condense_only is hurting product quality.**

**Expected ranges**:
- MV2H on classical AMT: SOTA ~0.65–0.80 composite (perfect = 1.0).
- Playability fraction: target ≥0.80 for "intermediate" outputs; current
  Oh Sheet output likely sits at 0.50–0.65 due to 2-voice cap dropping
  legitimate polyphony (`arrange.py:51-52`) and middle-C split forcing
  unreachable spans on bass lines.
- Voice-leading smoothness: median displacement ≤2.5 semitones per voice
  is "smooth" pop voicing.

**Gotchas**:
- MV2H is Java-canonical; the Python port is slower but acceptable for
  nightly. Don't run MV2H per-PR.
- Hand-reachability thresholds depend on tempo: a 14-semitone span at
  q=80 is broken-chord reachable; at q=180 it's not.
- music21 imports add ~3s of startup; cache-load it in long-running CI
  jobs.
- **There is currently no music21 in the backend**
  (`03-codebase-arrange-engrave.md` §"Engraving library audit"); we'll
  need it for Tier 3 score parsing. It's already in the project deps
  (`pyproject.toml:18`, `music21>=9.1`) but unused.

**When**: Per-PR — playability fraction + voice-leading smoothness +
polyphony density + engraving heuristic checks (cheap, RF). Nightly —
MV2H, musicdiff, sight-readability score.

#### 2.4 Tier 4 — perceptual / re-synthesis (reference-free end-to-end)

When you can't define a parallel reference, measure the transcription by
re-synthesizing it and comparing against the original input audio in a
perceptual embedding space. **Every Tier 4 metric is reference-free** in our
sense — they use the *input audio* as the reference, not a separate piano
cover. This is the metric class that gives us *production-monitoring*
quality measurement on user uploads.

| Metric | Library | Input → Output | RR/RF | Citation |
|---|---|---|---|---|
| **CLAP-music cosine similarity** | `LAION-CLAP` (https://github.com/LAION-AI/CLAP) ; `audiocraft.metrics.clap_consistency` | Original audio + resynthesized piano audio → cosine in [−1, 1] | RF (input-anchored) | https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/metrics/clap_consistency.html |
| **MERT embedding cosine** | `m-a-p/MERT-v1-330M` on HF | Same | RF | https://huggingface.co/m-a-p/MERT-v1-330M ; Li 2024 ICLR, https://arxiv.org/abs/2306.00107 |
| **Chroma cosine** (beat-aligned) | `librosa.feature.chroma_cqt` + bar-wise cosine | Same | RF | https://librosa.org/doc/main/generated/librosa.feature.chroma_cqt.html |
| Tonnetz / tonal-centroid distance | `librosa.feature.tonnetz` | Same → L2 distance | RF | https://librosa.org/doc/main/generated/librosa.feature.tonnetz.html |
| Grooving / rhythm cosine | Custom over librosa onsets | Same → bar-wise cosine | RF | Wang 2020 (POP909), https://arxiv.org/pdf/2008.07142v1 |
| **Round-trip self-consistency** (transcribe → engrave → resynth → re-transcribe) | mir_eval.transcription on (MIDI₁, MIDI₂) | Audio → 2 MIDIs → F1 between them | RF | Concept after Simonetta 2022, https://arxiv.org/abs/2202.12257 |
| FAD (Frechet Audio Distance) | `microsoft/fadtk` | Set of resynthesized clips, reference set → FAD scalar | RF (set-level) | Kilgour 2019, https://www.isca-archive.org/interspeech_2019/kilgour19_interspeech.pdf ; Gui 2023 (generative adaptation), https://arxiv.org/abs/2311.01616 |
| Perceptual resynthesis MOS-predictor | Simonetta 2022 model | Same | RF | https://link.springer.com/article/10.1007/s11042-022-12476-0 |

**Re-synthesis recipe (FluidSynth + a clean piano soundfont)**:

1. Load the engraved MIDI (URI from `EngravedOutput.humanized_midi_uri`,
   `shared/shared/contracts.py:348-377`) with `pretty_midi`.
2. Synthesize via fluidsynth using the bundled `TimGM6mb.sf2`
   (`scripts/eval_transcription.py:115-131`) — **already a working pattern**;
   reuse the binary lookup at `_find_fluidsynth` and the soundfont
   discovery at `_default_soundfont`. For higher-fidelity perceptual eval,
   consider a Steinway-D or Salamander Grand SF2 (CC-licensed).
3. Truncate the original audio and the resynth to identical 30-s windows.
4. Run CLAP-music / MERT over both → cosine.

**Round-trip diagnostic** (powerful for stage isolation):

```
audio → transcribe → MIDI₁                         (the actual transcription)
         ↓
       arrange → humanize → engrave → MIDI_engraved
                                          ↓ fluidsynth
                                       audio_resynth
                                          ↓ transcribe (same model)
                                       MIDI₂

mir_eval F1(MIDI₁, MIDI₂) measures how much the arrange/engrave hops lose.
A drop here implicates arrange or engrave; a low MIDI₁-vs-original-cover F1
implicates transcribe.
```

**Expected ranges** (pop, original audio vs resynthesized piano transcription):
- CLAP-music cosine: 0.50–0.85 for plausible piano renderings; <0.40 means
  the piano cover doesn't sound like the song. Use as a **screen, not a
  verdict**; the literature warns CLAP correlates only modestly with MOS.
- MERT cosine: similar ballpark, slightly more music-specific.
- Chroma cosine (per bar, then mean): 0.55–0.85 on faithful covers; 0.30–0.45
  on heavily simplified covers.
- Round-trip F1 (no-offset): 0.60–0.85 if engrave preserves MIDI losslessly;
  the current Oh Sheet engrave drops dynamics/articulations but keeps notes
  + pedal CCs (`backend/services/midi_render.py:99-117`), so we expect 0.70+;
  large drops here indicate `arrange_simplify` is decimating notes.

**Gotchas**:
- FAD requires hundreds of samples for stable estimates (sample-size bias
  is severe — never trust below 500 samples per the FAD paper). Run only
  at corpus scale.
- CLAP-music checkpoint outperforms general LAION-CLAP for music; use the
  `music_audioset_epoch_15_esc_90.14.pt` checkpoint.
- Spectrogram-domain phase artifacts in resynthesis can artificially lower
  CLAP cosine without lowering musical fidelity. Prefer high-quality SF2;
  consider neural piano synthesis (DDSP-Piano) if budget allows.
- The current pipeline drops dynamics at engrave (`runner.py:516-520` —
  `includes_dynamics=False` hard-coded), so the resynth piano will sound
  dynamically flat. This will *systematically suppress* CLAP scores until
  the engrave stage learns dynamics. **Useful side effect**: when we ship
  dynamics, CLAP cosine should jump.

**When**: Per-PR (per-song CLAP cosine + chroma cosine + round-trip F1
on a 5-song subset). Nightly (full corpus + FAD).

#### 2.5 Tier 5 — human evaluation

Automated metrics never close the loop alone. Tier 5 anchors all the cheaper
metrics in human judgement. **Tier 5 is the only Tier that can validate
Tiers 1–4 themselves**.

| Protocol | What | When | Cost |
|---|---|---|---|
| **Multi-axis MOS** | N≥30 listeners rate each item 1–5 on Faithfulness / Playability / Readability / Musicality / Expressivity | Every release | $300–800 per release on Prolific |
| **A/B preference (paired comparison)** | N≥30 listeners pick {Oh Sheet new vs. previous release} or {Oh Sheet vs. baseline}; report win-rate with binomial CI | Every release | $200–500 |
| **Pianist rubric (expert)** | ≥3 pianists score on 5-axis Likert: enharmonic spelling, beam grouping, voice separation, fingering implications, stem direction. Calibrate with Krippendorff α ≥ 0.6 | Every release | $300–600 (3 pianists × 1.5 hr × $80/hr) |
| **Sight-readability test** | Show transcription to N≥10 pianists at the target skill level; measure error rate, time-to-fluency, self-reported readability | Quarterly | $800–1500 (10 pianists × 2 hr × $40/hr) |
| **In-network musician volunteers** | Same protocols on smaller scale | Weekly | $0 |

**Calibration** (run once at project setup, every 6 months thereafter):
- Inter-rater reliability target: Krippendorff's α ≥ 0.6 (acceptable for
  subjective music ratings) — see Castro et al. on MIR rater reliability.
- Train a regression: `MOS ≈ f(Tier 2/3/4 metrics)` and require *mean
  absolute error ≤ 0.5* on a 5-point scale before trusting the
  automated triple as a proxy.
- Pre-register the **win-rate threshold**: e.g. ≥55% blind preference vs.
  previous release with binomial 95% CI excluding 50%.

**Multi-axis MOS rubric (recommended starting point)**:

```
1. Faithfulness — "Does this sound like a piano cover of the right song?"
   1 = unrecognizable; 3 = recognizable with effort; 5 = clearly the song
2. Playability — "Could a typical pianist actually play this at sight?"
   1 = unplayable; 3 = needs slow practice; 5 = sight-readable
3. Readability — "Is the engraved notation clear and well-organized?"
   1 = chaotic; 3 = clear with awkwardness; 5 = clean and idiomatic
4. Musicality — "Does this feel musical, not robotic?"
   1 = flat/mechanical; 3 = passable; 5 = expressive
5. Expressivity — "Does dynamics/articulation/pedaling enhance the piece?"
   1 = none; 3 = present but generic; 5 = nuanced
```

Open-source the rubric (publish to GitHub) so the field can compare. This
matches MMMOS-style multi-axis rating
(https://arxiv.org/html/2507.04094v2).

**Recruitment**:
- **Prolific** for MOS / A/B at scale: ~$8/listener-hour, music-listener
  filter available.
- **Music Hackathon / r/piano / friends-of-team Discord** for cheap
  pianist rubric.
- **TheoryTab community** has a pool of musically-literate volunteers.

---

### 3. Reference data design

#### 3.1 Build vs. buy — verdict: build a 30-50 song internal pop eval set

**Recommendation**: contract with 1-2 transcribers over 4-8 weeks to produce
30-50 paired (audio, piano cover MIDI, engraved PDF, structural ground truth)
artifacts. This is the unique cost-bearing investment; everything else
follows public datasets.

**Justification**:
- No public pop-paired-piano dataset exists. Pop2Piano's PSP corpus is not
  released; POP909 is MIDI-only with no audio (`09-research-datasets.md` §1).
- Klangio / MuseScore.com piano covers are derivative works under
  copyrighted source audio — not redistributable
  (`09-research-datasets.md` §2).
- Public alternatives are wrong-distribution (MAESTRO/MAPS = classical;
  Slakh = synthetic; MUSDB18 = no MIDI labels).

**Cost estimate**:
- 30 transcriptions × ~$30/song (contract pianist) = ~$900.
- Audio licensing for a redistributable subset: prefer FMA `commercial=true`
  pop tracks (free) for 10 songs; license 20 commercial tracks at ~$50
  each (synchronization license for internal-only research) ≈ $1000.
- Engineer time to QA + structurally annotate (key, sections, chord
  progression): 1 week × $3000.
- **Total: ~$5k for the first 30-song eval set.** Scale to 50 with another
  $1.5k-$2k.

#### 3.2 Per-song artifact bundle

For each eval song, store under `eval/pop_eval_v1/<song-slug>/`:

```
eval/pop_eval_v1/
├── manifest.yaml          # version, license, source URL, song id
└── songs/
    └── <song-slug>/
        ├── source.audio.json          # link to audio (NOT the audio itself if licensed-only),
        │                              # plus content_hash, sample_rate, duration, format
        ├── source.audio                # IFF redistributable (FMA / CC-licensed track)
        ├── reference.piano_cover.mid  # contract-transcribed MIDI ground truth
        ├── reference.piano_cover.musicxml  # engraved reference (if commissioned)
        ├── reference.piano_cover.pdf       # PDF of the engraved reference
        ├── structural.yaml             # human-verified key, time-sig, tempo, sections,
        │                              # chord progression (Harte notation), downbeats
        ├── tier1_baseline.json         # baseline mir_eval scores (last release)
        └── notes.md                    # transcriber notes, known difficulties
```

**Key fields in `structural.yaml`**:

```yaml
key: "C major"
time_signature: "4/4"
tempo_bpm: 124.5
tempo_map: [[0.0, 124.5], [60.0, 122.0]]   # optional gradual changes
sections:
  - {name: "intro",   start_sec: 0.0,  end_sec: 12.5,  bars: "1-4"}
  - {name: "verse 1", start_sec: 12.5, end_sec: 28.7,  bars: "5-12"}
  - {name: "chorus",  start_sec: 28.7, end_sec: 44.5,  bars: "13-20"}
chord_progression:
  - {start_sec: 0.0,  label: "C:maj"}
  - {start_sec: 4.2,  label: "F:maj"}
  - {start_sec: 8.4,  label: "G:7"}
downbeat_sec: [0.0, 1.95, 3.90, 5.85, ...]
license: "FMA-commercial / Internal Research Only / SFM Custom $50"
```

The contract here is `HarmonicAnalysis` (`shared/shared/contracts.py:185-190`)
extended with downbeats — store the YAML in a way that loads directly into
that Pydantic model.

#### 3.3 Holdout split — never used for tuning, always used for release gates

Split the 30-50 song corpus once, freeze, never re-split:

- **Tune set (50%)**: 15-25 songs. Used for hyperparameter tuning, model
  selection, prompt engineering. CI runs on this set every PR.
- **Holdout (50%)**: 15-25 songs. **Never seen by any tuning loop.** Run
  *only* at release gate. If holdout regresses, the release blocks.

This is a hard rule. The temptation to peek at holdout to debug a
regression is enormous; resist by storing holdout IDs in a separate
encrypted manifest (`eval/pop_eval_v1/holdout_manifest.yaml.enc`) that
only the release-bot has the key for. Engineers debugging regressions
must reproduce the regression on tune set.

#### 3.4 Versioning — eval sets are tagged + frozen

Adopt semver for eval sets:
- `pop_eval_v1.0.0` — initial 30-song release
- `pop_eval_v1.1.0` — bug-fix in structural.yaml of 2 songs
- `pop_eval_v2.0.0` — added 20 songs to round out coverage

Every CI run logs the eval-set version it ran against. Cross-version
comparisons require re-running the older release on the new eval set.
Store `eval-baseline.json` per `(eval_version, oh_sheet_release)` tuple.

#### 3.5 Public-corpus complements

To anchor classical-piano regressions and provide "hard-mode" diagnostic
inputs:

| Corpus | What it gives | Where it goes in eval | License |
|---|---|---|---|
| **MAESTRO v3 test split** | ~20h Disklavier-clean classical piano with paired MIDI | Tier 1 regression baseline; if classical drops, AMT is broken | CC-BY-NC-SA 4.0 (research only) |
| **MAPS** | Classical piano + Disklavier subset | Cross-dataset OOD check | CC-BY-NC-SA 2.0 FR |
| **POP909** | 909 MIT-licensed pop *piano arrangements* (MIDI only) | Symbolic-side eval: synthesize POP909 MIDIs to audio, transcribe, expect to recover the original; rough ceiling for pop-piano AMT | MIT |
| **Slakh2100** | 2100 synthetic multitrack pop-style mixes with paired MIDI | Stem-separation regression; multi-instrument AMT test | CC-BY 4.0 |
| **MUSDB18-HQ** | 150 real pop tracks with V/D/B/O stem isolation | Source-separation regression (no piano MIDI labels) | Educational only — research use |
| **clean_midi (Lakh subset)** | 25 MIDIs already in `eval/fixtures/clean_midi/` | Existing fluidsynth-rendered baseline; keep but rename to `eval/synth_baseline_v1/` | CC-BY 4.0 |

**Recommendation**: keep the existing `eval/fixtures/clean_midi/` as a
narrow regression check ("FluidSynth piano render → Basic Pitch → mir_eval
should not crater") and label it `synth_baseline_v1`. Add MAESTRO test
split (CC-BY-NC-SA → research-only, fine for internal eval) for classical
regression. Add POP909 (MIT) for symbolic-side pop-piano sanity. Use
MUSDB18-HQ for separation eval. The new `pop_eval_v1` eval set sits on
top as the pop-audio anchor.

#### 3.6 Reference-free fallback when ground truth is missing

For an arbitrary user upload (no ground truth at all), only Tiers 2 / 3 / 4
are usable:

| Tier | Metrics that work without reference |
|---|---|
| Tier 2 | Key/tempo/beat/chord/section *agreement vs. audio-side estimator on the same input audio* — measures internal consistency |
| Tier 3 | Playability fraction, voice-leading smoothness, polyphony density, sight-readability, engraving heuristic checks |
| Tier 4 | All of CLAP-music cosine, MERT cosine, chroma cosine, round-trip self-consistency |

These constitute the **production "quality score"** (§8) — every user
upload gets a score that we can log, alert on, and surface to users as a
confidence signal.

---

### 4. Test harness architecture

#### 4.1 Filesystem layout (`eval/` directory expansion)

The current layout (`eval/fixtures/clean_midi/`,
`scripts/eval_transcription.py`, `eval-baseline.json`) is too narrow. Expand
to:

```
eval/
├── README.md                          # how to run + what each set is
├── pop_eval_v1/                       # the curated pop set (§3)
│   ├── manifest.yaml
│   ├── holdout_manifest.yaml.enc      # encrypted holdout list
│   └── songs/<slug>/...
├── synth_baseline_v1/                 # renamed from `fixtures/clean_midi/`
│   └── ABBA/Knowing Me, Knowing You.5.mid
│   └── ...
├── maestro_test/                      # gitignored; instructions in README
├── pop909/                            # gitignored; download script
├── slakh2100/                         # gitignored
├── musdb18hq/                         # gitignored
├── baselines/
│   ├── pop_eval_v1__release_2026.04.json
│   ├── synth_baseline_v1__release_2026.04.json
│   └── maestro_test__release_2026.04.json
└── runs/                              # gitignored — per-run artifacts
    └── 2026-04-25T13-22-00__abc123/
        ├── per_song/<slug>/
        │   ├── transcription.mid
        │   ├── arrangement.mid
        │   ├── humanized.mid
        │   ├── engraved.musicxml
        │   ├── resynth.wav
        │   ├── tier1.json
        │   ├── tier2.json
        │   ├── tier3.json
        │   ├── tier4.json
        │   └── stage_artifacts/...
        ├── aggregate.json
        ├── per_tier_summary.md
        └── manifest.yaml              # eval-set version, oh-sheet sha,
                                       # config.py snapshot, env hash
```

#### 4.2 Driver script — Click CLI extending the existing pattern

Build a **single CLI** with subcommands, modeled on the working
`scripts/eval_transcription.py` pattern. Add a thin Click wrapper so we
get subcommands without a re-architecture.

```
scripts/eval.py
├── eval transcribe     # Tier 1+2 — audio→MIDI only (replaces current eval_transcription.py)
├── eval arrange        # Tier 3 — MIDI→PianoScore (skip transcribe)
├── eval engrave        # Tier 3 — PianoScore→MusicXML
├── eval end-to-end     # All tiers
├── eval round-trip     # transcribe→engrave→resynth→re-transcribe self-consistency
├── eval ci             # Cheap PR gate: 5-song subset, Tier 2/3 only
├── eval nightly        # Full corpus, all tiers including FAD
└── eval compare        # diff two run JSONs (ports `compare_eval_runs.py`)
```

Per-stage diagnostic mode is the key win: each subcommand can take a
`--start-from` flag (`--start-from=transcribe`, `--start-from=arrange`,
`--start-from=engrave`) so a regression in arrange can be reproduced
without re-running transcribe. **This requires the harness to cache stage
outputs in the run directory** (the layout in §4.1 above already does).

The existing harness:
- `scripts/eval_transcription.py:592-602` invokes
  `backend.services.transcribe._run_basic_pitch_sync` directly. We extend
  this to `_run_arrange_sync` (`backend/services/arrange.py:427-517` —
  there's already an `_arrange_sync`), `_run_engrave_sync`
  (`backend/services/midi_render.py:36-137` + `ml_engraver_client.py:107-141`).
- `scripts/compare_eval_runs.py` already exists for run-to-run diffing.
  Extend it with Tier-2/3/4 metrics.

#### 4.3 Per-stage diagnostic mode (worked example)

```python
## Pseudocode for eval/diagnostic.py

def diagnose_song(song_slug: str, eval_set: str) -> Diagnosis:
    artifacts = load_artifacts(eval_set, song_slug)
    audio_uri = artifacts.audio_uri
    ref_midi = artifacts.reference_piano_cover_mid
    ref_xml = artifacts.reference_piano_cover_musicxml
    ref_struct = artifacts.structural

    # Stage 1: transcribe-only
    txr = run_transcribe(audio_uri)
    tier1 = mir_eval_transcribe(txr.midi, ref_midi)
    tier2_t = compare_structural(txr.analysis, ref_struct)

    # Stage 2: arrange-only (use transcribe output)
    score = run_arrange(txr)
    tier3_a = playability(score) | voice_leading(score)

    # Stage 3: engrave-only (use arrange output)
    perf = run_humanize(score)
    eng = run_engrave(perf)
    tier3_e = mv2h(eng.musicxml, ref_xml)

    # Stage 4: end-to-end perceptual
    resynth = fluidsynth(eng.midi)
    tier4 = clap_cosine(audio, resynth) | round_trip_f1(audio, eng.midi)

    return Diagnosis(tier1, tier2_t, tier3_a, tier3_e, tier4)
```

If the diagnosis shows `tier1` regressed but `tier3_a/tier3_e` are stable,
transcribe regressed. If `tier1` stable but `tier3_a` dropped, arrange
regressed. If `tier4` dropped but everything upstream is stable, engrave
regressed (e.g., dynamics dropped, MIDI corrupted). **This is the unique
value Oh Sheet's eval needs**: the current single-F1 approach can never
do this.

#### 4.4 Round-trip diagnostic (transcribe → engrave → resynth → re-transcribe)

A self-consistency probe that requires **no reference MIDI**:

```python
def round_trip(audio_path: Path) -> dict:
    midi1 = transcribe(audio_path)         # the actual transcription
    score = arrange(midi1)
    perf = humanize(score)
    eng_midi = engrave_midi(perf)          # render_midi_bytes (midi_render.py:36)
    resynth_wav = fluidsynth(eng_midi)
    midi2 = transcribe(resynth_wav)        # re-transcribe the engraved MIDI
    return {
        "round_trip_f1_no_offset": mir_eval_f1(midi1, midi2, offset_ratio=None),
        "round_trip_f1_with_offset": mir_eval_f1(midi1, midi2, offset_ratio=0.2),
        "n_notes_dropped": len(midi1) - len(midi2),
    }
```

Drops in round_trip_f1 implicate the arrange/humanize/engrave hops. The
critical loss right now is at engrave (`midi_render.py:99-117` only writes
notes + pedal CCs; `runner.py:516-520` hard-codes
`includes_dynamics=False`). The round-trip metric will quantify the loss
song-by-song without needing a reference cover.

#### 4.5 Parallelization — running 30 songs in <15 min on a 16-core CI machine

The current harness loops songs sequentially
(`scripts/eval_transcription.py:1055-1060`, ~4–10 s per song). For 30
songs × all five tiers × ~30 s/song/stage that's ~75 min sequential.

**Parallelization strategy**:
- Fan out per song with `concurrent.futures.ProcessPoolExecutor`. Pop2Piano
  / Demucs are GPU-bound on the typical CI runner (no GPU) so they're
  I/O-light; CPU concurrency works.
- Cap workers at `min(n_cores - 2, n_songs)` — leave headroom for the
  process pool overhead.
- Cache stage artifacts under `eval/runs/<run-id>/per_song/<slug>/`.
  Hashing keys: `sha256(audio_content || stage_config_json)`.
- Reuse the pretty_midi WAV synthesis cache pattern at
  `scripts/eval_transcription.py:347-371` for resynthesis — the same
  `(midi, soundfont, sample_rate)` tuple should hit the cache across runs.

Targets:
- `eval ci` (5 songs, Tier 2+3 only, no Demucs): **<2 min** wall on 4-core
  GitHub Actions runner.
- `eval nightly` (30 songs, all tiers): **<15 min** wall on 16-core runner
  (4 songs in parallel × 4 stages each = ~30 s × 2 batches ≈ 4 min for
  inference; FAD adds ~3 min on the corpus; CLAP/MERT adds ~2 min).

**GPU acceleration** is optional but worth it for FAD and MERT: a single
T4 cuts FAD from ~3 min to ~30 s. Add a `--device=cuda` flag and let CI
opt in.

#### 4.6 Storage of per-run artifacts

Every run stores under `eval/runs/<run-id>/`:

- Per-song: transcription MIDI, arrange MIDI, humanized MIDI, engraved
  MusicXML, resynthesized WAV — for human spot-checks and bisect.
- Aggregate JSON with all tier scores.
- A markdown summary (`per_tier_summary.md`) that quickly says "Tier 1
  changed by +X.X ppt vs. baseline; Tier 3 playability dropped by Y%" so
  reviewers don't have to grep JSON.
- A manifest with eval-set version, oh-sheet git SHA, env hash, config
  snapshot. Without this, A/B comparisons are meaningless.

Retain artifacts for: every release, every nightly, last 7 days of PR
runs. Auto-prune older PR runs.

---

### 5. CI gates and release gates

#### 5.1 Cheap CI on every PR (<2 min on every PR)

Run `eval ci`: a 5-song subset of the tune set (chosen for diversity:
1 mainstream pop, 1 hip-hop, 1 ballad, 1 K-pop, 1 indie/electronic).

**Metrics computed**:
- Tier 2: key, tempo, chord (`mirex` mode), beat F1 — RF, ~5 s/song.
- Tier 3: playability fraction, voice-leading smoothness, polyphony
  density, basic engraving warnings — RF, ~3 s/song.
- Tier 4: CLAP-music cosine, chroma cosine, round-trip self-consistency
  F1 — RF, ~10 s/song.
- Per-stage timing (regression on latency).

**Gates** (block PR merge if violated):
- Tier 2 chord-mirex accuracy regresses > 3 ppt vs. main.
- Tier 3 playability fraction drops > 5 ppt vs. main.
- Tier 4 round-trip F1 drops > 5 ppt vs. main (catches engrave regressions).
- CLAP-music cosine drops > 0.05 absolute vs. main.
- Any Tier 3 engraving heuristic check raises a *new* error class
  (e.g., "voice crossing detected" appearing where it didn't before).

**Hard fail (always)**:
- Stage-pipeline returns the C-E-G-C stub
  (`backend/services/transcribe_result.py:255-264`) on a real audio input
  — a test for "is the stub firing in production code paths".
- Engrave HTTP service returns < 500 bytes (the existing
  `_looks_like_stub` check at `ml_engraver_client.py:103-104` — also assert
  this in CI).

**Soft warn (review required, don't block)**:
- Per-song variance: any single song regresses Tier 2/3/4 by > 10 ppt
  while corpus mean is stable. This catches "we improved on average but
  destroyed Despacito".

#### 5.2 Nightly: full corpus, all tiers including FAD

Run `eval nightly`:
- 30-song tune set + 30-song MAESTRO test split + 50-song POP909 sample.
- All tiers including Tier 1 (mir_eval onset/offset/velocity F1) on
  references, MV2H on references, sight-readability, FAD over the
  resynthesized corpus.
- Render PDFs (when engrave service supports it) and pin them to the run
  directory for human spot-check.
- Output a single markdown summary with sparklines vs. last 14 nights.

**Alerts**:
- Slack `#oh-sheet-eval` channel: any Tier metric regresses > 5 ppt
  night-over-night.
- Pager: any Tier 1 *crater* — F1 drops > 15 ppt.
- A/B tracking: every Slack post links to the per-song dashboard.

#### 5.3 Release gate: Tier 5 + nightly + pre-registered minimum win rate

Block a release if:
- Holdout (§3.3) regresses any tier > 5 ppt vs. previous release.
- A/B win-rate vs. previous release < 55% with binomial 95% CI excluding
  50%. Sample size: 30 listeners × 30 songs minimum. Pre-register both
  the win-rate threshold and the sample size *before* running the study.
- Pianist rubric (3 raters × 30 songs) shows mean drop on any axis > 0.4
  (on 1-5 scale).
- Sight-readability test (10 pianists × 5 songs): error rate > +20% vs
  previous release.

The pre-registration matters: define the test in `RELEASE-GATES.md` in
the repo, sign off by product+eng before the study. This avoids
post-hoc cherry-picking.

#### 5.4 Statistical methodology

**Bootstrap confidence intervals** (replace point-estimate reporting):
- Resample 1000× over songs (for per-corpus aggregates) or over
  rater-song pairs (for MOS).
- Report 95% CI on each tier metric. A "regression" is real only when
  the CIs don't overlap.
- Per-song, report CI by re-running with different seeds (where
  applicable, e.g., Pop2Piano sampling temperature).

**Multiple-comparison correction**:
- We're computing ~10 metrics across ~30 songs each PR. Bonferroni at
  α=0.05 per-PR is too conservative; use Benjamini-Hochberg FDR at q=0.10
  for "this PR caused a regression somewhere" tests.
- For headline release-gate metrics (key, playability, CLAP, round-trip,
  win-rate), pre-register the alpha and don't correct further; for
  exploratory drill-downs, FDR-correct.

**Sample-size sufficiency**:
- A 30-song eval set gives ±0.10 95% CI on a metric with σ=0.30
  (typical Tier 2 chord accuracy variance). To detect a 5 ppt
  regression with 80% power requires ~50 songs. **Plan to grow eval
  set to 50 by end of Q2.**
- For MOS / A/B, 30 listeners × 30 items gives ±0.20 95% CI on MOS;
  to detect a 0.4 MOS difference with 80% power requires ~60 listeners
  per group. Budget for 60 raters at every release study.

---

### 6. Continuous benchmarking

#### 6.1 Internal Grafana dashboard

Per-tier, per-song, per-release time-series charts. Production
deployment ships an evaluator-as-a-sidecar: every job emits a Tier 2/3/4
summary into a Postgres table; Grafana reads from there.

Schema:
```sql
CREATE TABLE eval_runs (
  run_id UUID PRIMARY KEY,
  created_at TIMESTAMP NOT NULL,
  eval_set_version TEXT NOT NULL,
  oh_sheet_sha TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  is_release_run BOOLEAN NOT NULL DEFAULT FALSE,
  is_nightly BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE TABLE eval_song_scores (
  run_id UUID REFERENCES eval_runs(run_id),
  song_slug TEXT NOT NULL,
  tier TEXT NOT NULL,           -- '1', '2', '3', '4', '5'
  metric_name TEXT NOT NULL,
  metric_value DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (run_id, song_slug, tier, metric_name)
);
CREATE TABLE eval_production_quality_scores (
  job_id UUID NOT NULL,
  created_at TIMESTAMP NOT NULL,
  user_audio_hash TEXT NOT NULL,    -- sha256 of audio bytes; not the audio itself
  composite_quality_score DOUBLE PRECISION NOT NULL,    -- §8
  tier2_chord_accuracy DOUBLE PRECISION,
  tier3_playability_fraction DOUBLE PRECISION,
  tier4_clap_cosine DOUBLE PRECISION,
  tier4_round_trip_f1 DOUBLE PRECISION,
  ...
);
```

#### 6.2 Public leaderboard — wait

Recommendation: **don't publish a public leaderboard until pop_eval_v1
is published as a redistributable corpus** (which requires a
licensing-clean subset). Premature leaderboards on internal-only data
draw confusion. After pop_eval_v1.0 has 10+ FMA-licensed songs, publish
those + a public leaderboard at e.g. `eval.oh-sheet.com` with model
weights people can submit.

#### 6.3 Per-song dashboards for engineer A/B

Every per-song page shows:
- Audio waveform of the input.
- Resynthesized output waveform.
- Score panel: rendered MusicXML.
- Tier 1/2/3/4 metric history (last 30 nights) for this song.
- Diff against last release: which notes were added/dropped.

Build with Streamlit on top of the Postgres tables. ~1 week of work.

#### 6.4 Alerting on regression > X ppt

Wire Slack alerts:
- Nightly Tier 2 chord drop > 3 ppt → `#oh-sheet-eval` post.
- Nightly Tier 4 round-trip drop > 5 ppt → page on-call.
- Per-song crater (any single song > 15 ppt drop on any tier metric) →
  `#oh-sheet-eval` post (no page, since aggregate may be fine).

#### 6.5 A/B framework — treat model upgrades like product experiments

When swapping models (Basic Pitch → Kong, condense_only → arrange,
rule-humanize → trained), use a feature flag and a traffic split:
- 5% traffic to the new path for 24 hours; emit production quality
  scores (§8) into Postgres.
- Compare distributions: KS-test on composite quality score, plus
  per-tier comparisons.
- If new path wins on composite quality + no tier regresses, ramp to
  20%, 50%, 100%.

This needs the production quality score (§8) to actually correlate with
human judgement. Tier 5 calibration validates that.

---

### 7. Human-in-the-loop evaluation

#### 7.1 Pianist rubric (5 axes, 1-5 Likert)

| Axis | Description | Anchor at 1 | Anchor at 5 |
|---|---|---|---|
| **Faithfulness** | Does it sound like a piano cover of the right song? | Unrecognizable | Clearly the song |
| **Playability** | Could a typical pianist actually play this? | Physically impossible | Sight-readable |
| **Readability** | Is the engraved notation clear? | Chaotic, illegible | Clean, idiomatic |
| **Musicality** | Does it feel musical, not robotic? | Mechanical | Expressive |
| **Expressivity** | Does dynamics/articulation/pedaling enhance the piece? | None present | Nuanced |

**Calibration session**: before the first study, hold a 1-hour video call
with all raters where they:
1. Score 5 reference items together (3 from existing piano cover sites,
   2 from Oh Sheet beta).
2. Discuss disagreements, lock anchors per axis.
3. Run a 10-item warmup independently; compute Krippendorff's α.
4. Repeat until α ≥ 0.6 on each axis (acceptable for subjective ratings).
5. Lock the rubric. **Do not modify mid-study.**

Open-source the calibration data + rubric on the Oh Sheet GitHub so the
field can use it.

#### 7.2 Listener MOS protocol (audio side-by-side, ABX)

For listener (non-pianist) MOS:
- Audio side-by-side: listener hears (a) the original audio for 15 s, then
  (b) the resynthesized piano transcription for 15 s. Rate 1-5 on
  faithfulness only (listeners often can't judge readability).
- ABX: when comparing two systems, listener hears A, B, then a
  randomly-selected hidden A or B; must say which they heard. This is
  the gold standard for "can listeners tell these apart at all".

Tools: [Beaqle](https://github.com/HSU-ANT/beaqle) for ABX; Prolific
for recruitment; Pavlovia for hosting.

#### 7.3 Recruiting + cost

- **Prolific**: ~$8/listener-hour, music-listener filter available;
  60 listeners × 30 min = $240/study. Run quarterly: $1k/year.
- **Music Hackathon / r/piano / friends**: free; lower volume but useful
  between formal studies.
- **In-network musician volunteers**: find 3-5 pianists in the team's
  extended network for the rubric. Pay $80/hr for 1.5 hr × 5 raters =
  $600/study.

#### 7.4 Open-source the rubric

Publish to `eval/rubric/pianist_rubric.md`,
`eval/rubric/listener_mos_protocol.md`, `eval/rubric/calibration_data.csv`
in the public Oh Sheet repo. The field has no standard rubric for
audio→sheet evaluation; this contribution is a small but concrete public
good and increases citation surface.

---

### 8. Reference-free evaluation deep-dive

This is the **unique production challenge for Oh Sheet**: most users will
upload songs we have no piano cover for. So every metric we use to monitor
production quality must work without a reference.

#### 8.1 Reference-free metric inventory (from prior tiers)

**Tier 2 (RF)**: key (audio-side estimate vs. transcription-side estimate),
tempo, beat F1, downbeat F1, time-sig, chord progression, sections.

**Tier 3 (RF)**: playability fraction, voice-leading smoothness, polyphony
density, sight-readability score, engraving heuristic checks.

**Tier 4 (RF, audio-anchored)**: CLAP-music cosine, MERT cosine, chroma
cosine, round-trip self-consistency F1.

#### 8.2 Composite per-song quality score

Combine into a single scalar `Q ∈ [0, 1]`:

```python
def quality_score(song) -> float:
    # Tier 2 — structural agreement (RF, audio-anchored)
    s_key   = mir_eval.key.weighted_score(audio_key, txr_key)         # [0, 1]
    s_chord = mir_eval.chord.weighted_accuracy(audio_chord, txr_chord)
    s_beat  = mir_eval.beat.f_measure(audio_beats, txr_beats, 0.07)
    s_tempo = tempo_p_score(audio_tempo, txr_tempo)
    tier2 = (s_key + s_chord + s_beat + s_tempo) / 4

    # Tier 3 — playability (RF)
    s_play = playability_fraction(score)                              # [0, 1]
    s_vleading = 1 - voice_leading_displacement(score) / 12           # rescale st to [0, 1]
    s_density = density_in_target_range(score, target=2.5)            # [0, 1]
    tier3 = 0.5 * s_play + 0.3 * s_vleading + 0.2 * s_density

    # Tier 4 — perceptual (RF, audio-anchored)
    s_clap = (clap_cosine(audio, resynth) + 1) / 2                    # [-1, 1] → [0, 1]
    s_chroma = chroma_cosine(audio, resynth)
    s_rtf1 = round_trip_f1(audio)                                     # [0, 1]
    tier4 = (s_clap + s_chroma + s_rtf1) / 3

    # Weighted composite. Weights chosen by Tier 5 calibration.
    return 0.30 * tier2 + 0.30 * tier3 + 0.40 * tier4
```

The weights (0.30 / 0.30 / 0.40) are placeholders — they get **calibrated
against human MOS** in §8.3. Initial intuition: Tier 4 weighted highest
because it's the closest to "does this sound like a piano cover of the
song"; Tier 3 weighted at 0.30 because playability is a hard floor for
the product; Tier 2 weighted at 0.30 because pop chord/key correctness
is most of what listeners notice.

#### 8.3 Validation — does it correlate with human judgement?

Run Tier 5 (pianist rubric + listener MOS) on 30 songs, get mean MOS
per song, then:

1. Compute Spearman rank correlation `ρ(Q, MOS_overall)`.
2. Compute Kendall's `τ` for robustness.
3. **Target: `ρ ≥ 0.7` for the composite score against listener
   faithfulness MOS, `ρ ≥ 0.6` against pianist musicality MOS.**
4. If correlation is below target, tune the weights via constrained
   ridge regression: `MOS ≈ w₁·tier2 + w₂·tier3 + w₃·tier4` subject
   to `wᵢ ≥ 0`, `Σwᵢ = 1`. Refit at every Tier 5 study.
5. Publish the calibration as a versioned artifact: `Q_v1.0` is the
   version released with the original rubric; `Q_v1.1` after first
   recalibration.

If Tier 4 (perceptual) doesn't correlate, drop CLAP, try MERT.
If Tier 3 dominates correlation, that's a signal that the user-perceived
problem is mostly playability — focus engineering there.

#### 8.4 Production deployment

Every job emits a `Q` score into the Postgres table from §6.1. Surface
to the user as a confidence:
- `Q ≥ 0.75` → "High quality — ready to play"
- `0.55 ≤ Q < 0.75` → "Decent — may need some clean-up"
- `Q < 0.55` → "Lower confidence — try a clearer source recording"

This single number is the smallest contract between automated eval and
user expectations.

---

### 9. Concrete first-90-days execution plan

#### Week 1 — foundation

**Goal**: stand up the eval-set scaffolding + first Tier 2/3 metrics.

**PRs**:
- **PR-1: `eval/` directory restructure**. Rename
  `eval/fixtures/clean_midi/` → `eval/synth_baseline_v1/`. Add
  `eval/README.md` with the layout in §4.1. Update
  `scripts/eval_transcription.py` paths.
- **PR-2: New CLI scaffold**. Add `scripts/eval.py` with a Click app
  that wraps `eval_transcription.py` as the `transcribe` subcommand.
  No new metrics yet; just the harness.
- **PR-3: Tier 2 RF metrics module**. Add `eval/tier2_structural.py`
  with key/tempo/beat/chord comparison against an audio-side
  estimator (madmom downbeats + Krumhansl-Schmuckler key + chord
  recognition). Existing functions in
  `backend/services/key_estimation.py:60-75` and
  `backend/services/audio_timing.py:99-148` give us the audio-side
  side.

#### Sprint 1 — first weeks (weeks 2-3)

**Goal**: First curated pop eval set; Tier 3 RF metrics; CI gate.

**Tasks**:
- Contract a transcriber (Upwork / Fiverr music transcriber, $30/song
  × 10 songs to start).
- Source 10 FMA-commercial pop tracks (free) for the redistributable
  bucket; 20 commercial via $50 sync licenses for internal-only.
- **PR-4: Tier 3 RF metrics**. Add `eval/tier3_arrangement.py` with
  playability, voice-leading, polyphony density. Use music21 (already
  in deps, `pyproject.toml:18`).
- **PR-5: CI gate config**. GitHub Actions workflow `eval-ci.yml` that
  runs `eval ci` on PRs to main; gates on §5.1 thresholds.
- **PR-6: Per-stage diagnostic harness**. Extend `scripts/eval.py` with
  `arrange`, `engrave`, `round-trip` subcommands. Cache stage
  artifacts per §4.5.
- **PR-7: Tier 4 RF metrics — chroma + round-trip**. Reuse FluidSynth
  pattern from existing harness; reuse mir_eval calls.

#### Month 1 — pop eval set v1.0 + nightly

**Goal**: 30-song eval set frozen; nightly running; Grafana dashboard.

- **30-song corpus**: complete the contract transcriptions. Lock the
  holdout split. Tag `pop_eval_v1.0.0`.
- **PR-8: Nightly workflow**. GitHub Actions cron at 2 AM UTC; runs
  `eval nightly`; posts summary to `#oh-sheet-eval` Slack.
- **PR-9: Postgres + production telemetry**. Schema from §6.1; emit
  composite quality score from every production job.
- **PR-10: Grafana dashboard**. Time-series panels for per-tier
  metrics, per-song panels, CI vs. nightly comparison.
- **PR-11: Tier 4 CLAP-music**. Wire LAION-CLAP music checkpoint;
  verify cost (~10 s/song); land in nightly first, then per-PR after
  benchmarking.

#### Quarter 1 — Tier 1 + MV2H + Tier 5 + composite-Q calibration

**Goal**: full ladder running; first Tier 5 study; Q calibrated to MOS.

- **Tier 1 onset/offset/velocity F1** against the 30-song reference
  MIDIs (all already accessible since pop_eval_v1 has them). Add
  velocity F1 to existing harness — 1 day of work given
  `mir_eval.transcription_velocity`.
- **MV2H** — wire the Java canonical implementation into the nightly.
  ~3 days for the JNI/subprocess plumbing.
- **First Tier 5 release-gate study** — 30 listeners × 30 songs +
  3 pianists × 30 songs. ~$1.5k on Prolific, $1.5k on rater fees;
  total ~$3k. Use the rubric from §7.1.
- **Composite quality score calibration** — fit `Q` weights on the MOS
  data; publish `Q_v1.0`.
- **Public leaderboard** — publish `pop_eval_v1.0_public/`
  (the 10 FMA-licensed tracks + their transcriptions) as a CC-BY 4.0
  release on a static site; accept community model submissions.

#### Quarter 1 deliverables checklist

- [ ] `eval/pop_eval_v1.0.0/` — 30 songs, frozen, holdout encrypted
- [ ] `scripts/eval.py` — Click CLI with all subcommands
- [ ] `eval/tier{1,2,3,4}.py` — metric modules
- [ ] `eval-ci.yml` + `eval-nightly.yml` — CI workflows
- [ ] Postgres telemetry schema + Grafana dashboard
- [ ] Tier 5 release rubric + first calibration study
- [ ] Composite quality score `Q_v1.0` calibrated to MOS
- [ ] Public leaderboard + 10-song CC eval subset

---

### 10. Open problems

The strategy above is necessary but not sufficient. Outstanding gaps:

#### 10.1 Subjective taste in arrangement
There's no metric for "this is a beautiful arrangement" beyond MOS
musicality scores. Pop2Piano's authors, PiCoGen2's authors, and Etude's
authors all run their own subjective studies because no objective
metric has emerged. **Revisit when**: a music-foundation-model-based
"taste model" appears (likely 2027+); Music Flamingo + listening-MOS
correlation is one direction (NVIDIA, Nov 2025,
https://research.nvidia.com/labs/adlr/MF/).

#### 10.2 Rare time signatures + unusual meters
Oh Sheet's chord/key/meter detection only handles 3/4 and 4/4 + major/minor
(`backend/services/key_estimation.py:38-44`). Tier 2 metrics will
underweight the few 6/8 and 12/8 songs in the eval set. **Revisit when**:
modal + compound-meter detection lands; track 6/8 success in a separate
subset metric until then.

#### 10.3 Multi-language lyrics
For a future "lead sheet with lyrics" mode (per the README's vague
mention), lyrics transcription via Whisper (or Music Flamingo for
non-English) needs its own metrics: per-character edit distance, word
WER, alignment F1. Out of scope for v1; track separately when lyrics
ship.

#### 10.4 Difficulty grading correctness
Tier 3 includes a sight-readability score, but mapping from that to
Henle/ABRSM grade is empirical and unsolved at production-grade
accuracy. Use RubricNet
(https://arxiv.org/html/2509.16913) as a starting point, but expect
±1 grade error and treat the difficulty label as advisory.

#### 10.5 Long-form coherence (>4 bars)
Pop2Piano explicitly admits 4-beat context limit; the eval set is
biased toward 30-s windows (matches the existing
`max_duration=30 s` in `eval_transcription.py:88`). Long-form
arrangement quality is unmeasured. **Revisit when**: full-song
arrangement becomes a product priority — likely after Pop2Piano
replacement (Etude / AMT-APC) is in production.

#### 10.6 Genre coverage
30-50 songs cannot represent the full pop genre space. Even at 50
songs, K-pop / hip-hop / electronic / Latin / country are 10 songs
each — too small for genre-stratified analysis. Either grow the eval
set to 100+ over Y2 or stratify gates per top-3 genres only.

#### 10.7 User-feedback loop
The strategy above is offline. A future addition: collect user
in-app feedback ("rate this transcription 1-5", "this note is wrong")
and feed back into eval. Risks: feedback bias (only unhappy users
report), spam, GDPR. Out of scope for v1; track for v2.

#### 10.8 What if Tier 5 doesn't validate Tier 4?
The whole composite-Q proposal depends on `ρ(Q, MOS) ≥ 0.7`. If the
first calibration study comes back at `ρ = 0.4` (CLAP-music doesn't
help), the strategy reverts to: Tier 3 playability + Tier 2 chord +
round-trip F1 as the per-song RF triple, with Tier 5 always run
manually. Build in this fallback: don't gate releases on `Q` until
calibration confirms it.

#### 10.9 The Cogliati & Duan + musicdiff + OMR-NED gap
We listed all three as Tier 3 reference-required score-similarity
metrics but didn't pick a default. Recommendation: pin one
(musicdiff, since it has a Python package) for nightly; treat MV2H as
the primary score-similarity metric since it's better-cited; add
musicdiff for engrave-stage QA. OMR-NED requires rendering pages —
defer until pdf generation works.

#### 10.10 Holdout leakage discipline
The single hardest non-technical problem is preventing engineers from
peeking at holdout. Encryption helps mechanically, but social pressure
to "just check why this song regressed" is enormous during incident
response. Recommended posture: holdout is a release-gate-only artifact
that the on-call engineer **cannot** unlock; only the release manager
can. Document this in `RELEASE-GATES.md`.

---

### Appendix — code-level integration points (all absolute paths)

| Hook | Where to add | What goes there |
|---|---|---|
| Existing eval entry | `/Users/ross/oh-sheet/scripts/eval_transcription.py:592-602` | Replace `_run_basic_pitch_sync` direct call with a configurable per-stage runner; preserve as `transcribe` subcommand |
| New eval CLI | `/Users/ross/oh-sheet/scripts/eval.py` (new) | Click app with subcommands per §4.2 |
| Tier 2 module | `/Users/ross/oh-sheet/eval/tier2_structural.py` (new) | Audio-side anchor + comparator |
| Tier 3 module | `/Users/ross/oh-sheet/eval/tier3_arrangement.py` (new) | music21-based playability + voice-leading + density |
| Tier 4 module | `/Users/ross/oh-sheet/eval/tier4_perceptual.py` (new) | CLAP/MERT/chroma + round-trip |
| Tier 1 (existing, extend) | `/Users/ross/oh-sheet/scripts/eval_transcription.py:502-585` already does P/R/F1 + per-dim errors; add velocity F1 via `mir_eval.transcription_velocity` |
| Resynth recipe | `/Users/ross/oh-sheet/scripts/eval_transcription.py:115-131` (FluidSynth + bundled SF2) reused in Tier 4 |
| Per-stage runners (transcribe) | `/Users/ross/oh-sheet/backend/services/transcribe.py:122-216` `TranscribeService.run` |
| Per-stage runners (arrange) | `/Users/ross/oh-sheet/backend/services/arrange.py:427-517` `_arrange_sync` |
| Per-stage runners (humanize) | `/Users/ross/oh-sheet/backend/services/humanize.py:196-271` |
| Per-stage runners (engrave) | `/Users/ross/oh-sheet/backend/services/midi_render.py:36-137` + `/Users/ross/oh-sheet/backend/services/ml_engraver_client.py:107-141` |
| Run manifest | `/Users/ross/oh-sheet/eval/runs/<run-id>/manifest.yaml` (new) | git SHA, env hash, config snapshot |
| Production-quality emitter | `/Users/ross/oh-sheet/backend/jobs/runner.py:431-536` engrave block; emit `composite_quality_score` to telemetry alongside `EngravedOutput` |
| Postgres telemetry | new `/Users/ross/oh-sheet/backend/eval/telemetry.py` (new) | INSERT into `eval_production_quality_scores` after every job |
| Grafana datasource | new `/Users/ross/oh-sheet/grafana/dashboards/oh-sheet-eval.json` (new) | dashboard JSON |
| CI workflow | `/Users/ross/oh-sheet/.github/workflows/eval-ci.yml` (new) | every PR |
| Nightly workflow | `/Users/ross/oh-sheet/.github/workflows/eval-nightly.yml` (new) | cron 2 AM UTC |
| Release-gate doc | `/Users/ross/oh-sheet/RELEASE-GATES.md` (new) | pre-registered thresholds |
| Rubric | `/Users/ross/oh-sheet/eval/rubric/pianist_rubric.md` (new) | publish |

---

### Citations index

- mir_eval — Raffel 2014 https://colinraffel.com/posters/ismir2014mir_eval.pdf ; docs https://mir-eval.readthedocs.io/latest/
- MV2H — McLeod & Steedman 2018 ISMIR https://ismir2018.ismir.net/doc/pdfs/148_Paper.pdf ; code https://github.com/apmcleod/MV2H
- musicdiff — Foscarin 2019 https://inria.hal.science/hal-02267454v2/document
- OMR-NED — ISMIR 2025 https://arxiv.org/abs/2506.10488
- madmom — Böck 2016 https://arxiv.org/abs/1605.07008 ; docs https://madmom.readthedocs.io/
- MSAF — Nieto & Bello 2015 https://ccrma.stanford.edu/~urinieto/MARL/publications/NietoBello-ISMIR2015.pdf
- Davies (beat-eval) — 2014 https://archives.ismir.net/ismir2014/paper/000238.pdf
- CLAP — LAION https://github.com/LAION-AI/CLAP ; audiocraft https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/metrics/clap_consistency.html
- MERT — Li 2024 ICLR https://arxiv.org/abs/2306.00107 ; HF https://huggingface.co/m-a-p/MERT-v1-330M
- FAD — Kilgour 2019 https://www.isca-archive.org/interspeech_2019/kilgour19_interspeech.pdf ; fadtk https://github.com/microsoft/fadtk ; generative adaptation Gui 2023 https://arxiv.org/abs/2311.01616
- Simonetta perceptual resynthesis — 2022 https://link.springer.com/article/10.1007/s11042-022-12476-0
- Ycart — perceptual validity of AMT eval, TISMIR https://transactions.ismir.net/articles/10.5334/tismir.57
- Riley et al. (robust AMT) — 2024 https://arxiv.org/pdf/2402.01424
- Towards Musically Informed Evaluation — 2024 https://arxiv.org/html/2406.08454v2
- music21 — https://music21.org
- partitura — https://github.com/CPJKU/partitura
- Nakamura & Sagayama (piano reduction / playability) — https://arxiv.org/pdf/1808.05006
- Ramoneda (score difficulty) — 2023 https://arxiv.org/abs/2306.08480
- RubricNet (interpretable difficulty) — 2025 https://arxiv.org/html/2509.16913
- Pop2Piano — Choi 2022 https://arxiv.org/abs/2211.00895
- POP909 — Wang 2020 https://arxiv.org/pdf/2008.07142v1 ; repo https://github.com/music-x-lab/POP909-Dataset (MIT)
- MAESTRO — magenta.tensorflow.org/maestro-wave2midi2wave
- MMMOS — 2025 https://arxiv.org/html/2507.04094v2
- Holzapfel ethnomusicology study — 2019 https://archives.ismir.net/ismir2019/paper/000082.pdf
- Henle Difficulty Levels — https://www.henle.de/Levels-of-Difficulty/
- FMA — https://github.com/mdeff/fma

— END —

---

# Part IV — Codebase Deep-Dives

_Phase 1 reports from agents 01–03 (`feature-dev:code-explorer` subagent type, READ-ONLY). Every claim is cited with file:line; stub-vs-real status is called out throughout._

## Oh Sheet Codebase & Pipeline Architectural Analysis

### 1. Top-level layout

```
/Users/ross/oh-sheet/
├── backend/                # FastAPI app + Celery monolith workers
│   ├── api/routes/         # uploads, jobs, ws, artifacts, stages
│   ├── jobs/               # JobManager, PipelineRunner, JobEvent
│   ├── services/           # ingest, transcribe (split into ~10 sub-modules),
│   │                       # arrange, condense, transform, humanize, refine,
│   │                       # ml_engraver_client, midi_render, ...
│   ├── storage/            # thin re-export of shared/storage
│   ├── workers/            # Celery task wrappers (one per stage)
│   ├── config.py           # 645 lines, pydantic-settings (OHSHEET_*)
│   └── contracts.py        # re-exports shared/shared/contracts.py
├── shared/shared/          # contracts.py + storage/ used by all services
├── svc-decomposer/         # standalone Celery worker — STUB transcription
├── svc-assembler/          # standalone Celery worker — STUB arrangement
├── eval/fixtures/clean_midi/   # 25-MIDI subset (Lakh MIDI Clean) for eval
├── scripts/eval_transcription.py  # MIDI→synth→transcribe→mir_eval harness
├── tests/                  # pytest, all transcription mocked to stub
├── frontend/               # Flutter (web/iOS/Android)
├── docker-compose.yml      # redis + orchestrator + 6 workers
├── pyproject.toml          # backend + extras: basic-pitch, pop2piano, demucs, eval
└── eval-baseline.json      # F1=0.368 no-offset, 0.130 with-offset
```

### 2. Architecture (ASCII)

```
                               POST /v1/jobs
                                    │
                                    ▼
 FastAPI orchestrator (backend/main.py) ── JobManager (in-memory dict)
                                    │            in-process asyncio.Queue
                                    ▼            fan-out to WS subscribers
                              PipelineRunner.run()
                                    │
                          (claim-check via LocalBlobStore — file:// URIs only)
                                    │
                  ┌─ Celery (Redis broker + result backend) ─┐
                  │                                          │
        ┌─────────┼─────────┬─────────┬──────────┬───────────┤
        ▼         ▼         ▼         ▼          ▼           ▼
     ingest  transcribe  arrange  condense   humanize    refine
    (worker)  (worker)   (worker) (worker)  (worker)   (worker)
                  │                                          │
   yt-dlp WAV     │                    score_pipeline=        engrave is
   librosa probe  │                  "condense_only" replaces  inline:
                  │                  arrange with condense     PipelineRunner
   ┌──────────────┴───────────┐                                 invokes
   ▼                          ▼                              midi_render →
 Pop2Piano                  Demucs(htdemucs) → 4 stems       ml_engraver_client
 (sweetcocoa/                vocals→CREPE+BP →MELODY         → external HTTP
  pop2piano)                 bass→BP → BASS                    /engrave → MusicXML
  (single                    other→BP → CHORDS
   transformer pass)         drums→beat track
   ↓ falls back               ↓ falls back
 Demucs+BP path            Single-mix BP + Viterbi
   ↓ falls back               melody/bass split
 Single-mix BP             ↓ falls back to
                           4-note PIANO stub

 Title-lookup branch only: orchestrator may shortcut to TuneChat HTTP
 service if OHSHEET_TUNECHAT_ENABLED=true; otherwise runs full pipeline.
```

The runner serializes state to blob between every stage (`backend/jobs/runner.py:170-217`). `apply_async` is used when the task is registered locally; otherwise `send_task` with no broker fallback in eager mode (`backend/jobs/runner.py:200-217`). The whole pipeline awaits each stage in turn, so it is sequential despite the Celery facade.

### 3. Per-stage deep dive

| Stage | File | Real / Stub | Inputs → Outputs | Key lossy operations |
|---|---|---|---|---|
| **ingest** | `backend/services/ingest.py:323-413` | Real | `InputBundle` → `InputBundle` | yt-dlp converts to WAV (`outtmpl … preferredcodec: "wav"`) — re-encodes lossy YouTube AAC (line 99-101). Cover-search may **swap** the user's URL for a piano cover (line 143-265) — destructive, by design. Soundfile probe (`_probe_audio_sync`, line 276) only fills metadata, doesn't downmix. No HPSS/normalize here. |
| **transcribe** | `backend/services/transcribe.py:122-216` + `transcribe_pipeline_*.py` | Real (with stub fallback at `transcribe_result.py:249-278`) | `InputBundle.audio` (file:// only — `transcribe_audio.py:31-33`) → `TranscriptionResult` | • Audio always loaded **mono at 22050 Hz** (`transcribe_pipeline_pop2piano.py:65`, `transcribe_pipeline_single.py:74`). All stereo and >22 kHz content is destroyed before any model sees it. <br>• Pop2Piano collapses everything to a **single piano stream** by design (`transcribe_pop2piano.py:120-170`). Drums, vocals, bass all become piano notes.<br>• Demucs path collapses 4 stems into 3 BP passes; **drums are not transcribed at all** (only used for beat tracking).<br>• Basic Pitch is single-stream polyphonic (banner string at `transcribe_result.py:191-193`).<br>• Cleanup drops "ghost", "octave", "ghost-tail" notes (`transcription_cleanup.py:1-30`) — can drop legitimate octave doublings (config notes acknowledge `cleanup_chords_octave_amp_ratio: 0.5`).<br>• `cleanup_energy_gate` truncates sustains over 2.0 s (`config.py:144-147`).<br>• Stub fallback emits 4 fixed notes C/E/G/C (`transcribe_result.py:255-263`). |
| **arrange** | `backend/services/arrange.py:427-517` | Real | `TranscriptionResult` → `PianoScore` | • Hard split at MIDI 60 (`SPLIT_PITCH = 60`, `arrange.py:43`) for any non-MELODY/BASS track. Doesn't honor refined `staff_split_hint`.<br>• Drops tracks with `confidence < 0.35` (line 53, 142-146). Pop2Piano emits one track with arbitrary confidence — entire piece can vanish.<br>• `MAX_VOICES_RH = MAX_VOICES_LH = 2` (lines 51-52); excess polyphony is dropped (line 226-227 — `continue # exceeds polyphony`).<br>• Greedy quantization to 1/16 grid (default `QUANT_GRID = 0.25`), with adaptive grid in `[0.167, 0.25, 0.333, 0.5]` (`config.py:444`). No 1/32 or swung grid options.<br>• Same-pitch dedup keeps loudest within tolerance (line 178-191) — kills tremolos and trills.<br>• Velocity normalization rescales every velocity globally (line 365-390) — destroys dynamic range info from transcription.<br>• `arrange_simplify_min_velocity = 55` (`config.py:219`) drops every quiet note before engrave (`arrange_simplify.py:99`); LH bass dynamics are typically below 55. |
| **condense** | `backend/services/condense.py:131-194` | Real (used when `OHSHEET_SCORE_PIPELINE=condense_only`, the **default** — `config.py:562`) | Same I/O as arrange | Replaces arrange entirely. **Ignores `MidiTrack.instrument`** (line 5-9) — melody/bass/chord routing is thrown away; everything is hand-split at MIDI 60. **No quantization** (line 13). 16-voice cap (line 45). Cleaner / denser than arrange but has even less musical structure. |
| **transform** | `backend/services/transform.py:11-15` | **STUB** | `PianoScore` → `PianoScore` | `return score` — pure passthrough. The runner skips it entirely when `condense_only` (`shared/contracts.py:417-419`). |
| **humanize** | `backend/services/humanize.py:196-271` | Real (rule-based) | `PianoScore` → `HumanizedPerformance` | • Hard-coded per-beat-phase timing (downbeat anticipation -5 ms, backbeat push +3 ms — `humanize.py:48-58`).<br>• Velocity offsets via `math.sin(progress * pi)` per section (line 81-87) — synthetic phrase shape unrelated to audio.<br>• Pedal generation either chord-driven or per-bar fallback (line 121-155); doesn't read audio.<br>• Quality string: "Rule-based humanization (no trained model yet)" (line 269) — explicit self-flag as not real.<br>• Skipped entirely when `sheet_only` variant (`shared/contracts.py:406`).|
| **refine** | `backend/services/refine.py:63-130` | Real (LLM, opt-in via API key) | Score envelope → Score envelope (metadata-only) | LLM annotates title/composer/key/sections only — does **not** touch notes (system prompt explicitly: `Do NOT invent note data` — `refine_prompt.py:31`). Can override detected key/time-sig/tempo. Pass-through on any failure (line 107-119).|
| **engrave** | `backend/jobs/runner.py:431-536` + `backend/services/midi_render.py:36-137` + `backend/services/ml_engraver_client.py:107-141` | **External HTTP only** — no local fallback | `HumanizedPerformance` → `EngravedOutput(musicxml_uri, humanized_midi_uri)` | • Renders **only MIDI** (`midi_render.py:36-137`); the entire `ExpressionMap` (dynamics, articulations) is **dropped** — only `pedal_events` survive as CC64/66/67 (`midi_render.py:99-117`). Velocity offsets and timing offsets are baked into note timing/velocity. The remote engraver therefore receives a **plain MIDI**, with no markings beyond pedal CCs.<br>• `EngravedScoreData` reports `includes_dynamics=False, includes_pedal_marks=False, includes_fingering=False, includes_chord_symbols=False` — hard-coded `False` at runner.py:516-520.<br>• `pdf_uri=None` always (line 524) — PDF output has been retired. |
| **decomposer (svc-decomposer)** | `svc-decomposer/decomposer/tasks.py:32-75` | **STUB** | `InputBundle` → `TranscriptionResult` (4 fixed notes) | Comment line 58: `"decomposer stub — real transcription not wired yet"`. Not invoked by the live pipeline (no `decomposer.run` in `STEP_TO_TASK`). |
| **assembler (svc-assembler)** | `svc-assembler/assembler/tasks.py:29-69` | **STUB** | `TranscriptionResult` → `PianoScore` (1 RH, 1 LH note) | Stub. Likewise unused in live routing. |

#### Stub giveaways (direct quotes)

- `svc-decomposer/decomposer/tasks.py:60`: `warnings=["decomposer stub — real transcription not wired yet"]`
- `svc-assembler/assembler/tasks.py:32-34`: `def _stub_arrangement(txr: TranscriptionResult) -> PianoScore: """Tiny shape-correct fallback so downstream stages still run."""`
- `backend/services/transform.py:14-15`: `async def run(self, score: PianoScore) -> PianoScore: return score`
- `backend/services/transcribe_result.py:249-278` `_stub_result(reason)` — the ubiquitous fallback returning C/E/G/C

### 4. Data-contract walkthrough (`shared/shared/contracts.py`)

| Contract | What it carries | Conspicuously missing |
|---|---|---|
| `RemoteAudioFile` (lines 42-49) | `sample_rate`, `channels`, `duration_sec`, `format`, `content_hash` | Bit depth, codec, original SR before yt-dlp re-encode, channel layout. |
| `RemoteMidiFile` (52-56) | `ticks_per_beat` | Tracks, channels, GM programs, original tempo map. |
| `InputMetadata` (130-143) | `title`, `artist`, `source`, `prefer_clean_source`, `source_url` | Genre, BPM hint, key hint, tuning (Hz), language. |
| `Note` (157-161) | `pitch, onset_sec, offset_sec, velocity` | **Pitch bend, vibrato, micro-tuning (cents)**, original instrument, channel. Pop2Piano output `pitch_bends` is read at `transcribe_pop2piano.py:114` then **discarded** (`[]` placeholder). |
| `MidiTrack` (164-168) | `notes`, `instrument` (5-value enum), `program`, `confidence` | Per-track GM bank/family beyond the 5-value enum, expression CCs, panning, original count of pre-cleanup notes. |
| `RealtimeChordEvent` (171-176) | Harte label, root, confidence | **Inversion** (explicitly disclaimed in `chord_recognition.py:31`), bass note, slash-chord support. |
| `HarmonicAnalysis` (185-190) | `key`, `time_signature`, `tempo_map`, `chords`, `sections` | **Mode beyond major/minor** (key_estimation explicitly v1-scope in `key_estimation.py:39-41`), beat hierarchy, pickup beat, anacrusis, swing ratio. |
| `ScoreNote` (208-214) | `pitch, onset_beat, duration_beat, velocity, voice` | Tie/slur info, articulation hint, accidental override, fingering. |
| `ScoreMetadata` (250-265) | `key, time_signature, tempo_map, difficulty, sections, chord_symbols, title, composer, arranger, tempo_marking, staff_split_hint, repeats` | Section dynamics, repeat structure (only one Repeat type), DS/DC/coda, voltas-as-data, lyrics, chord voicing. |
| `ExpressiveNote` (279-291) | `timing_offset_ms` (±50), `velocity_offset` (±30), `hand`, `voice` | Per-note swing, agogic accent, micro-timing pattern (only attack-side nudge by design — line 287-289). |
| `ExpressionMap` (320-324) | dynamics, articulations, pedal_events, tempo_changes | Phrasing slurs, fingering, hairpins (only one DynamicMarking type carries span_beats), trills/turns/mordents. |
| `EngravedOutput` (348-377) | URIs, hard-coded `EngravedScoreData` (`includes_*` all False) | Page count, measure count, validation report, render warnings. |

**Critical missing channel:** there is no contract for **multi-track preservation** — Basic Pitch / Pop2Piano output collapses to a single `MidiTrack` (or 3 in the Demucs path), and once the audio→pianoroll→MIDI hop happens, the source-instrument identity is lost forever.

### 5. Pipeline variants (`shared/shared/contracts.py:393-422`)

```python
"full":         ["ingest", "transcribe", "arrange", "humanize", "engrave"],
"audio_upload": ["ingest", "transcribe", "arrange", "humanize", "engrave"],
"midi_upload":  ["ingest", "arrange", "humanize", "engrave"],          # NO transcribe
"sheet_only":   ["ingest", "transcribe", "arrange", "engrave"],        # NO humanize
```

Then `condense_only` (default! `config.py:562`) replaces `arrange` with `condense` and skips `transform`. With `enable_refine=True` (also default), `refine` is inserted before `engrave`.

So the real default plan is: `ingest → transcribe → condense → humanize → refine → engrave` for the audio/full route.

### 6. Storage / Claim-Check

`shared/shared/storage/local.py` — only filesystem support. URIs are `file://...`, validated to live under `blob_root` (`local.py:33-36`). `BlobStore` Protocol allows future S3 (`storage/base.py:12-26`) but the transcribe stage hard-codes `file://` in `transcribe_audio.py:30-33` (raises `ValueError` on anything else). All inter-stage data is JSON-serialized and round-tripped through disk twice per stage.

### 7. Lossy-format hops

| Hop | Data carried in | Data carried out | Loss |
|---|---|---|---|
| YouTube → WAV | yt-dlp opus/AAC (≤256 kb/s) | 44.1 kHz WAV stereo (`ingest.py:99-101`) | One re-encode pass, but minor. |
| WAV → librosa load | 44.1 kHz stereo | **22.05 kHz mono** (`transcribe_pipeline_pop2piano.py:65`, `transcribe_pipeline_single.py:74`) | **All stereo info, all >11 kHz content destroyed before any model.** Required by Basic Pitch but Pop2Piano was trained at 22.05 kHz too — still mono-collapsed. |
| Audio → contour matrix | (frames × 264 bins) salience, ~33 cents resolution | Discrete `NoteEvent` tuples post-Viterbi | Continuous F0 → quantized MIDI integer pitch. Pitch bends from Pop2Piano are read but stored as `[]` (`transcribe_pop2piano.py:114`). |
| `NoteEvent` → contract `Note` | (start,end,pitch,amp,bends) | (pitch,onset_sec,offset_sec,velocity) | **`bends` dropped entirely** in `_event_to_note` (`transcribe_result.py:45`). |
| `TranscriptionResult` → `PianoScore` | seconds, multi-track | **beats**, two hands | Tempo-map errors compound (`audio_timing.py:1-29`); hand split discards instrument identity for non-MELODY/BASS; quantization to 1/16 grid; voice cap at 2; velocity range globally remapped to [35,120]. |
| `PianoScore` → `HumanizedPerformance` → MIDI | beat positions + ExpressionMap | rendered seconds, sustained pedals only | `midi_render.py:99-117` only emits sustain/sostenuto/una-corda CCs. Dynamics, articulations, tempo_changes are **dropped**. |
| MIDI → MusicXML (external) | MIDI bytes | MusicXML bytes | Black-box `oh-sheet-ml-pipeline` HTTP service. Defaults to `localhost:8080` (`config.py:608`). Rejects sub-500-byte responses (`ml_engraver_client.py:93-104`); no inspection beyond size. |

### 8. Eval harness

`scripts/eval_transcription.py:1-100` synthesizes 25 ground-truth MIDIs from `eval/fixtures/clean_midi/` to WAV via fluidsynth + TimGM6mb, runs the **same** `_run_basic_pitch_sync` entrypoint (line 24-27), and scores with `mir_eval.transcription`. The aggregate baseline is **F1=0.368 no-offset, F1=0.130 with-offset** (`eval-baseline.json:362-368`). Per-role:
- `melody`: F1=**0.093** (`eval-baseline.json:373`) — terrible
- `bass`: F1=0.178
- `chords`: F1=0.341
- `piano`: 0 active files (the 5-role enum routes everything into MELODY/BASS/CHORDS)

Importantly the eval score works on **MIDI synthesized via FluidSynth on a clean piano soundfont** — i.e., it tests Basic Pitch's behavior on its easiest possible input, not on real pop music. There is no eval for Pop2Piano, no eval for arrange/humanize/engrave correctness, and one MIDI fails to synthesize at all (`eval-baseline.json:76`).

### 9. Tests touching the pipeline

`tests/conftest.py:46-68` `skip_real_transcription` autouse fixture replaces `TranscribeService.run` with a stub returning the 4-note C/E/G/C result. **Every** pipeline integration test runs against the stub — there is no end-to-end transcription test. `tests/conftest.py:81-95` `stub_ml_engraver` similarly fakes the engraver to a 100-byte MusicXML stub. Result: tests prove the pipeline orchestrates correctly but say nothing about transcription quality.

### 10. Configuration surface

`backend/config.py` is **645 lines** of Pydantic settings — over 80 transcription-related knobs, including stem-specific BP thresholds, CREPE hybrid weights, melody/bass Viterbi parameters, chord HMM transition params, key-confidence floors, beat tracker selection, adaptive grid candidates, etc. The sheer size implies extensive empirical tuning — yet the eval F1 stays at 0.368.

Notable hard-coded defaults that bias the result:
- `score_pipeline = "condense_only"` (line 562) — the default kills the melody/bass routing.
- `arrange_simplify_min_velocity = 55` (line 219) — silently drops half of LH bass on most pop material.
- `MAX_VOICES_RH = MAX_VOICES_LH = 2` (`arrange.py:51-52`) — cannot represent piano scores with > 2 voices per staff.
- `arrange.SPLIT_PITCH = 60` (`arrange.py:43`) — hand split is at middle C with no per-piece adaptation.
- `melody_low_midi=48 / high_midi=96` (`config.py:166-167`) — caps melody to C3-C7, will miss bass-vocal hooks and high lead lines.
- `cleanup_energy_gate_max_sustain_sec = 2.0` (`config.py:145`) — kills sustained whole notes longer than 2 s.

### 11. Worker boundaries

The monolith workers (`backend/workers/`) all hit `from backend.services.<X> import <X>Service` — they cannot run without the full backend codebase. The "decomposer" and "assembler" microservices (`svc-decomposer/`, `svc-assembler/`) are stubs that import only `shared/shared/contracts.py` — they were intended as a real microservice split but **the runner never dispatches to them** (`STEP_TO_TASK` at `backend/jobs/runner.py:49-57` has no `decomposer` / `assembler` entries). They are dead code in the current pipeline. The engrave stage retired its Celery worker in favor of inline HTTP (docker-compose.yml comment line 113).

### 12. TODOs / FIXMEs / "stub" comments in pipeline code

- `backend/services/transform.py:1-5`: "Transform stage — post-condense refinement (stub / passthrough). Planned home for voicing, register, or style transforms on a `PianoScore`. For now this stage returns the input unchanged so the pipeline shape is stable."
- `backend/services/humanize.py:8`: "A future revision will swap the rule-based core for a trained model."
- `svc-decomposer/decomposer/tasks.py:1-6`: "Returns a shape-correct TranscriptionResult using only the shared contracts package. When real transcription is wired up, the stub body will be replaced …"
- `svc-assembler/assembler/tasks.py:1-6`: same shape, "When real arrangement logic is wired up, the stub body will be replaced …"
- `backend/services/key_estimation.py:39-44`: "v1 scope: Key: major/minor only — no modal detection (Dorian, Mixolydian, etc.); Meter: 3/4 vs 4/4 only; The denominator is always 4."
- `backend/services/chord_recognition.py:31`: "No inversion tracking — the root we return is the template root, not necessarily the lowest note."
- `backend/services/ingest.py:36`: "YouTube URL helpers — stubs, implementation TBD"

### 13. Ranked architectural weaknesses for pop transcription

| Rank | Severity | Weakness | Evidence |
|---|---|---|---|
| 1 | **HIGH** | Audio is collapsed to **mono 22.05 kHz** before any model. Pop is fundamentally stereo (panned vocals, doubled instruments) and contains content above 11 kHz. | `transcribe_pipeline_pop2piano.py:65`, `transcribe_pipeline_single.py:74`, `transcribe_pipeline_stems.py:407` |
| 2 | **HIGH** | The default **`score_pipeline = "condense_only"`** silently throws away the melody/bass/chords role tagging produced by transcribe — every track is hand-split at middle C regardless of musical role. | `config.py:562`, `condense.py:1-9, 86-103`. The whole melody-extraction, bass-extraction, chord-recognition pipeline (700+ LOC) is computed and then discarded for hand assignment. |
| 3 | **HIGH** | `arrange_simplify_min_velocity = 55` and global velocity remap to [35,120] crush dynamic range. Pop bass and inner-voice notes from BP often sit at velocities 30-50 — every one of them is dropped before engrave. | `config.py:219`, `arrange.py:56-58`, `arrange_simplify.py:99` |
| 4 | **HIGH** | Hard cap of **2 voices per hand** drops legitimate polyphony. Pop piano covers routinely use 3-4 voices in the right hand for chord+melody. | `arrange.py:51-52, 226-227` (`continue # exceeds polyphony — drop`) |
| 5 | **HIGH** | Pop2Piano (default!) collapses everything to a **single piano stream by design**, then we run Viterbi melody/bass/chord splits with `contour=None` (`transcribe_pipeline_pop2piano.py:73-74`) — the splits silently degrade because they were designed for the BP contour matrix. Per the comment, "extractors handle this gracefully" but they actually skip back-fill (their main quality lever). |
| 6 | **HIGH** | The whole **dynamics + articulation channel is dropped at engrave**: `midi_render.py:36-137` writes only notes + pedal CCs to MIDI; the external ML engraver receives plain MIDI; `EngravedScoreData.includes_dynamics=False` is hard-coded (`runner.py:516`). | `midi_render.py:99-117`, `runner.py:514-528` |
| 7 | **MED** | Cleanup heuristics (`cleanup_octave_amp_ratio=0.6`, `cleanup_energy_gate_max_sustain_sec=2.0`) are static thresholds that **can't tell intentional octave doublings or whole-note pads from artifacts**. The codebase notes this in `config.py:139-140` ("real chord octave doublings are common"). | `transcription_cleanup.py:1-30, 53-62`, `config.py:144-147` |
| 8 | **MED** | Beat / tempo / key all detected then **hardcoded to 4/4 + major/minor only**. Pop in 6/8 (`Hey Jude`) or modal (`Riders on the Storm` — Dorian) is silently misnotated. | `key_estimation.py:38-44`, `config.py:412-413` |
| 9 | **MED** | Quantization grid is restricted to `[0.167, 0.25, 0.333, 0.5]` (`config.py:444`) — no 1/32, no swing-aware grid, no tempo-aware piecewise grid. Pop with 16th-note hi-hat patterns or shuffle feel get smeared. | `arrange.py:77-108`, `config.py:443-445` |
| 10 | **MED** | The runner runs sequentially even though Celery is set up (`runner.py:257-545` is a `for step in plan` await loop). Stages can't pipeline; a slow transcribe blocks the whole job. | `backend/jobs/runner.py:257-552` |
| 11 | **MED** | Track-level `confidence < 0.35` → silent track drop (`arrange.py:53, 142-146`). On a noisy pop track Pop2Piano can emit one track with mean amplitude < 0.35 and the entire piece vanishes. | `arrange.py:53, 142-146` |
| 12 | **MED** | The `transcription_midi` MIDI is dropped at the engrave hop. Engrave receives only the **humanized** (rule-perturbed) MIDI, not the raw transcription. The remote engraver therefore can't re-do its own quantization or use raw onsets. | `runner.py:501-512` |
| 13 | **MED** | Humanization is **rule-based with magic numbers** (downbeat -5 ms, backbeat +3 ms, sin-curve phrase shape). The result is "humanized" away from what the audio actually does. For a transcription product (where the goal is fidelity), humanization is anti-correlated with quality. | `humanize.py:34-94, 269` |
| 14 | **MED** | The eval harness scores transcribe-only on **fluidsynth-rendered piano MIDI**, not on real pop audio. The reported F1=0.368 is on the easiest possible inputs; real-pop F1 is unknown and unmeasured. | `scripts/eval_transcription.py:1-100`, `eval-baseline.json:362` |
| 15 | **LOW** | LocalBlobStore only — `transcribe_audio.py:30-33` raises on non-`file://` URIs. S3 / GCS deployments need code change. | `transcribe_audio.py:24-33` |
| 16 | **LOW** | Stage timeouts default to 600 s (`config.py:34`) — Pop2Piano on long audio can blow this and the failure surface is "task timeout" with no partial result. | `config.py:34`, `runner.py:215` |
| 17 | **LOW** | The decomposer / assembler microservices are stub-only dead code. Any future contributor reading the README will think they do real work. | `svc-decomposer/decomposer/tasks.py:32-75`, `svc-assembler/assembler/tasks.py:29-69` |

### 300-word summary: most damaging architectural weaknesses

**1. Audio is mono-22.05-kHz before any model touches it.** Every transcription path (Pop2Piano, Demucs+BP, single-mix BP) calls `librosa.load(..., sr=22050, mono=True)` (`transcribe_pipeline_pop2piano.py:65`, `transcribe_pipeline_single.py:74`). All stereo information and content above 11 kHz are destroyed upstream of inference.

**2. The default `score_pipeline="condense_only"` (`config.py:562`) discards the role-tagging the transcribe stage works hard to produce.** `condense.py:1-9, 86-103` ignores `MidiTrack.instrument` and hand-splits everything at MIDI 60. The 700-LOC melody/bass/chord pipeline is computed then thrown away.

**3. Two-voice cap + velocity threshold gut piano arrangements.** `arrange.py:51-52` caps `MAX_VOICES_RH=MAX_VOICES_LH=2`, dropping excess polyphony (`arrange.py:226-227`). `arrange_simplify_min_velocity=55` (`config.py:219`) removes every quiet note (`arrange_simplify.py:99`); pop LH bass routinely lives below 55. Velocity is then globally remapped to [35,120] (`arrange.py:365-390`), erasing dynamics from transcription.

**4. Engrave is a black-box HTTP call that receives plain MIDI.** `midi_render.py:36-137` writes only note onsets + pedal CCs to MIDI; the entire `ExpressionMap` (dynamics, articulations, tempo_changes) is dropped. `runner.py:516-520` hard-codes `includes_dynamics=False, includes_pedal_marks=False, includes_chord_symbols=False` regardless of what was computed upstream.

**5. Pop2Piano (default) collapses everything to one piano stream by design** (`transcribe_pop2piano.py:120-170`), then post-processing tries to split it without a contour matrix (`transcribe_pipeline_pop2piano.py:73-74`).

**6. Eval is on fluidsynth-synthesized clean piano MIDI**, not real pop. The reported F1=0.368 (`eval-baseline.json:364`) over-reports real-world quality and the per-role melody F1 is **0.093** (`eval-baseline.json:373`).

**7. Key/meter detection is forced to major/minor and 3/4 vs 4/4** (`key_estimation.py:38-44`); pop in 6/8 or modal music is silently misnotated.

---

## Oh Sheet Transcribe Stage Deep-Dive

### 1. Module / Pipeline Topology

**Celery task:** `backend/workers/transcribe.py:12` — wraps `TranscribeService.run` in `asyncio.run` (line 20).

**Service entry:** `backend/services/transcribe.py:122` — `TranscribeService.run` resolves audio URI, dispatches to `_run_basic_pitch_sync` in a thread (line 178).

**Three-way dispatcher (priority order):** `backend/services/transcribe.py:52-119`:

1. **Pop2Piano path** (`pop2piano_enabled=True` by default at `config.py:244`) — `transcribe_pipeline_pop2piano.py`
2. **Demucs+Basic Pitch path** (`demucs_enabled=True` by default at `config.py:293`) — `transcribe_pipeline_stems.py`
3. **Single-mix Basic Pitch path** (fallback) — `transcribe_pipeline_single.py`

So unless dependencies fail, **Pop2Piano is actually the active path in production**, not Basic Pitch. Basic Pitch only runs when Pop2Piano fails (`transcribe.py:84-96`).

### 2. Basic Pitch Wiring

**Inference call:** `backend/services/transcribe_inference.py:123-129`:

```python
model_output, midi_data, note_events = predict(
    str(inference_path),
    model_or_model_path=model,
    onset_threshold=onset_threshold if onset_threshold is not None else settings.basic_pitch_onset_threshold,
    frame_threshold=frame_threshold if frame_threshold is not None else settings.basic_pitch_frame_threshold,
    minimum_note_length=settings.basic_pitch_minimum_note_length_ms,
)
```

**Model load:** `backend/services/transcribe_inference.py:39-58`. Uses `ICASSP_2022_MODEL_PATH` (Spotify's bundled 2022 model). Cached process-wide with double-checked locking. Backend auto-picks (CoreML/ONNX/TFLite) via `basic_pitch.inference.Model`.

**Version pin:** `pyproject.toml:46-47, 83` — `basic-pitch>=0.4`. The Makefile (`Makefile:78-83`) installs with `--no-deps` to bypass the `tensorflow-macos` hard-pin issue on Darwin/Python 3.13.

### 3. Parameter Table — Exactly What's Passed to `predict()`

| Parameter | Value passed | BP default | Source |
|---|---|---|---|
| `onset_threshold` | `0.5` (global) | `0.5` | `config.py:45` |
| `frame_threshold` | `0.3` (global) | `0.3` | `config.py:46` |
| `minimum_note_length` | `127.7` ms | `127.7` ms | `config.py:47` |
| `minimum_frequency` | **NOT PASSED** (None) | None (no floor) | — |
| `maximum_frequency` | **NOT PASSED** (None) | None (no ceiling) | — |
| `multiple_pitch_bends` | **NOT PASSED** (False) | False | — |
| `melodia_trick` | **NOT PASSED** (True) | True | — |
| `midi_tempo` | **NOT PASSED** (120) | 120 | — |
| `debug_file` | **NOT PASSED** | None | — |

**Per-stem overrides** (only when Demucs path runs, `transcribe_pipeline_stems.py:185-203`):
- vocals: `onset=0.6`, `frame=0.25` (`config.py:70-71`)
- bass: `onset=0.5`, `frame=0.3` (`config.py:77-78`)
- other: `onset=0.5`, `frame=0.3` (`config.py:82-83`)

These mirror Spotify upstream `DEFAULT_*` values exactly. **No frequency band restrictions are configured**; pitch-bends and melodia-trick are at upstream defaults.

### 4. Audio Preprocessing Chain

Optional, `audio_preprocess_enabled=False` by default (`config.py:108`). When enabled, `backend/services/audio_preprocess.py`:

1. `librosa.load(..., sr=None, mono=True)` — preserves native SR, forces mono (`audio_preprocess.py:298`).
2. **Min-duration gate**: skip if < 0.25s (`audio_preprocess.py:69, 190-193`).
3. **HPSS harmonic extraction** (`hpss_margin=1.0`) via `librosa.effects.harmonic` (`audio_preprocess.py:204`). This intentionally removes percussive transients — drums.
4. **RMS normalize to -20 dBFS, peak ceiling -1 dBFS** (`audio_preprocess.py:217-227`). No look-ahead limiter; computes a single gain value with a peak guard.
5. Writes 32-bit float WAV tempfile at native SR (`audio_preprocess.py:326`).

**Basic Pitch's own internal preprocessing** then resamples to 22050 Hz mono (per `basic_pitch.constants.AUDIO_SAMPLE_RATE = 22050`, `FFT_HOP = 256`, `ANNOTATIONS_FPS ≈ 86.13`). The pipeline does not control this.

In the single-mix path, audio is also reloaded once at sr=22050 mono for post-processing (`transcribe_pipeline_single.py:73-74`). No length cap, no segmenting, no ffmpeg invocation in transcribe stage itself.

### 5. Post-Processing Chain — Single-Mix Path

In order, on `note_events` returned by Basic Pitch:

1. **Cleanup phase** (`transcription_cleanup.py:435-514`, called from `transcribe_inference.py:144-164`):
   - Pass 1: Merge fragmented sustains (gap ≤ 0.03s) — `cleanup_merge_gap_sec` (`config.py:119`)
   - Pass 2: Octave-ghost prune (`amp_ratio=0.6`, `onset_tol=0.05s`) — `config.py:120-121`
   - Pass 3: Ghost-tail prune (< 0.05s AND < 0.5×median amp) — `config.py:122-123`
   - Pass 5: Energy gating using RMS envelope (max sustain 2.0s, floor 0.1) — `config.py:144-147`. Pass 4 (overlap trim) is "placeholder" per docstring at `transcription_cleanup.py:460-461`.

2. **Viterbi melody/bass split** over `model_output["contour"]` (264-bin × 86 Hz salience matrix) — `transcribe_pipeline_single.py:108-144`:
   - melody band: MIDI 48–96 (C3–C7), `voicing_floor=0.15`, `transition_weight=0.25` (`config.py:166-171`)
   - bass band: MIDI 28–55 (E1–G3), `voicing_floor=0.12`, `transition_weight=0.40` (`config.py:187-192`)
   - Uses `BASE_MIDI=21` (A0), 3 bins/semitone (`melody_extraction.py:53-56`)

3. **Onset refinement** — `librosa.onset.onset_strength` at hop=256 (~11.6 ms), max shift 50 ms (`onset_refine.py`, `config.py:154-156`).

4. **Duration refinement** — per-pitch CQT energy decay (`duration_refine.py`, `config.py:457-461`).

5. **Tempo map** — madmom `RNNBeatProcessor + DBNBeatTrackingProcessor`, fallback librosa (`audio_timing.py:99-148`, `config.py:438`).

6. **Key + meter** — Krumhansl-Schmuckler chroma-CQT correlation, plus 3/4-vs-4/4 onset periodicity (`key_estimation.py`).

7. **Chord recognition** — chroma + 24 triad templates, optional HMM smoothing, optional 7th templates (`chord_recognition.py`, `config.py:199-205`).

8. **Result assembly** (`transcribe_result.py:70-246`): velocity = `int(round(127 * amplitude))` clamped to [1,127] (`transcribe_result.py:46-47`). Per-track `confidence = clamp(mean(amplitudes), 0.1, 1.0)`. Tracks ordered MELODY, BASS, CHORDS, PIANO.

### 6. Stub-Fallback Path

`backend/services/transcribe_result.py:249-278` — `_stub_result(reason)`. Returns 4 hardcoded notes (C4, E4, G4, C5), each 0.5s @ vel=80, MELODY role, "C:major" 4/4, 120 BPM, confidence 0.3. Triggered by:
- `payload.audio is None` (`transcribe.py:142-143`)
- Audio fetch failure / staged file missing (`transcribe.py:160-176`)
- `ImportError` (deps unavailable) (`transcribe.py:181-183`)
- Any other inference exception (`transcribe.py:184-186`)

The `skip_real_transcription` test fixture (`tests/conftest.py:46-68`) monkey-patches `TranscribeService.run` itself to return the stub plus a fake `b"MThd..."` MIDI header.

### 7. Output → Arrange Wiring

The Pydantic `TranscriptionResult` (`shared/shared/contracts.py`) carries `midi_tracks: list[MidiTrack]`, `analysis: HarmonicAnalysis(key, time_signature, tempo_map, chords, sections=[])`, and `quality: QualitySignal`. Notes carry only `pitch, onset_sec, offset_sec, velocity` — **no pitch-bend data, no per-frame confidence, no instrument program, no pan/effects**. Pitch bends from BP are dropped at result assembly (`transcribe_midi.py:50-54`). Sections are always `[]`.

### 8. Auxiliary Signal Extraction

| Signal | Source | File |
|---|---|---|
| Beats / tempo | madmom DBN → librosa fallback | `audio_timing.py:99-148` |
| Key | KS profile vs chroma_cqt | `key_estimation.py:60-75` |
| Time signature | Beat-onset periodicity (3/4 vs 4/4 only) | `key_estimation.py:38-44` |
| Chords | chroma + 24 triad templates + optional HMM | `chord_recognition.py` |
| Sections | **Not implemented** | — |
| Downbeats | **Not extracted explicitly** | — |
| Lyrics | **Not extracted** | — |

### 9. Tests

- `tests/test_transcribe_stems.py` — 15 tests; cover dispatcher, parallel/serial stems, fallback chains, CREPE hybrid fusion.
- `tests/test_transcribe_pop2piano.py` — 9 tests; PrettyMIDI conversion, dispatcher routing, fallback on errors.
- `tests/test_transcription_cleanup.py` — covers merge/octave/ghost/end-to-end + defaults match config.
- `tests/test_audio_preprocess.py`, `test_onset_refine.py`, `test_duration_refine.py`, `test_melody_extraction.py`, `test_bass_extraction.py`, etc.

**NOT covered**: real Basic Pitch inference on real audio (every test forces stub via `skip_real_transcription` fixture in `conftest.py:46-68`). No coverage of: drum-heavy material, vocal-only material, sub-bass < 32 Hz, polyphony > N notes, or > 5 minute clips.

### 10. Resource Handling

- **GPU/CPU**: Basic Pitch model auto-picked on first load (`transcribe_inference.py:53-57`), CoreML on Darwin, ONNX on Linux. Backend selection is left to upstream auto-pick order via `ICASSP_2022_MODEL_PATH`.
- **Warm-up**: Lazy-loaded, cached process-wide (`transcribe_inference.py:35-58`). Costs ~1s per process.
- **Timeouts**: `job_timeout_sec: int = 600` (10 min) in `config.py:34`. No per-stage timeout. Pop2Piano timeout would be inherited from Celery worker.
- **Concurrency**: Stem passes parallelized via ThreadPoolExecutor (`transcribe_pipeline_stems.py:230-243`); model claimed thread-safe.

### 11. Comparison: Basic Pitch Capability vs. What Pop Music Needs

| Need | Basic Pitch built-in | Codebase config | Gap |
|---|---|---|---|
| Drum rejection | None — will detect every spectral onset | HPSS (off by default `config.py:108`) + ghost-tail prune | **Severe**: drums get transcribed as pitched notes |
| Vocal handling | None — vocals tracked as polyphonic pitches | CREPE on Demucs vocals stem (Demucs path only) | OK *only when* Demucs path active |
| Sub-bass < 32 Hz | None — model trained on A0 (27.5 Hz) up; weak on bass | No `minimum_frequency` set | Bass below ~30 Hz lost |
| Velocity dynamics | Direct from amplitude | `vel = round(127*amp)` linear (`transcribe_result.py:46`) | Linear amplitude→velocity ignores perceptual loudness |
| Polyphony cap | Implicit — model emits multi-stream notes but tops out around ~6 simultaneous | None; cleanup may drop more | Dense pop polyphony lost |
| Pitch bends | `multiple_pitch_bends=False` upstream default | Not configured; pitch bends dropped at MIDI rebuild (`transcribe_midi.py:50-54`) | Bent notes flattened to nearest semitone |
| Tempo / beat | None — BP doesn't estimate | madmom + librosa fallback present | OK |
| Key / time signature | None | KS profile + 3/4 vs 4/4 only | 6/8, modal music misclassified |
| Single global threshold | Yes — one onset, one frame threshold | Single global pair (`0.5/0.3`) per pass | **No per-frequency-band thresholds**; mid-range bias |
| Frame resolution | hop=256 → ~86 fps (~11.6 ms) | onset_refine adds ~11.6 ms grid | Fast 16th notes at 180+ bpm marginal |

### 12. Ranked Gaps (severity)

#### HIGH severity
1. **`minimum_frequency` / `maximum_frequency` never set** (`transcribe_inference.py:123-129`). Defaults are `None` → no frequency floor. Bass guitar fundamentals below ~30 Hz are weak in Basic Pitch's mel features regardless, but more importantly, sub-bass rumble + drum kick is allowed to register as "notes". Setting a floor (e.g. 32 Hz on full mix, with per-stem differentiation) would suppress kick-drum-as-note artifacts on non-Demucs path.
2. **Single global `onset_threshold=0.5` / `frame_threshold=0.3`** for the full mix in the single-mix path (`config.py:45-46`, `transcribe_inference.py:126-127`). Basic Pitch is amplitude-sensitive across all bands — pop mixes with loud drums + quiet melody force a compromise. **Per-frequency-band thresholding does not exist** in the codebase.
3. **Pop2Piano is the actual default path** (`config.py:244`), but Basic Pitch is still the second-tier fallback. If Pop2Piano fails on common pop material it falls through to a Basic Pitch path that has none of the pop-specific tuning Pop2Piano provides.
4. **No source separation upstream of Basic Pitch in single-mix path**. `demucs_enabled=True` defaults on, but if the demucs extra is not installed, single-mix runs the polyphonic tracker directly on the full mix, which transcribes drums + vocals + bass as piano notes.
5. **Velocity collapse on stub fallback** (`transcribe_result.py:259-262`) — every stub note is vel=80. Even on real path, `velocity = round(127 * amplitude)` is a linear mapping with no perceptual loudness curve.
6. **Vocals transcribed as pitched notes on the single-mix and Pop2Piano paths**. Only the Demucs path routes vocals through CREPE (`transcribe_pipeline_stems.py:128-153`). On single-mix, every legato vocal phrase becomes a noisy run of polyphonic pitch detections.

#### MEDIUM severity
7. **`audio_preprocess_enabled=False` default** (`config.py:108`). HPSS would help reject drums on the single-mix path, but it's off because of marginal eval gains on the dev fixture (3 takes of "Rising Sun", non-pop). For pop, the cost-benefit reverses.
8. **Pitch bends are silently dropped** on MIDI rebuild (`transcribe_midi.py:50-54`, "Per-note pitch bends are intentionally dropped"). `multiple_pitch_bends` is never enabled. Bent guitar/vocal notes quantize to the nearest semitone.
9. **Tempo map clamps BPM to [30, 300]** (`audio_timing.py:43-44`); fine-grained micro-timing wobble is preserved but the map has no swing/groove model.
10. **Time signature detector only picks 3/4 or 4/4** (`key_estimation.py:38-44`). 6/8 ballads, 7/8, 12/8 shuffles all become 4/4.
11. **Key estimator is major/minor only** (`key_estimation.py:39-41`). Modal pop (Dorian, Mixolydian) → wrong accidentals in PDF.
12. **Stub fallback emits hardcoded C-E-G-C** (`transcribe_result.py:255-264`). Any silent failure produces a C-major arpeggio, masking real pipeline issues from end-users.
13. **No length cap or segmentation** for very long audio. Cleanup passes are O(n) but Viterbi over `contour` is O(frames × bins) — a 10-min track is 51600 × 264 ≈ 13.6M cells.
14. **`midi_tempo` parameter never passed to `predict()`** — Basic Pitch's own MIDI tempo defaults to 120 (`basic-pitch.predict default`). The codebase rebuilds the blob MIDI with the audio-derived tempo (`transcribe_midi.py:74`), but the contract `tempo_map` and the BP-internal MIDI disagree until rebuild.

#### LOW severity
15. **No real-Basic-Pitch coverage in tests** — `skip_real_transcription` fixture is `autouse=True` (`conftest.py:46`). Configuration-level regressions in BP wiring are invisible in CI.
16. **`melodia_trick` defaults to True** (BP upstream) — fine, but never tested off; some material may benefit from disabling.
17. **`overall_confidence` floor of 0.1** (`transcribe_result.py:236-237`) hides total failures from downstream consumers.
18. **Drum stem is used only for beat tracking** (`transcribe_pipeline_stems.py:486-491`); never explicitly excluded as a pitched-note source on single-mix path.

### 250-Word Summary

The transcribe stage is a three-way dispatcher (`backend/services/transcribe.py:52-119`): Pop2Piano → Demucs+BP → single-mix BP. Basic Pitch's `predict()` is called at exactly one call site, `backend/services/transcribe_inference.py:123-129`, with only three of the nine available parameters set (`onset_threshold=0.5`, `frame_threshold=0.3`, `minimum_note_length=127.7ms`). Critical pop-music knobs are left at upstream defaults: `minimum_frequency=None`, `maximum_frequency=None`, `multiple_pitch_bends=False`, `melodia_trick=True`, `midi_tempo=120` (`config.py:45-47`).

The biggest config-side problems for pop music: (1) **No `minimum_frequency`** floor — kick drum and sub-bass rumble register as notes (`transcribe_inference.py:123`). (2) **Single global `0.5/0.3` thresholds** across all frequency bands on the full mix — drum-heavy mixes force a compromise that loses quiet melody (`config.py:45-46`). (3) **Audio preprocessing (HPSS) defaults off** because dev fixtures aren't pop (`config.py:108`); without it the polyphonic tracker treats drums as pitched onsets. (4) **Vocals are transcribed as polyphonic pitches** on every path except Demucs+CREPE (`transcribe_pipeline_stems.py:128-153`); single-mix pop with vocals → noisy melody.

Post-processing-side problems: **pitch bends are intentionally dropped** at MIDI rebuild (`transcribe_midi.py:50-54`); **velocity is a linear `round(127*amp)`** with no perceptual curve (`transcribe_result.py:46-47`); **time signature detector only picks 3/4 or 4/4** (`key_estimation.py:38-44`); **key estimator is major/minor only**; **stub fallback emits a hardcoded C-major arpeggio** (`transcribe_result.py:255-264`); and **no test covers real Basic Pitch inference** (`tests/conftest.py:46-68` monkey-patches it out via the auto-use `skip_real_transcription` fixture).

---

## Oh Sheet — Arrange / Humanize / Engrave Deep Dive

### Pipeline at a Glance

The 5-stage plan (`shared/shared/contracts.py:400-422`) for an audio job is:
`ingest → transcribe → arrange → humanize → refine → engrave`

`refine` is inserted before `engrave` only if `enable_refine=True` (default). For `title_lookup` jobs, the entire local pipeline can be short-circuited and TuneChat is used instead (`backend/jobs/runner.py:303-358`).

---

### Stage 1: ARRANGE

**Worker**: `backend/workers/arrange.py` (Celery task on the `arrange` queue, dispatched by `PipelineRunner`).
**Service**: `backend/services/arrange.py` (`_arrange_sync`, ~580 lines, real implementation).
**Post-pass**: `backend/services/arrange_simplify.py` (5-step density reducer).

#### What it actually does (real, not stub)

1. **Hand assignment** (`_assign_hands`, `arrange.py:135-156`): `MELODY` → RH, `BASS` → LH, everything else split by **`pitch >= 60` (middle C)**. No hand-position / reachability analysis.
2. **Beat-domain conversion** via `sec_to_beat` and `tempo_map`.
3. **Adaptive grid estimation** (`_estimate_best_grid`, lines 78-108): tries candidates `0.167, 0.25, 0.333, 0.5` (sixteenth/eighth/triplet) and picks lowest-residual.
4. **Quantize + dedup + voice-assign** (`_resolve_overlaps`, lines 163-229): same-pitch dedup keeps loudest; greedy voice fill; **MAX_VOICES_RH = MAX_VOICES_LH = 2** (lines 51-52).
5. **Beat-snap** (`_beat_snap`, lines 261-358): ±1 grid step shift if it improves alignment.
6. **Velocity normalize** (`_normalize_velocity`, lines 365-390): linear remap to [35, 120] mean ~75.
7. **Chord/section conversion**: copies `RealtimeChordEvent` → `ScoreChordEvent` (no chord generation here).
8. **Simplify pass** (`arrange_simplify.py`): velocity < 55 dropped, durations snapped to {0.25, 0.5, 1.0, 2.0, 4.0}, micro-notes dropped, chord-cluster merging within 1/32 beat, density cap of 4 distinct onsets/beat.

#### Real-vs-stub verdict
- **Local rules backend**: real but simplistic.
- **`hf_midi_identity` backend** (`arrange.py:520-533` and `hf_arrange/inference.py:21-23`):
  ```python
  if inference_mode == "identity":
      return midi_in
  ```
  **Stub identity transform** — no model.
- **Alternative `svc-assembler`** (`svc-assembler/assembler/tasks.py:29-54`): hardcoded 2-note score:
  ```python
  def _stub_arrangement(txr: TranscriptionResult) -> PianoScore:
      """Tiny shape-correct fallback so downstream stages still run."""
      ...
      right_hand=[ScoreNote(id="rh-0001", pitch=60, ...)]
      left_hand=[ScoreNote(id="lh-0001", pitch=48, ...)]
  ```
  Kept around as a placeholder for a future remote service.

#### Key failure modes for pop readability
- **Naive middle-C split** for any non-melody/non-bass material (`arrange.py:155`). Catastrophic for tenor-range vocal lines or guitar/keyboard parts that span both staves.
- **No hand-position / reachability analysis** — a single voice may contain a 3-octave span the LH cannot physically play.
- **No block-chord inference** beyond `arrange_simplify`'s 0.125-beat onset clustering.
- **Voice-cap = 2 per hand**: anything denser is silently dropped (`_resolve_overlaps` line 227 `continue # exceeds polyphony`).
- **Velocity-normalization erases dynamics** before humanize ever sees them: original loud/soft contour is squashed to fit [35, 120], so subsequent dynamics inference becomes meaningless.
- **`arrange_simplify` defaults are aggressive**: `min_velocity=55` plus `max_onsets_per_beat=4` means the LH (where Basic Pitch tends to produce quieter notes) frequently goes nearly empty (compare `config.py:215-222` tuning history note about "LH untouched because bass notes from Basic Pitch rarely dip below 40").
- **Condense path** (`backend/services/condense.py:42-103`) uses identical middle-C split with no quantization at all.

#### Score pipeline mode
Default is `condense_only` (`config.py:562`), which **swaps out arrange entirely** for condense → transform (passthrough). Transform is a literal stub:
```python
## backend/services/transform.py:14-15
async def run(self, score: PianoScore) -> PianoScore:
    return score
```

---

### Stage 2: HUMANIZE

**Worker**: `backend/workers/humanize.py`.
**Service**: `backend/services/humanize.py` (real but rule-based).

#### What it does
- **Timing offsets** (`_humanize_timing`, lines 44-59): downbeat → -5 ms, backbeat → +3 ms, plus Gaussian noise. Stored as `timing_offset_ms` per ExpressiveNote.
- **Velocity offsets** (`_humanize_velocity`, lines 66-94): downbeat accent +5, off-beat -2, plus a section-wise `sin(progress·π)` shape (lines 81-88).
- **Dynamics** (`_infer_dynamics`, lines 101-114): one DynamicMarking per `ScoreSection` based on average velocity. **If `sections=[]` (which is the default — see audit below), no dynamics are produced.**
- **Pedal** (`_generate_pedal`, lines 121-155): one PedalEvent per chord change if chord_symbols present, else **one bar-length event per measure**. So pedal is always populated.
- **Articulations** (`_detect_articulations`, lines 162-189): staccato when duration < 40% of next-note gap, legato when >= 95%, accent when off-beat AND velocity > 100.
- Returns `HumanizedPerformance` with `quality.warnings=["Rule-based humanization (no trained model yet)"]` (`humanize.py:269`).

#### Real-vs-stub verdict
**Real**, fully implemented rule-based heuristics. Self-described as a placeholder ("A future revision will swap the rule-based core for a trained model" — `humanize.py:8`).

#### Key failure modes
- **Dynamics depend on sections existing**, but transcribe never produces any sections (`transcribe_result.py:188` `sections=[]`); only `RefineService` (Anthropic Claude) can fill them. Without an `OHSHEET_ANTHROPIC_API_KEY`, **dynamics list is always empty**.
- **Pedal fallback is dumb**: per-measure full-bar events ignore actual harmonic rhythm.
- **Articulations are brittle**: staccato/legato thresholds use neighbor-gap comparisons that misbehave on dense piano output.
- **No swing / rubato / agogic accents / phrase shape beyond crude sine-modulation.**
- **No grace notes, ornaments, fermatas (the type exists), or trills.**
- **No tests exist for humanize** (no `test_humanize.py` in the suite), so behavior changes are not regression-protected.

---

### Stage 3: ENGRAVE

This is the largest gap between README and reality.

#### What the README claims
README at `README.md:127` lists `engrave.py # music21 → MusicXML, LilyPond → PDF`, line 248 lists `POST /v1/stages/engrave`, line 303 says "Engraving: music21, LilyPond, pretty_midi".

#### What actually exists
**No `engrave.py` service. No engrave Celery worker. No `/v1/stages/engrave` endpoint. No music21 import in the entire backend. No LilyPond invocation. Engrave is dispatched _inline_ from `PipelineRunner`** (`backend/jobs/runner.py:431-536`):

1. Hydrate a `HumanizedPerformance`.
2. `render_midi_bytes(perf)` (`backend/services/midi_render.py`).
3. POST those MIDI bytes to an **external** `oh-sheet-ml-pipeline` HTTP service (`backend/services/ml_engraver_client.py:107-173`) at `OHSHEET_ENGRAVER_SERVICE_URL` (default `http://localhost:8080`).
4. Receive MusicXML bytes, save to blob, **set `pdf_uri=None`** (`runner.py:524`).

```python
## runner.py:514-528
result_dict = EngravedOutput(
    schema_version=SCHEMA_VERSION,
    metadata=EngravedScoreData(
        includes_dynamics=False,
        includes_pedal_marks=False,
        includes_fingering=False,
        includes_chord_symbols=False,
        title=resolved_title,
        composer=resolved_composer,
    ),
    pdf_uri=None,
    musicxml_uri=musicxml_uri,
    humanized_midi_uri=midi_uri,
    audio_preview_uri=None,
).model_dump(mode="json")
```

The `includes_*=False` constants are explicit acknowledgement that no expression data is in the output.

#### What `render_midi_bytes` puts in the MIDI request
(`backend/services/midi_render.py:36-136`)
- Initial tempo + first time-signature change.
- Note on/off (with `timing_offset_ms` applied as **onset-only** nudge).
- Pedal CCs (CC64/66/67 — sustain/sostenuto/una corda) from `expression.pedal_events`.
- Same-pitch overlap clipping; min duration filter (30 ms).

#### What `render_midi_bytes` DROPS / DOES NOT SEND
- Key signature (no `KeySignature` events, even though `metadata.key` is known)
- Voice number (`ScoreNote.voice` 1 vs 2)
- Hand assignment beyond what's implicit in pitch
- All dynamics (`expression.dynamics`)
- All articulations (`expression.articulations`) — staccato/accent/etc never reach the engraver
- All chord symbols (`metadata.chord_symbols`) — `RealtimeChordEvent` data captured by `chord_recognition.py` is silently discarded
- Sections / phrase boundaries / repeats
- Title, composer, tempo marking text
- Difficulty, staff_split_hint
- Lyrics (no field exists, expected for pop)
- Tempo changes / accel / rit (`expression.tempo_changes` is also always empty — `humanize.py:254`)

#### Failure modes from the engrave choice
- **PDF artifact is `None`** for every audio_upload / midi_upload job. The frontend tries `/v1/artifacts/{job}/pdf` and falls back to MusicXML rendered client-side via OSMD/VexFlow.
- **OSMD rendering of an unstructured MusicXML** produced from a barebones MIDI gives the typical "raw transcription" look: every note its own beat, no proper voicing, no key signature, no dynamics, no chord symbols. The "publication-quality engraving" claim does not match the artifact shape.
- **No local fallback**: an outage of the external service hard-fails the job (`test_ml_engraver_error_propagation.py`).
- **`_looks_like_stub`** (`ml_engraver_client.py:103-104`) treats responses < 500 bytes as stub and raises — meaning if the upstream ML service is itself a stub returning skeleton MusicXML, jobs fail.

#### TuneChat alternative (parallel/replacement engraving)
For `title_lookup` jobs only (`runner.py:303-358`), the entire pipeline is delegated to TuneChat which uses tcalgo + MuseScore. The Oh Sheet team's own comment (`runner.py:299-302`):
> "TuneChat uses tcalgo + MuseScore which produces much cleaner scores than Basic Pitch + music21."
Acknowledges that the local engrave path is the lower-quality option even from the team's own perspective.

---

### PianoScore field-by-field audit

`PianoScore` (`shared/shared/contracts.py:268-272`) and `ScoreMetadata` (`contracts.py:250-265`):

| Field | Filled by | Reaches engraver? |
|---|---|---|
| `right_hand[].pitch/onset/duration/velocity/voice` | arrange | partial — no voice in MIDI |
| `left_hand[].*` | arrange | partial |
| `metadata.key` | transcribe (`key_estimation`) | NO — discarded by `midi_render` |
| `metadata.time_signature` | transcribe / pretty_midi parse | YES (one event) |
| `metadata.tempo_map` | transcribe (audio beat tracking) | partial — initial BPM only |
| `metadata.difficulty` | hardcoded `"intermediate"` (`runner.py` -> `arrange_sync` default) | NO |
| `metadata.sections` | **always `[]` in transcribe** (`transcribe_result.py:188`); only refine fills it | NO |
| `metadata.chord_symbols` | `chord_recognition.py` (real) | NO — silently dropped |
| `metadata.title` | refine LLM only | YES (sent in `EngravedScoreData`, **not in MusicXML/MIDI**) |
| `metadata.composer` | refine LLM only | YES (in `EngravedScoreData` only) |
| `metadata.arranger` | refine LLM only | NO |
| `metadata.tempo_marking` | refine LLM only | NO |
| `metadata.staff_split_hint` | refine LLM only | NO — arrange has already split |
| `metadata.repeats` | refine LLM only | NO |

`ExpressionMap` (`contracts.py:320-324`):
| Field | Filled by | Reaches engraver? |
|---|---|---|
| `dynamics` | humanize, **only if sections exist** | NO |
| `articulations` | humanize | NO |
| `pedal_events` | humanize (chord-driven or per-bar fallback) | YES (CC64/66/67) |
| `tempo_changes` | always `[]` (`humanize.py:254`) | NO |

`EngravedScoreData` (`contracts.py:339-346`): all four "includes_*" flags are hardcoded `False` (`runner.py:517-520`).

---

### Engraving library audit

| Library | Imported anywhere in `backend/services/`? | Used? |
|---|---|---|
| `music21` | NO | NO |
| `abjad` | NO | NO |
| LilyPond binary | NO | NO |
| `verovio` | NO | NO |
| `pretty_midi` | YES (`midi_render.py`, `pretty_midi_tracks.py`, `transcribe_*`) | only for MIDI write, not score-rendering |
| OSMD (`opensheetmusicdisplay`) | only in Flutter `frontend/lib/widgets/sheet_music_viewer_web.dart` | client-side rendering |
| Tone.js | only frontend | playback |

**Conclusion**: there is no symbolic-score engraving in the backend. Score appearance is entirely a function of (a) what the external `oh-sheet-ml-pipeline` ML service returns, and (b) how OSMD renders that MusicXML in the browser. Headers like `drawTitle: false, drawComposer: false` (per the OSMD config) mean the title metadata returned in `EngravedScoreData` isn't even displayed.

---

### Test coverage gaps

| Stage | Tests | What's not tested |
|---|---|---|
| arrange | `test_arrange.py`, `test_arrange_simplify.py`, `test_hf_arrange.py` | hand-position playability, voice leading, anything resembling pop-readability |
| humanize | **none** | everything |
| midi_render | `test_midi_render.py` | only error paths + happy-path bytes signature |
| engrave (inline) | `test_ml_engraver_*.py`, `test_refine_engrave_metadata.py` | no end-to-end visual / structural assertion on MusicXML; ML engraver is always faked with `<score-partwise version="3.1"><part id="P1"/></score-partwise>` |
| Output PDF | none | no PDF is ever produced locally |

Critical: no test inspects the MusicXML payload structure. The fake stubs every test uses (`conftest.py:24-27` `_FAKE_ML_MUSICXML`) is itself **`< 500 bytes`** so `_looks_like_stub` would reject it in production — but in tests the stub `engrave_midi_via_ml_service` is monkeypatched to bypass that check.

---

### Ranked failure modes for pop sheet readability/playability

1. **(Engrave) The ML engraver receives only MIDI** — every piece of expression, structure, and metadata produced upstream (dynamics, chord symbols, articulations, sections, key signature, tempo marking, voice numbers, repeats, title) is discarded by `render_midi_bytes`. The ML service has to re-derive structure from notes alone. This is the single largest quality loss.
2. **(Engrave) The "engrave stage" is an out-of-process black box.** Quality is bounded by whatever `oh-sheet-ml-pipeline` returns — there is no guarantee of voice grouping, bar correctness, or stem direction. `pdf_uri` is always None, so what users see is OSMD's best effort on whatever XML comes back.
3. **(Arrange) Naive middle-C hand split** for non-melody/non-bass material with no consideration of hand reachability, voice leading, or playability.
4. **(Arrange) `condense_only` is the default `score_pipeline`** (`config.py:562`), which means most jobs use `condense.py` (no quantization, no dedup, just chronologically-merged middle-C split) followed by a no-op transform — i.e. the rules-based arrange path documented above is *not even running by default*.
5. **(Engrave/Pipeline) Chord symbols are detected but never written.** `chord_recognition.py` produces real Harte-notation labels, `arrange.py` propagates them as `ScoreChordEvent`, but `midi_render.py` doesn't include them in MIDI and the engraver has no other channel.
6. **(Humanize/Pipeline) Dynamics depend on sections, sections depend on RefineService LLM.** Without `OHSHEET_ANTHROPIC_API_KEY`, the `dynamics` list is empty in 100% of jobs and `_infer_dynamics` produces nothing.
7. **(Arrange) Velocity squash to [35,120] flattens dynamic contour** before humanize sees it (`_normalize_velocity` lines 365-390).
8. **(Arrange) Voice cap of 2 per hand silently drops polyphony**; `arrange_simplify` aggressively trims notes (default `min_velocity=55`) which can decimate the LH on quietly-transcribed bass.
9. **(Engrave) No key signature in MIDI** even though `metadata.key` is detected and refined. Engraver must guess the key from accidentals.
10. **(Humanize) Pedal fallback is per-measure full-bar** when no chord symbols are present — produces unmusical pedaling on pop tunes with chords-per-half-note rhythm.
11. **(All) Test coverage on humanize is zero**; tests on engrave only exercise transport-layer error paths, not score quality.
12. **(Pipeline) Tempo changes are never produced** (`humanize.py:254` `tempo_changes=[]`). Even fermatas (Articulation type exists) are not emitted in tempo_changes.
13. **(Arrange) No phrase / measure / section structure inference** in any local stage — comes only from refine, which costs an LLM call and isn't guaranteed.

### 250-Word Summary

The biggest quality losses in Oh Sheet's arrange-to-engrave pipeline come from a single architectural decision: the "engrave" stage, despite the README's claim of music21/LilyPond, is a thin HTTP wrapper that POSTs **only MIDI bytes** (`backend/services/midi_render.py`) to an external `oh-sheet-ml-pipeline` service. Every piece of structural and expressive data the upstream stages compute — dynamics, articulations, chord symbols, sections, key signature, voice numbers, repeats, tempo marking, title, composer — is discarded before the engraver sees it. The `EngravedOutput` itself hardcodes `includes_dynamics=False, includes_pedal_marks=False, includes_chord_symbols=False, includes_fingering=False` (`runner.py:517-520`), and the local PDF artifact is always `None`. Frontend rendering is OSMD/VexFlow on whatever bare MusicXML the remote model produces.

Second, the default `score_pipeline=condense_only` (`config.py:562`) bypasses the rules-based arranger in favor of `condense.py`, which is a chronological track-merge with a naive `pitch >= 60` middle-C hand split, no quantization, and a passthrough `transform.py` stub. Even when `arrange.py` does run, it splits by middle-C for any non-melody/non-bass track and caps voices at 2 per hand.

Third, humanize's dynamics inference requires `ScoreSection`s that transcribe never produces (`sections=[]` is hardcoded), so unless the optional Anthropic-backed `RefineService` runs, dynamics are always empty. The result: a clean technical pipeline whose final artifact is a structureless MIDI-derived MusicXML rendered in a browser.

---

# Part V — Technology Research

_Phase 1 reports from agents 04–10 (`general-purpose` subagent type with WebSearch/WebFetch). Every numerical claim and technology recommendation has an inline URL._

## Automatic Music Transcription (AMT) State-of-the-Art Survey
### For Oh Sheet's Pop-to-Piano Pipeline | April 2026

---

### 0. Scope and Reading Guide

This report surveys polyphonic AMT models with explicit attention to **transcribing full-mix pop songs into piano-playable representations**. Oh Sheet currently uses Spotify's Basic Pitch — a CNN model that is well below 20 MB on disk and **fewer than 17,000 parameters** (note: the prompt's "~17M params" figure is incorrect; Basic Pitch is much smaller). It is excellent on monophonic / single-instrument signals but degrades on dense pop mixes with vocals, drums, mastering, and codec artifacts.

A note on the published numbers below: **most MAESTRO note-F1 scores are author-reported** on a fixed v3.0.0 test split; reproducibility is generally good for the well-cited models but not independently audited model-by-model. Where third-party reproductions or fairness concerns exist, they are flagged explicitly.

---

### 1. Comparison Table (≥10 models)

The "Note F1" column reports MAESTRO v3.0.0 *Note onset* F1 unless otherwise noted; "Note+Off" is Note-with-offset F1 (the harder metric). "Slakh F1" is Slakh2100 multi-instrument F1 (token-level, "Onset+Offset+Program Flat" or comparable). All numbers are author-reported.

| # | Model | Year | Params | Train data | MAESTRO Note F1 | MAESTRO Note+Off F1 | Slakh Multi F1 | Pop suitability | License | OSS repo | Inference latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Onsets & Frames (Hawthorne) | 2018 | ~26M (large), ~10M (small) | MAESTRO | 94.80% (onset) | ~78% | n/a (piano only) | Low — piano solo only | Apache-2.0 | [magenta/magenta](https://github.com/magenta/magenta) (also [jongwook/onsets-and-frames](https://github.com/jongwook/onsets-and-frames)) | ~0.5–1× real-time on CPU |
| 2 | Kong "High-Resolution" (ByteDance) | 2021 | ~20M | MAESTRO v2 | **96.72%** | ~84% | n/a (piano + pedals) | Low–Med — strong on piano stems after separation | Apache-2.0 / MIT | [bytedance/piano_transcription](https://github.com/bytedance/piano_transcription); pip [piano_transcription_inference](https://pypi.org/project/piano-transcription-inference/) | 0.5–2× RT CPU; ~10× RT GPU |
| 3 | Sequence-to-Sequence (Hawthorne 2021) | 2021 | T5-small ~54M | MAESTRO | ~96% | ~83% | n/a | Low — piano solo | Apache-2.0 | [magenta/mt3](https://github.com/magenta/mt3) | Slower (autoregressive) |
| 4 | MT3 (Gardner) | 2021/22 | T5-small ~54M | Slakh, Cerberus, MAESTRO, MusicNet, GuitarSet, URMP | ~94% (multi-track) | – | 0.50 (Onset+Off+Prog Flat) | Med — multi-instrument but leaky on pop | Apache-2.0 | [magenta/mt3](https://github.com/magenta/mt3) | Slow (autoregressive) |
| 5 | Basic Pitch (Spotify) | 2022 | <17K params (~20 MB peak RAM) | GuitarSet, iKala, MAESTRO, MedleyDBPitch, Slakh | ~88% (note, MAESTRO) | – | – | Low–Med — generic, weak on dense polyphony | Apache-2.0 | [spotify/basic-pitch](https://github.com/spotify/basic-pitch) | **>10× RT on CPU** |
| 6 | Pop2Piano (Choi) | 2022 | T5-based ~50M | 300h scraped (Pop, Piano-cover) pairs | n/a (cover, not transcription) | n/a | n/a | High for *cover generation*, not faithful transcription | Apache-2.0-ish | [sweetcocoa/pop2piano](https://github.com/sweetcocoa/pop2piano), [HF docs](https://huggingface.co/docs/transformers/en/model_doc/pop2piano) | Seconds per song |
| 7 | Sheet Sage (Donahue) | 2022 | Uses Jukebox feats (5B) | HookTheory + Jukebox priors | n/a (lead sheet, +20% rel. melody F1 vs spectro) | n/a | n/a | High for **lead sheet from pop audio** | MIT (code); models CC BY-NC-SA 3.0 | [chrisdonahue/sheetsage](https://github.com/chrisdonahue/sheetsage) | 12 GB GPU; minutes/song |
| 8 | hFT-Transformer (Toyama, Sony) | 2023 | **5.5M** | MAESTRO v3 | **97.44%** (with half-stride) | **90.53%** | n/a (piano only) | Med after stem separation | MIT | [sony/hFT-Transformer](https://github.com/sony/hFT-Transformer) | Fast — small model; sub-RT on GPU |
| 9 | Jointist (Cheuk) | 2023 | – | Slakh-pop blends | – | – | "+1 ppt on pop vs MT3" | Med – instrument-aware multitrack | – | [paper](https://arxiv.org/abs/2302.00286) | – |
| 10 | DiffRoll (Cheuk, Sony) | 2022/23 | UNet-diffusion | MAESTRO + unpaired | competitive | – | n/a | Low–Med — research stage | MIT-ish | [sony/DiffRoll](https://github.com/sony/DiffRoll) | Slow (diffusion) |
| 11 | MR-MT3 (Tan) | 2024 | ≈MT3 | Slakh | – | – | **+2.4 ppt over MT3** (67.3 Flat), instrument leakage 1.65→1.05 | Med — better leakage handling | – | [gudgud96/MR-MT3](https://github.com/gudgud96/MR-MT3) | Same as MT3 |
| 12 | YourMT3+ / YPTF.MoE (Chang) | 2024 | MT3 + <2.5% (PerceiverTF + MoE) | 10 datasets including MIR-ST500, ENST-Drums, MAESTRO, Slakh, MusicNet, URMP, EGMD, GuitarSet, CMedia, SMT-Bass | **96.52%** (MAESTRO v3) | – | **64–75%** (Slakh Multi) | **High — best multi-instrument incl. vocals & drums in one model** | GPL-3.0 (mimbres) | [mimbres/YourMT3](https://github.com/mimbres/YourMT3) | Med — encoder heavier |
| 13 | PerceiverTF | 2023 | Perceiver-based | Slakh + others | – | – | competitive | Med | – | [paper](https://arxiv.org/abs/2306.10785) | – |
| 14 | Mobile-AMT | 2024 | MBConv + RNN; mobile-sized | MAESTRO + augmentations | -82.9% compute vs SOTA, +14.3 F1 on noisy real audio | – | n/a | Med — explicit pop-noise hardening | – | [openreview](https://openreview.net/forum?id=1QTsNlmlDk) | **Real-time on mobile** |
| 15 | Streaming Piano Transcription (Watanabe 2025 ISMIR) | 2025 | small | MAESTRO | comparable to offline SOTA | – | n/a | Med | – | [paper](https://arxiv.org/abs/2503.01362) | **Streaming/real-time** |
| 16 | PiCoGen / PiCoGen2 (Chen) | 2024 | uses SheetSage encoder + symbolic decoder | piano-only pretrain + weakly-aligned pop pairs | n/a (cover) | – | – | High for *piano cover from pop*, not faithful | – | [PiCoGen project](https://tanchihpin0517.github.io/PiCoGen/) | Seconds/song |
| 17 | Etude (Chen 2025/26) | 2025 | three-stage Extract + Beat + Decode | – | n/a (cover) | – | – | High for human-like piano covers | – | [paper](https://arxiv.org/abs/2509.16522) | – |
| 18 | Omnizart (CITI) | 2021 | toolbox of CNNs | per-task | – | – | piano + drums + chord + vocal melody (66.59% MusicNet, 74% ENST drums) | Med — multi-task batch | MIT | [Omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) | Med |
| 19 | Noise-to-Notes (N2N) | 2025 | diffusion + foundation feats | drums | n/a | n/a | n/a | High for *drums only* | – | [paper](https://arxiv.org/abs/2509.21739) | Slow (diffusion) |
| 20 | ADTOF | 2021–24 | CRNN | Crowdsourced drum charts | n/a (drums) | – | – | High — best open-source drum AMT | – | [paper](https://www.researchgate.net/publication/356491421_ADTOF_A_large_dataset_of_non-synthetic_music_for_automatic_drum_transcription) | Real-time |

(Sources for this table are linked inline and consolidated at the end.)

---

### 2. Detailed write-ups of the top candidates

#### 2.1 hFT-Transformer (Sony, ISMIR 2023) — current SOTA on MAESTRO

**Claim source:** Toyama et al., "Automatic Piano Transcription with Hierarchical Frequency-Time Transformer," [arXiv:2307.04305](https://arxiv.org/abs/2307.04305), [ISMIR 2023 paper](https://archives.ismir.net/ismir2023/paper/000024.pdf), [github.com/sony/hFT-Transformer](https://github.com/sony/hFT-Transformer).

**Architecture.** Two hierarchies: (1) per-time-frame, a CNN block on the time axis, a Transformer encoder over frequency, and a Transformer decoder that maps to a fixed pitch dimension; (2) a Transformer encoder over time. Total ~5.5M parameters — *more than 4× smaller* than Onsets & Frames.

**Performance (author-reported).**
- MAESTRO v3.0.0 Note F1: **97.43–97.44%**
- Note-with-offset F1: **90.32–90.53%**
- Note-with-offset-and-velocity F1: **89.25–89.48%**
- MAPS Note F1: 85.14%, Note+Off F1: 66.34%

These outperform Kong's High-Resolution model (96.72% onset) and Onsets & Frames (~94.80%).

**Independence note:** Numbers are author-reported on an established public split. Multiple follow-ups ([Streaming PT 2025](https://arxiv.org/abs/2503.01362), [arXiv:2509.09318](https://arxiv.org/html/2509.09318)) treat hFT as the de-facto piano-transcription baseline, which is implicit confirmation.

**License:** MIT, PyTorch implementation, training on RTX 2080 Ti / A100; pretrained MAESTRO checkpoints downloadable.

**Pop-fitness verdict.** Excellent **once the piano stem (or piano-only post-separation) is isolated** — but the model has only ever seen MAESTRO (acoustic grand) so it will degrade on synth pads, electric pianos, and full mixes. **Combine with Demucs, not as a drop-in for full pop audio.**

---

#### 2.2 YourMT3+ / YPTF.MoE+Multi (Chang et al., MLSP 2024) — best general-purpose multi-instrument

**Claim source:** Chang et al., "YourMT3+: Multi-instrument Music Transcription with Enhanced Transformer Architectures and Cross-dataset Stem Augmentation," [arXiv:2407.04822](https://arxiv.org/abs/2407.04822), [MLSP 2024 PDF](http://eecs.qmul.ac.uk/~simond/pub/2024/ChangEtAl-MLSP-2024.pdf), [github.com/mimbres/YourMT3](https://github.com/mimbres/YourMT3).

**Architecture.** MT3-style T5 encoder/decoder, but the encoder is replaced with **PerceiverTF (Lu, ISMIR 2023, [arXiv:2306.10785](https://arxiv.org/abs/2306.10785))** — a hierarchical time-frequency Perceiver with spectral cross-attention — and the FFN is upgraded to a **mixture-of-experts** layer. Cross-dataset stem augmentation mixes paired stems across heterogeneous datasets (e.g., adding ENST-Drums onto MAESTRO piano + MIR-ST500 vocals).

**Training data:** MusicNet-EM, GuitarSet, MIR-ST500 (vocals), ENST-Drums, Slakh, EGMD, MAESTRO, CMedia, URMP, SMT-Bass — ten datasets including **vocals and drums explicitly**.

**Performance (author-reported, with pitch-shifting):**
- MAESTRO v3 Note F1: **96.52%** (vs. MT3 ~94%)
- Slakh Multi F1: **74.84%** (vs. MT3 ~62%)
- MusicNet Strings EM: **90.07%**
- URMP Multi F1: **67.98%**

Parameter increase over MT3: under 2.5%. So you get cross-instrument coverage with marginal model growth.

**Independence note:** These are author-reported, post-augmentation, with their best variant. The 2025 NeurIPS AMT Challenge ([results paper](https://openreview.net/pdf?id=NG187AZ71W)) reported that "two teams outperformed MT3 baseline" using MT3-derived architectures with MoE / hierarchical-attention extensions — broadly consistent.

**License:** Repository is **GPL-3.0** — this is the biggest production gotcha. GPL-3.0 will likely contaminate Oh Sheet's commercial license unless you either (a) keep the YourMT3 model behind a microservice boundary that you self-host, or (b) reach out to authors for relicensing, or (c) reimplement based on the paper. Apache-2.0 base components (PerceiverTF, MT3) are still cleanly available.

**Pop-fitness verdict.** **Best single-model candidate** for full-mix pop transcription that respects multiple instruments. Vocals, drums, bass, piano, guitar are all covered. The HuggingFace Spaces demo (Aug 2024) lets you A/B test before commitment.

---

#### 2.3 Kong High-Resolution Piano Transcription (ByteDance, 2021) — best piano-only production option

**Claim source:** Kong et al., "High-Resolution Piano Transcription With Pedals by Regressing Onset and Offset Times," [arXiv:2010.01815](https://arxiv.org/abs/2010.01815), [TASLP 2021](https://dl.acm.org/doi/abs/10.1109/TASLP.2021.3121991), [github.com/bytedance/piano_transcription](https://github.com/bytedance/piano_transcription).

**Architecture.** CNN+BiGRU stack, two regression heads predicting precise onset/offset times relative to the closest frame, plus a parallel pedal head. ~20M parameters.

**Performance.** Onset F1 96.72% on MAESTRO, pedal-onset F1 91.86% (first pedal benchmark). Independent third-party evaluations (e.g., the 2024 noise-injection robustness study, [arXiv:2410.14122](https://arxiv.org/abs/2410.14122)) confirm the 96.7% baseline figure on clean MAESTRO.

**Production properties.** Pip-installable wheel as `piano_transcription_inference`; uses PyTorch; one of the most battle-tested piano AMT systems in third-party tools (e.g., [pianotrans GUI](https://github.com/azuwis/pianotrans)). Apache-2.0.

**Pop-fitness verdict.** Like hFT, **only piano-trained**. Best used as the piano stem head in a separation+transcribe pipeline. Better real-world track record than hFT because it has been deployed widely; hFT is newer with stronger paper numbers.

---

#### 2.4 MT3 + descendants (Magenta 2021, MR-MT3 2024) — multi-instrument token decoders

**Claim source:** Gardner et al., "MT3: Multi-Task Multitrack Music Transcription," [arXiv:2111.03017](https://arxiv.org/abs/2111.03017), [openreview](https://openreview.net/forum?id=iMSjopcOn0p), [github.com/magenta/mt3](https://github.com/magenta/mt3); Tan et al., "MR-MT3," [arXiv:2403.10024](https://arxiv.org/abs/2403.10024), [github.com/gudgud96/MR-MT3](https://github.com/gudgud96/MR-MT3).

**Architecture.** Encoder-decoder T5-small (~54M) with a custom MIDI-event vocabulary (time, program, note-on, note-off, velocity tokens). Spectrogram in, MIDI events out, autoregressively. Trained on a buffet of single- and multi-instrument datasets. MR-MT3 adds memory-retention across windows to mitigate "instrument leakage" — the well-known MT3 failure where notes get assigned to wrong programs across boundaries.

**Performance (author-reported, Slakh2100 Onset+Offset+Program "Flat" F1):**
- MT3: 0.48 (paper) / 0.5039 (community reproduction)
- MR-MT3: **0.673** (from-scratch) / **0.730** (continual training)
- Instrument-leakage ratio φ improves from 1.65 → 1.05 (MR-MT3)

**License:** Apache-2.0 for MT3 itself; MR-MT3 inherits.

**Pop-fitness verdict.** MT3 was famously fragile on pop because of instrument leakage and confusion between similar timbres. MR-MT3 is the clearest direct fix. YourMT3+ is the more aggressive successor with both architectural and data fixes.

---

#### 2.5 Onsets & Frames (Hawthorne 2018) — historical but still common

**Claim source:** Hawthorne et al., [arXiv:1710.11153](https://arxiv.org/abs/1710.11153), [Magenta blog](https://magenta.tensorflow.org/onsets-frames).

**Architecture.** CNN + BiLSTM with two heads (onset, frame); inference gates the frame head on onsets. ~26M params.

**Performance.** Note onset F1 ~94.80% on MAESTRO. Solid for the era; surpassed by Kong (2021) and hFT (2023).

**Pop-fitness verdict.** **Don't pick it for pop.** Better than nothing for piano stems, but Kong and hFT are strict upgrades. Still common as a baseline because of its CC code, ease of training, and clear architecture.

---

#### 2.6 Pop2Piano + PiCoGen2 + Etude — *cover generation* (not faithful transcription)

These are not transcribers; they are **piano-arrangement generators conditioned on pop audio**. Critical distinction:

- **Pop2Piano** ([arXiv:2211.00895](https://arxiv.org/abs/2211.00895), [HF docs](https://huggingface.co/docs/transformers/en/model_doc/pop2piano)) is a T5 encoder-decoder trained on ~300h of YouTube-scraped {pop audio, piano cover} pairs. Outputs MIDI piano cover; no melody/chord intermediate. Trained mostly on K-Pop, generalizes to other pop genres.
- **PiCoGen2** ([arXiv:2408.01551](https://arxiv.org/abs/2408.01551), [project page](https://tanchihpin0517.github.io/PiCoGen/)) freezes a Sheet Sage encoder, then decodes piano via transfer-learning. Two-stage (melody first, then accompaniment).
- **Etude** ([arXiv:2509.16522](https://arxiv.org/abs/2509.16522)) — three-stage extract → beat-structure → decode. Subjective listening tests claim near-human composer quality.

**Pop-fitness verdict for Oh Sheet:** if your goal is "the user uploads a pop song and we make playable piano sheets that *sound like a piano cover*" rather than "we faithfully transcribe every instrument as piano notes," **a Pop2Piano-style model is arguably the most direct path** and can replace the Transcribe + Arrange stages entirely. Tradeoff: less faithful to the original audio, more "interpretive."

---

#### 2.7 Sheet Sage (Donahue, 2022) — pop-aware lead-sheet transcription

**Claim source:** Donahue et al., "Melody Transcription via Generative Pre-Training," [arXiv:2212.01884](https://arxiv.org/abs/2212.01884), [github.com/chrisdonahue/sheetsage](https://github.com/chrisdonahue/sheetsage).

**Architecture.** Borrows internal representations from OpenAI's Jukebox model (a 5B-parameter generative music model) as features for melody transcription, on top of madmom for beat/downbeat and a custom chord head. **Improves melody transcription by 20% relative** vs. spectrogram features.

**Tradeoffs.** Requires Docker, ~12 GB GPU for Jukebox features, **CC BY-NC-SA 3.0** on the trained models — incompatible with a commercial product unless you retrain from your own data on TheoryTab (which has its own licensing). **Code is MIT but the weights are not freely commercializable.**

**Pop-fitness verdict.** Best public option for **lead-sheet** style output (melody + chords + downbeats), which is closer to what most pop piano sheet music looks like in commercial products. License is the blocker; consider it as inspiration rather than a drop-in.

---

### 3. Hybrid stack candidates

The realistic answer for a production pop-to-piano pipeline is rarely a single model. Below are concrete stacks ranked by my expected quality/effort tradeoff for Oh Sheet.

#### Stack A — Demucs → hFT-Transformer (piano stem) + ADTOF (drums)
1. **Demucs v4 / htdemucs_ft** ([github](https://github.com/facebookresearch/demucs)) splits vocals / drums / bass / other. Optional `htdemucs_6s` adds a piano stem (low quality per Meta's own README).
2. Drop the drum stem (or send to ADTOF for percussion shape).
3. Run **hFT-Transformer** ([sony/hFT-Transformer, MIT](https://github.com/sony/hFT-Transformer)) on the *bass+other* mix to get pitched MIDI; remap to a single piano program.
4. Run **ADTOF / Noise-to-Notes** ([N2N paper](https://arxiv.org/abs/2509.21739)) on the drum stem if you need rhythmic guide notes.
5. Feed all into the existing Arrange stage.

Pros: All Apache/MIT-style permissive licenses; hFT is small (~5.5M params); known SOTA on piano. Cons: hFT was *only* trained on acoustic grand; aggressive pop processing on the "other" stem will surprise it. Mitigate with [Towards Robust Transcription noise-injection augmentation, arXiv:2410.14122](https://arxiv.org/abs/2410.14122) by retraining/finetuning on Demucs-stems + white-noise.

#### Stack B — Demucs → Kong (ByteDance) for piano + Basic Pitch for melodic "other"
Replace hFT in Stack A with the ByteDance Kong system (more battle-tested in production tools), and keep Basic Pitch for less common timbres (synth, brass) where Kong's piano-only bias hurts. Same drum head.

Pros: Both Apache-2.0, both proven; trivially Python-pip integratable. Cons: Marginally lower piano accuracy than hFT; still no instrument identification in the pitched stage.

#### Stack C — Demucs → YourMT3+ end-to-end multi-track
Skip the per-stem head: feed the raw mix (or Demucs-cleaned mix) directly into **YourMT3+ / YPTF.MoE+Multi** ([github GPL-3.0](https://github.com/mimbres/YourMT3)). It outputs MIDI for all instruments (vocals included as monophonic notes) in one shot. Then apply Oh Sheet's Arrange stage to fold everything onto two piano staves.

Pros: One model, vocals + drums + bass + piano + guitar in one forward pass; SOTA among multi-instrument transcribers. Cons: **GPL-3.0 license blocker**; heavier model; autoregressive decoding is slower.

#### Stack D — Sheet Sage / Pop2Piano direct (skip transcribe entirely)
If user-perceived "this sounds like a real piano cover" matters more than faithful transcription, route audio straight through **Pop2Piano** ([HF transformers](https://huggingface.co/docs/transformers/en/model_doc/pop2piano)) or **PiCoGen2** ([arXiv:2408.01551](https://arxiv.org/abs/2408.01551)). Output MIDI is already piano-styled; the Arrange stage can then focus on engraving readability rather than rearrangement.

Pros: One model; best subjective quality for pop covers per Etude's listening tests; outputs a *piano-idiomatic* MIDI rather than a literal transcription that might be unplayable. Cons: Less faithful, harder to QA, harder to give the user fine-grained control ("transpose this part," "drop bass octave").

#### Stack E — Demucs → Sheet Sage (lead sheet) → symbolic piano arranger
Use Sheet Sage to produce melody + chords + downbeats, then a symbolic arranger to render piano. This matches commercial products like Klangio's Melody Scanner. **Blocked on Sheet Sage's NC-SA license unless retrained.**

#### My ranking for Oh Sheet specifically
1. **Stack B** (Demucs + Kong + Basic Pitch + ADTOF) — ship this in week one, all permissive licenses, all pip-installable.
2. **Stack A** (Demucs + hFT) — ship next, gives a paper-claimed +1 ppt onset / +6 ppt note-with-offset over Kong on piano benchmarks.
3. **Stack D** (Pop2Piano direct) — ship as a "piano cover mode" alongside the transcription mode; users will love it for casual use.
4. **Stack C** (YourMT3+) — only after the GPL-3.0 question is resolved with the authors.

---

### 4. Pop-specific priorities — what the literature says

| Concern | Best public answer | Source |
|---|---|---|
| Pitch under drums/vocals | Demucs-separate first; YourMT3+ also robust because it was trained with cross-dataset stem augmentation | [Demucs](https://github.com/facebookresearch/demucs); [YourMT3+](https://arxiv.org/abs/2407.04822) |
| Bass-line preservation | Demucs has a dedicated bass stem; YourMT3+ trained on SMT-Bass dataset | [SMT-Bass via YourMT3+](https://arxiv.org/abs/2407.04822) |
| Dense polyphony | hFT-Transformer's hierarchical attention; PerceiverTF latents | [hFT](https://arxiv.org/abs/2307.04305); [PerceiverTF](https://arxiv.org/abs/2306.10785) |
| Vocal-as-pitched-instrument | YourMT3+ with MIR-ST500 vocal training; Sheet Sage for melody only | [YourMT3+](https://arxiv.org/abs/2407.04822); [Sheet Sage](https://arxiv.org/abs/2212.01884) |
| MP3 codec / mastering robustness | Mobile-AMT's augmentation scheme (+14.3 F1 on real-world); Onsets-and-Frames noise-injection retraining | [Mobile-AMT EUSIPCO 2024](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000036.pdf); [Towards Robust Transcription](https://arxiv.org/abs/2410.14122) |

**Practical takeaway on robustness:** The 2024 study ([arXiv:2410.14122](https://arxiv.org/abs/2410.14122)) showed Kong's model drops ~5% F1 at 12 dB SNR and ~10% at 9 dB; pop mixes after Demucs separation typically sit at 6–15 dB SNR for the residual "other" stem. **Plan to fine-tune any chosen piano transcriber on Demucs-stems-with-noise-augmentation** if you want resilience.

---

### 5. Specific recommendations to replace Basic Pitch in Oh Sheet

#### Recommendation R1 — Short-term (1–2 weeks): Demucs + ByteDance Kong
- Add Demucs v4 (`htdemucs_ft`) as a new ingest stage.
- Replace Basic Pitch in `transcribe` with `piano_transcription_inference` (ByteDance Kong) on the (bass + other) downmix.
- Keep Basic Pitch as a fallback path for non-pop inputs (MIDI uploads bypass everything).
- All Apache-2.0; trivial Python integration; no GPU required for Kong (CPU-tolerable).

#### Recommendation R2 — Medium-term (4–6 weeks): swap Kong for hFT-Transformer
- After R1 is in production, build a parallel branch using `sony/hFT-Transformer` (MIT) with their pretrained MAESTRO checkpoint.
- A/B against Kong on a curated set of 30 pop songs; pick the winner per-song or default to hFT for the +6 ppt note-with-offset gain.
- Optional: fine-tune hFT on Demucs-output-piano-stem pairs synthesized from MAESTRO+pop-mastering augmentation.

#### Recommendation R3 — Parallel "Cover Mode" (3–4 weeks): Pop2Piano direct
- Add a new pipeline variant `pop_cover` that bypasses Transcribe/Arrange and routes Demucs-cleaned audio (or raw audio) into Pop2Piano via HuggingFace transformers.
- Surface this as a UI toggle: "Faithful transcription" vs. "Piano cover."
- Pop2Piano output is piano-idiomatic MIDI which already plays cleanly on two staves — Engrave stage gets simpler.

#### Recommendation R4 — Long-term (quarter+): YourMT3+ multi-instrument
- Resolve license question with the authors (request Apache/MIT relicense, or build behind a hosted-microservice boundary that cleanly isolates GPL).
- Wire into the pipeline as the unified Transcribe stage; drop separate piano/drum heads.
- Yields biggest accuracy gain on full pop mixes (+12 ppt Slakh F1 vs MT3) and removes the brittle Demucs→single-instrument-head chain.

---

### 6. What would change in Oh Sheet for each integration

| Recommendation | Changes to `backend/workers/transcribe` | Changes to pipeline contracts | New deps | Disk / RAM | License risk |
|---|---|---|---|---|---|
| R1 (Kong + Demucs) | New `_run_demucs()` pre-step; replace `basic_pitch.predict` call with `piano_transcription_inference.PianoTranscription` | Add `audio_stems: dict[str, BlobURI]` to `TranscriptionResult` (claim-checked) | `demucs`, `piano_transcription_inference` | ~2 GB Demucs model + ~300 MB Kong | Apache-2.0 across the board |
| R2 (hFT) | Swap Kong for hFT inference loader; small refactor of post-processing (hFT outputs note tuples already) | None | `torch`, hFT repo as submodule | ~70 MB checkpoint | MIT |
| R3 (Pop2Piano) | New pipeline variant `pop_cover`; new worker `cover.py` | New `PianoCoverResult` contract; `Variant` enum gets `pop_cover` | `transformers`, `pretty_midi`, `essentia` | T5-small ~250 MB | Apache-2.0 (model card) |
| R4 (YourMT3+) | Replace per-stem heads with single YourMT3+ pass; drop Demucs from required path | `TranscriptionResult` gains `instrument_tracks: list[NoteTrack]` | YourMT3+ repo, PerceiverTF | ~500 MB checkpoint | **GPL-3.0 — must resolve** |

---

### 7. Honesty about confidence in numbers

- All MAESTRO note-F1 figures above are **author-reported on the v3.0.0 test split** unless a third-party reproduction is cited. Onsets-and-Frames (94.80% onset), Kong (96.72% onset), and hFT (97.44% note F1) are all commonly cited and treated as trustworthy by the field; the 2024 robustness paper ([arXiv:2410.14122](https://arxiv.org/abs/2410.14122)) independently re-confirmed Kong at 96.7% F1 on clean MAESTRO.
- YourMT3+ scores (96.52% MAESTRO, 74.84% Slakh) are author-reported in [arXiv:2407.04822](https://arxiv.org/abs/2407.04822) and have **not** been independently audited; one signal of credibility is that the [2025 NeurIPS AMT Challenge results paper](https://openreview.net/pdf?id=NG187AZ71W) reported related YPTF-style architectures beating MT3.
- Slakh "Flat" F1 numbers from MT3 (0.48) versus the community reproduction (0.5039) — see [Magenta discuss thread](https://groups.google.com/a/tensorflow.org/g/magenta-discuss/c/nDXF4VrMWvs) — illustrate that even with a fixed protocol there is run-to-run variance of a few percentage points.
- Basic Pitch's "MAESTRO note F1 ~88%" is *my estimate* from the ICASSP 2022 paper combined with the deep-wiki / [grokipedia](https://grokipedia.com/page/Basic_Pitch) summary; the paper itself does not publish a single headline MAESTRO note-F1 because it positions itself as instrument-agnostic and reports per-dataset metrics. **Treat as approximate.**

---

### 8. Reference index

#### Papers
- Hawthorne et al. 2018 — Onsets and Frames. [arXiv:1710.11153](https://arxiv.org/abs/1710.11153)
- Hawthorne et al. 2021 — Sequence-to-Sequence Piano Transcription. [arXiv:2107.09142](https://arxiv.org/abs/2107.09142)
- Kong et al. 2021 — High-Resolution Piano Transcription with Pedals. [arXiv:2010.01815](https://arxiv.org/abs/2010.01815)
- Bittner et al. 2022 — Basic Pitch (ICASSP). [github.com/spotify/basic-pitch](https://github.com/spotify/basic-pitch)
- Gardner et al. 2021/22 — MT3 (ICLR). [arXiv:2111.03017](https://arxiv.org/abs/2111.03017), [openreview](https://openreview.net/forum?id=iMSjopcOn0p)
- Choi et al. 2022 — Pop2Piano. [arXiv:2211.00895](https://arxiv.org/abs/2211.00895)
- Donahue et al. 2022 — Sheet Sage / melody from Jukebox. [arXiv:2212.01884](https://arxiv.org/abs/2212.01884)
- Cheuk et al. 2023 — Jointist. [arXiv:2302.00286](https://arxiv.org/abs/2302.00286)
- Cheuk et al. 2023 — DiffRoll. [arXiv:2210.05148](https://arxiv.org/abs/2210.05148)
- Toyama et al. 2023 — hFT-Transformer (ISMIR). [arXiv:2307.04305](https://arxiv.org/abs/2307.04305)
- Lu et al. 2023 — PerceiverTF. [arXiv:2306.10785](https://arxiv.org/abs/2306.10785)
- Tan et al. 2024 — MR-MT3. [arXiv:2403.10024](https://arxiv.org/abs/2403.10024)
- Chang et al. 2024 — YourMT3+ (MLSP). [arXiv:2407.04822](https://arxiv.org/abs/2407.04822)
- Chen et al. 2024 — PiCoGen / PiCoGen2. [arXiv:2407.20883](https://arxiv.org/abs/2407.20883), [arXiv:2408.01551](https://arxiv.org/abs/2408.01551)
- Choi et al. 2024 — Mobile-AMT (EUSIPCO). [paper PDF](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000036.pdf)
- Yuan et al. 2023 — MERT. [arXiv:2306.00107](https://arxiv.org/abs/2306.00107)
- Watanabe et al. 2025 — Streaming Piano Transcription (ISMIR). [arXiv:2503.01362](https://arxiv.org/abs/2503.01362)
- 2024 — Towards Robust Transcription (noise injection). [arXiv:2410.14122](https://arxiv.org/abs/2410.14122)
- 2025 — Noise-to-Notes diffusion drum AMT. [arXiv:2509.21739](https://arxiv.org/abs/2509.21739)
- 2025 — Etude piano cover generation. [arXiv:2509.16522](https://arxiv.org/abs/2509.16522)
- 2025 — NeurIPS AMT Challenge results. [openreview](https://openreview.net/pdf?id=NG187AZ71W), [challenge site](https://ai4musicians.org/transcription/2025transcription.html)

#### Repositories (deployable)
- [spotify/basic-pitch](https://github.com/spotify/basic-pitch) — Apache-2.0
- [bytedance/piano_transcription](https://github.com/bytedance/piano_transcription) — Apache-2.0; pip [piano_transcription_inference](https://pypi.org/project/piano-transcription-inference/)
- [magenta/mt3](https://github.com/magenta/mt3) — Apache-2.0
- [magenta/magenta](https://github.com/magenta/magenta) (Onsets and Frames) — Apache-2.0
- [sony/hFT-Transformer](https://github.com/sony/hFT-Transformer) — MIT
- [mimbres/YourMT3](https://github.com/mimbres/YourMT3) — **GPL-3.0**
- [gudgud96/MR-MT3](https://github.com/gudgud96/MR-MT3) — research
- [sony/DiffRoll](https://github.com/sony/DiffRoll)
- [chrisdonahue/sheetsage](https://github.com/chrisdonahue/sheetsage) — code MIT, models CC BY-NC-SA 3.0
- [Music-and-Culture-Technology-Lab/omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) — MIT
- [facebookresearch/demucs](https://github.com/facebookresearch/demucs) — MIT
- [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)
- [HuggingFace transformers Pop2Piano](https://huggingface.co/docs/transformers/en/model_doc/pop2piano)

#### Commercial / SaaS comparators
- [Klangio (klang.io)](https://klang.io/) — proprietary; Piano2Notes, Melody Scanner, Scan2Notes
- [AnthemScore](https://www.lunaverus.com/) — proprietary


---

## Music Source Separation as Preprocessing for Piano Transcription

**Audience:** Oh Sheet engineering team
**Date:** 2026-04-25
**Question:** Will running MSS *before* Basic Pitch on a pop mix actually improve the resulting piano sheet music — and if so, which model and which integration pattern?

---

### TL;DR

Yes — for the kind of input Oh Sheet receives (drums + vocals + bass + guitar/keys layered pop mix), running a vocal/instrumental separator first will measurably help downstream transcription, even though the relationship between SDR and AMT-F1 is famously nonlinear. The strongest, most production-realistic recommendation is **a Mel-Band RoFormer "vocals/instrumental" model run via the `audio-separator` Python package**, not full 4/6-stem demucs separation. Use Pattern D (vocal+drum *suppressor*, transcribe the residual instrumental). HTDemucs `htdemucs_6s` is *not* recommended for the piano stem — its piano is famously bleedy and artifacted.

---

### 1. Comparison Table — Top Open-Source MSS Models (early 2026)

All numbers are pulled from the cited papers/repos. SDR is on MUSDB18-HQ unless noted. "Avg" is the mean of vocals/drums/bass/other. "License" is the *code/weights* license, which in some cases differs from the paper.

| Model | Year | Avg SDR (M18-HQ) | Vocals SDR | Stems | Params | License | Notes |
|---|---|---|---|---|---|---|---|
| **Spleeter** (Deezer) | 2019 | ~5.9 | 6.86 (MWF) | 2/4/5 | ~10M | MIT | Reference baseline, 100x realtime on GPU. [src](https://github.com/deezer/spleeter/wiki/Separation-Performances) |
| **Open-Unmix (UMX)** | 2019 | ~6.3 | ~6.3 | 4 | small (LSTM) | MIT | Reproducibility reference. [src](https://sigsep.github.io/open-unmix/) |
| **HTDemucs (v4)** | 2022 | 9.00 | ~7.9 | 4 | 41M | MIT | Standard production default. [src](https://github.com/facebookresearch/demucs) |
| **HTDemucs FT** | 2022 | 9.20 | 8.38 (vocals-only chk) | 4 (per-stem chk) | 41M ×4 | MIT | 4× slower; per-source fine-tunes. [src](https://github.com/facebookresearch/demucs) |
| **htdemucs_6s** | 2023 | n/a (worse on V/D/B/O) | — | 6 (+piano,+guitar) | 41M | MIT | **Piano stem has known bleed/artifacts**; experimental. [src](https://github.com/facebookresearch/demucs) |
| **MDX23C / TFC-TDF v3** | 2023 | 7.02–10.17 | 10.17 | 2 (V/I) | ~50M | MIT (UVR) | SDX23 finalist; v3 in audio-separator. [src](https://arxiv.org/abs/2306.09382) |
| **BS-RoFormer (L=12)** | 2023 | **10.02** | 11.02 | 2 / 4 | 82.8M (L=9) / 93.4M (L=12) | code MIT (lucidrains); ByteDance weights vary | SDX23 #1; viperx vocals chk: 12.98 SDR for vocals. [src](https://arxiv.org/abs/2309.02612), [audio-separator weights](https://github.com/nomadkaraoke/python-audio-separator) |
| **Mel-RoFormer** | 2024 | 9.64 (L=6) | 11.21–11.60 | 2 / 4 | 84.2M (L=6) | code MIT; weights MIT-via-UVR | **Best vocals**; +0.5 dB over BS on vocals. [src](https://arxiv.org/abs/2310.01809), [vocal-AMT paper](https://arxiv.org/abs/2409.04702) |
| **SCNet (XL)** | 2024 | 9.0 | competitive | 4 | 10.08M | code via repo (check) | **48% of HTDemucs CPU time**; 4× fewer params. [src](https://arxiv.org/abs/2401.13276) |
| **Banquet (query-band)** | 2024 | n/a (MoisesDB) | — | N (query) | 24.9M | CC BY-NC-SA 4.0 (research) | Beats `htdemucs_6s` on **piano + guitar** at MoisesDB. [src](https://arxiv.org/abs/2406.18747) |
| **Moises-Light** | 2025 | competitive at ~13× fewer params | — | 4 | small | research | Resource-efficient; not yet a standard checkpoint to grab. [src](https://arxiv.org/abs/2510.06785) |

Top vocal-stem leaderboard at MVSep (2025-07): BS-Roformer 11.89 SDR vocals / 18.20 SDR instrumental; Mel-Band RoFormer 11.35 / 17.66; ensemble (BS+Mel+SCNet) 11.93 / 18.23. [MVSep algorithms](https://mvsep.com/en/algorithms)

---

### 2. Inference Footprint (what matters for Cloud Run CPU)

Cloud Run today is CPU-only by default. GPU is available but provisioning latency + cost makes it awkward for spiky transcription jobs. The question is: **can we afford MSS in CPU?**

| Model | CPU real-time factor | 3-min song on 2 vCPU | RAM needed | Disk |
|---|---|---|---|---|
| HTDemucs (default segments) | ~1.5× audio (per Demucs README) | ~4.5 min | ~7 GB recommended (3 GB min w/ shorter segments) | ~80 MB |
| HTDemucs FT | ~6× (4× htdemucs slowdown) | ~18 min | ~7-8 GB | ~320 MB (4 chk) |
| `htdemucs_6s` | similar to htdemucs | ~4.5 min | ~7 GB | ~53 MB |
| `demucs.cpp` (C++, ggml/Eigen) | ~1× audio in low-mem mode | ~3 min | <2 GB tunable | 53–160 MB |
| SCNet | ~0.7× (48% of HTDemucs) | ~2 min | smaller (10M params) | ~40 MB est |
| BS-RoFormer (vocals) | not real-time on 2-vCPU | several minutes | 4-6 GB w/ 80M params at fp32 | ~330 MB |
| Mel-RoFormer (vocals) | similar to BS-RoFormer | several minutes | 4-6 GB | ~340 MB |

Sources: [Demucs README](https://github.com/facebookresearch/demucs), [demucs.cpp](https://github.com/sevagh/demucs.cpp/), [SCNet paper](https://arxiv.org/abs/2401.13276), [BS-RoFormer params](https://arxiv.org/html/2310.01809v1).

The one solid CPU benchmark from the wild: **a 7-minute track in ~2 min 15s on PyTorch CPU for HTDemucs** — that's roughly 0.32× realtime, which is *better* than the 1.5× Demucs README claim because the README counts model load + IO. [src](https://github.com/facebookresearch/demucs/issues/1)

So **HTDemucs base on a 2-vCPU Cloud Run instance for a 3-min song = ~1–2 minutes of wall-clock** is realistic. RoFormer models double that or worse on CPU because they are ~2× the parameters.

---

### 3. Does MSS preprocessing actually help downstream AMT?

This is the question that most blog posts skirt. Let's be honest about what the literature says.

#### 3a. Strong evidence FOR
- **Mel-RoFormer paper (Wang et al., 2024)** explicitly chains MSS then AMT for vocal melody transcription, finds the pre-trained separation backbone *fine-tuned* for transcription gives "substantial performance improvement (e.g., a 7.5 percentage point increase in COnPOff)" over a SpecTNT baseline. They argue training transcription from scratch on a mix yields "significantly inferior performance." [src](https://arxiv.org/html/2409.04702v1)
- **Lyric transcription with Whisper paper (2025)**: on MUSDB-ALT, group-level WER drops from 23.59% (raw mix) → 20.00% (mdx_extra-separated vocals into Whisper). On more diverse Jam-ALT they saw "essentially unchanged" WER. So **separation helps when the mix is dense and the transcriber is general-purpose**, less when the input is already vocal-prominent. [src](https://arxiv.org/html/2506.15514v1)
- **Jointist (2023)**: jointly training MSS + transcription improved transcription by 1+ pp F1 *and* MSS by +5 dB SDR. [src](https://ar5iv.labs.arxiv.org/html/2302.00286)
- **freemusicdemixer.com (production)**: explicitly markets a separate-first pipeline; claims 96% piano note accuracy on a separated stem (not independently verified, but a real product running this pattern). [src](https://freemusicdemixer.com/)

#### 3b. Evidence AGAINST / Caveats
- **The Whisper-ALT paper found higher hallucination rates on separated audio**: "MSS artifacts can trigger hallucinations" in the downstream transcriber. SDR going up does NOT always mean transcription going up. [src](https://arxiv.org/html/2506.15514v1)
- **MR-MT3 (2024)** addresses *instrument leakage* in transcription internally rather than via upstream MSS, suggesting that for multi-instrument transcription specifically, in-model handling can be more robust than separating first. [src](https://arxiv.org/html/2403.10024v1)
- **htdemucs_6s piano is too leaky** to feed straight into Basic Pitch — the developers themselves call it experimental with "a lot of bleeding and artifacts." [src](https://github.com/facebookresearch/demucs)

#### 3c. The honest synthesis
SDR and AMT-F1 measure different things. A model can have +1 dB SDR but introduce phase artifacts that confuse a frame-onset transcriber. Empirically, **vocal/drum suppression** consistently helps because the suppressed energy was definitely interfering with pitched-instrument detection — those classes are spectrally distinct from piano. **Full 6-stem isolation of piano** does NOT consistently help because today's piano-stem outputs have aggressive masking artifacts that look like spurious onsets.

---

### 4. Recommendation for Oh Sheet

#### Pick: Pattern D — *Vocal + Drum Suppression* with Mel-RoFormer (or BS-RoFormer)

**Why not Pattern A (split everything → transcribe each stem):**
- The piano stem from `htdemucs_6s` is unreliable.
- Banquet beats it on piano but is CC BY-NC-SA 4.0 (research-only, blocks commercial deployment without negotiation).
- Cumulative latency of 4–6 separations is brutal on Cloud Run CPU.

**Why not Pattern B (auxiliary conditioning):**
- Basic Pitch is a fixed black-box model; we can't add conditioning channels without retraining.

**Why not Pattern C (joint MSS+AMT):**
- MR-MT3 / Jointist are research artifacts on Slakh2100. No production-quality multi-instrument pop transcription model exists end-to-end. Bringing one online is an order of magnitude more work than D.

**Why D wins:**
- Use a Mel-RoFormer "vocals" model to extract a clean *instrumental* stem (which is what we actually want for Basic Pitch).
- A high-quality vocals/instrumental model achieves ~17–18 dB SDR on the instrumental side per MVSep — i.e., the residual is ~99% clean.
- Drums can be additionally suppressed with a second pass (Mel-RoFormer drums-only or HTDemucs drums-only checkpoint) if drum pitch detection is hurting Basic Pitch.
- Single-pass (vocals model) is the sweet spot: one inference, ~2–3 min on Cloud Run CPU for a 3-min song.

#### Concrete model choice (in priority order):
1. **`mel_band_roformer_vocals`** via `audio-separator` (the Wang/ByteDance model exposed through UVR's MIT-licensed weights). Vocals SDR ~12.98, instrumental SDR ~17.66. [src](https://github.com/nomadkaraoke/python-audio-separator)
2. **HTDemucs (`htdemucs`)** as the proven, easy fallback. MIT, well-supported, Python API trivial. Pull `vocals` and `drums` and reconstruct an instrumental from `bass+other`.
3. **SCNet** if CPU latency becomes a hard constraint — half the inference time of HTDemucs at comparable SDR. [src](https://arxiv.org/abs/2401.13276)

---

### 5. Practical Integration Sketch

#### 5.1 Stage placement in the Oh Sheet pipeline

Insert a new stage between *Ingest* and *Transcribe*, only when `pipeline_variant in {"full", "audio_upload"}` (skip for MIDI).

```
Ingest (decode/normalize) ──▶ Separate (NEW) ──▶ Transcribe (Basic Pitch on instrumental.wav) ──▶ Arrange ──▶ Humanize ──▶ Engrave
```

The new stage outputs a single artifact: `instrumental.wav` (44.1 kHz stereo). The original mix stays accessible for arrangement-side fallbacks.

#### 5.2 Python code sketch (Demucs path — simplest, MIT-clean)

```python
## backend/workers/separate.py
from pathlib import Path
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio

_MODEL = None  # warm singleton — Cloud Run keeps it across invocations

def get_demucs():
    global _MODEL
    if _MODEL is None:
        _MODEL = get_model("htdemucs")  # 4-stem, ~80 MB
        _MODEL.eval()
    return _MODEL

def separate_to_instrumental(input_wav: Path, output_wav: Path) -> None:
    model = get_demucs()
    wav, sr = torchaudio.load(str(input_wav))
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
    if wav.shape[0] == 1:                       # mono → stereo expected
        wav = wav.repeat(2, 1)
    wav = wav.unsqueeze(0)                      # batch dim

    with torch.no_grad():
        sources = apply_model(
            model, wav,
            device="cpu",
            shifts=0,                            # no test-time augmentation
            split=True, overlap=0.10,
            segment=8,                           # 8s segments → fits in <3GB
            num_workers=0,
        )                                       # [1, 4, 2, T] — drums, bass, other, vocals
    drums, bass, other, vocals = sources[0]
    instrumental = bass + other                  # drop vocals AND drums

    torchaudio.save(str(output_wav),
                    instrumental.cpu(),
                    model.samplerate,
                    encoding="PCM_F", bits_per_sample=32)
```

#### 5.3 Python code sketch (Mel-RoFormer path — better quality)

```python
## backend/workers/separate.py
from audio_separator.separator import Separator

_SEP = None

def get_separator():
    global _SEP
    if _SEP is None:
        _SEP = Separator(output_dir="/tmp/sep", output_format="wav")
        _SEP.load_model(model_filename="mel_band_roformer_vocals_v2.ckpt")
    return _SEP

def separate_to_instrumental(input_wav, output_dir):
    sep = get_separator()
    out = sep.separate(str(input_wav))
    # out = ["…_(Vocals).wav", "…_(Instrumental).wav"]
    return next(p for p in out if "Instrumental" in p)
```

[audio-separator package](https://pypi.org/project/audio-separator/)

#### 5.4 Cloud Run / Celery considerations
- Bake the model weights into the container (do **not** download per-cold-start). HTDemucs adds ~80 MB; Mel-RoFormer adds ~340 MB.
- Set `PYTORCH_NO_CUDA_MEMORY_CACHING=1` and pin segment length to keep RSS under 2 GB.
- Since `PipelineRunner` already dispatches via Celery (per the codebase), this is a new worker module, not a new service. Add a `separate` task to `backend/workers/` and a stage entry in `PipelineConfig`.
- Cache separated outputs by `sha256(audio_bytes) + model_id` in the existing blob store — same song separated twice = same output, especially likely on QA runs.

#### 5.5 Expected runtime budget on Cloud Run (2 vCPU, 4 GB RAM)
- HTDemucs: ~30–60 s for a 3-min song. Adds ~5 s cold start.
- Mel-RoFormer: ~90–180 s for a 3-min song.
- demucs.cpp (if we ever want to optimize): ~15–30 s. Adds C++ dependency to the container.

---

### 6. Gotchas

#### 6.1 Spectral artifacts that bite Basic Pitch
- **Phase artifacts**: spectrogram-domain models (BS/Mel-RoFormer, MDX23C) re-use the input mixture's phase. This causes audible "musical noise" that frame-onset detectors can mistake for transients. [src](https://en.wikipedia.org/wiki/Music_Source_Separation)
- **Onset smearing on transients**: drums leaking into "other" (or vice versa) at low SDR creates spurious onsets that Basic Pitch happily transcribes as 16th-note ghost notes. **This is the #1 reason Pattern D works**: removing drums removes the source of false-positive onsets.
- **Vocal residuals at the formant frequencies (~600 Hz–3 kHz)** show up as wandering pseudo-pitches. A 17+ dB SDR instrumental from Mel-RoFormer mostly avoids this; HTDemucs at 9 dB SDR does not.

#### 6.2 The "doesn't translate" problem
SDR ↑ does NOT mechanically mean F1 ↑ in the chained system:
- The Whisper-ALT paper documents *higher hallucination rates* on better-separated audio. [src](https://arxiv.org/html/2506.15514v1)
- This means we need to **verify on Oh Sheet's actual outputs** — pick 5 representative pop songs, run with and without separation, eyeball the resulting MIDI/PDF, and only then commit to the integration. Do not trust SDR proxies.

#### 6.3 License gotchas
- HTDemucs code/weights: MIT. Safe.
- BS-RoFormer original ByteDance weights: license unclear; the **`lucidrains/BS-RoFormer` code is MIT** but the high-SDR `viperx` and `jarredou` checkpoints come from the UVR community — *check each individual `.ckpt` license* before shipping commercially. The audio-separator repo asks for UVR attribution.
- Banquet: CC BY-NC-SA 4.0 — **non-commercial only**. Skip for prod.
- Moises-Light: paper code may be released under research license — check before adopting.

#### 6.4 Format gotchas
- All these models expect 44.1 kHz stereo. If users upload 22.05 kHz mono (YouTube extracts often), Oh Sheet's Ingest stage must upsample first or quality drops markedly.
- Demucs is documented at 44.1 kHz; running at other sample rates will work but degrade.
- Long files: HTDemucs has a max 7.8 s segment for the transformer attention — the Demucs library auto-chunks, but if you bypass that (e.g., direct ONNX), be aware.

#### 6.5 Demucs is archived
`facebookresearch/demucs` was **archived 2025-01-01** by Meta. The code still works, but no future fixes for new PyTorch versions etc. There are forks (e.g., `ZFTurbo/Music-Source-Separation-Training`) that retrain Demucs-arch models and are more actively maintained. [src](https://github.com/facebookresearch/demucs)

#### 6.6 Not all models output 6 stems with piano
If at some point you actually want a piano-only stem (Pattern A revisited):
- `htdemucs_6s`: works but *piano stem is bad*.
- Banquet: beats htdemucs_6s on piano, but research license.
- BS-Roformer-SW (community 6-stem variant): MVSep reports 7.83 SDR piano — modest. [src](https://mvsep.com/algorithms/34)
- MVSep's dedicated piano-only model: 7.83 SDR piano.
- **Bottom line**: piano isolation is still mediocre; don't bet the pipeline on it.

---

### 7. Open questions to resolve experimentally

1. Does Pattern D (vocal-suppressed instrumental) actually beat raw mix through Basic Pitch on Oh Sheet's test set? **A/B test on 10 songs.**
2. Does adding *drum* suppression on top of vocal suppression (two-pass MSS) help further, or do the artifacts stack? **Three-way A/B.**
3. Is the Mel-RoFormer quality gain over HTDemucs worth the 2–3× CPU latency hit? Or do users care more about turnaround than fidelity? **Latency vs F1 measurement on Cloud Run-grade hardware.**
4. Is there value in caching MSS outputs in the blob store keyed by `sha256(audio)`? Probable yes for QA reruns; measure cache hit rate after a week.

---

### Sources

- HTDemucs / Demucs repo: https://github.com/facebookresearch/demucs
- BS-RoFormer paper: https://arxiv.org/abs/2309.02612
- Mel-Band RoFormer paper: https://arxiv.org/abs/2310.01809 ; HTML: https://arxiv.org/html/2310.01809v1
- Mel-RoFormer for Vocal Sep + Melody Transcription: https://arxiv.org/abs/2409.04702 ; HTML: https://arxiv.org/html/2409.04702v1
- SCNet: https://arxiv.org/abs/2401.13276
- Banquet: https://arxiv.org/abs/2406.18747
- Moises-Light: https://arxiv.org/abs/2510.06785
- BSRNN: https://arxiv.org/abs/2209.15174
- KUIELab MDX-Net: https://arxiv.org/abs/2111.12203
- TFC-TDF-UNet v3: https://arxiv.org/abs/2306.09382
- Apollo audio restoration: https://arxiv.org/abs/2409.08514
- MR-MT3 (instrument leakage): https://arxiv.org/abs/2403.10024 ; HTML: https://arxiv.org/html/2403.10024v1
- Jointist: https://ar5iv.labs.arxiv.org/html/2302.00286
- MSS+Whisper for ALT (cascaded artifact study): https://arxiv.org/html/2506.15514v1
- Spleeter performance: https://github.com/deezer/spleeter/wiki/Separation-Performances
- Spleeter repo: https://github.com/deezer/spleeter
- Open-Unmix: https://sigsep.github.io/open-unmix/
- audio-separator: https://github.com/nomadkaraoke/python-audio-separator ; https://pypi.org/project/audio-separator/
- ZFTurbo Music-Source-Separation-Training: https://github.com/ZFTurbo/Music-Source-Separation-Training ; pretrained models: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md
- demucs.cpp: https://github.com/sevagh/demucs.cpp/
- demucs.onnx: https://github.com/sevagh/demucs.onnx
- MoisesDB paper: https://archives.ismir.net/ismir2023/paper/000073.pdf
- MVSep algorithm leaderboard: https://mvsep.com/en/algorithms ; https://mvsep.com/algorithms/34
- Free Music Demixer (production stack reference): https://freemusicdemixer.com/
- SDX 2023 Music Demixing Track summary: https://arxiv.org/html/2308.06979v4
- Demucs CPU benchmark anecdote: https://github.com/facebookresearch/demucs/issues/1
- Demucs htdemucs_6s piano artifact discussion: https://github.com/facebookresearch/demucs (release notes)

---

## Direct Pop-Audio → Piano Arrangement Models — Research Report

Date: 2026-04-25
Scope: Evaluate Pop2Piano-style models that bypass per-instrument transcription, plus
symbolic and hybrid alternatives, for replacement of Oh Sheet's stubbed `arrange` stage.

---

### 1. Pop2Piano — Deep Dive

#### 1.1 Paper, repo, samples
- Paper: Choi & Lee, *Pop2Piano: Pop Audio-based Piano Cover Generation*, ICASSP 2023.
  - arXiv: <https://arxiv.org/abs/2211.00895>
  - HTML: <https://ar5iv.labs.arxiv.org/html/2211.00895>
- Official code: <https://github.com/sweetcocoa/pop2piano>
- Pretrained weights: <https://huggingface.co/sweetcocoa/pop2piano>
- Demo Space (T4 GPU): <https://huggingface.co/spaces/sweetcocoa/pop2piano>
- Project page with audio: <https://sweetcocoa.github.io/pop2piano_samples/>
- Hugging Face Transformers integration: <https://huggingface.co/docs/transformers/model_doc/pop2piano>

#### 1.2 Architecture
- Encoder–decoder Transformer based on **T5-small** (~59 M params)
  ([ar5iv](https://ar5iv.labs.arxiv.org/html/2211.00895)).
- Encoder takes a mel/log-spectrogram derived from the raw waveform; decoder produces
  MIDI tokens auto-regressively.
- Token vocabulary (~2,400) is the union of:
  - **Note Pitch** (128 values, 88 piano keys actually used)
  - **Note On/Off** (2)
  - **Beat Shift** (100 values, quantized to 8th-note grid)
  - **EOS / PAD / special** ([HF docs](https://huggingface.co/docs/transformers/model_doc/pop2piano))
- A **composer condition token** (21 arrangers) lets the user pick stylistic flavor.

#### 1.3 Training data ("PSP" dataset)
- 5,989 YouTube piano-cover/original pop pairs scraped from 21 named arrangers, then
  filtered by sync-quality.
- Final aligned set: **307 hours / 4,989 tracks**, mostly **K-pop**, with some Western
  pop/hip-hop ([ar5iv §3](https://ar5iv.labs.arxiv.org/html/2211.00895)).
- Sync built with an automated pipeline (synchrotoolbox + key/tempo alignment).
- Ablations also use POP909 ([Wang 2020](https://arxiv.org/abs/2008.07142),
  909 Chinese pop songs with melody/bridge/piano stems).

#### 1.4 Reported evaluation
Pop2Piano's authors only report a **subjective listening test**:
- Winning-rate (Pop2Piano-PSP vs Pop2Piano-POP909): **70.4%**
- MOS Pop2Piano-PSP: **3.216 / 5**, vs POP909 baseline 2.856; human-played
  ground truth (GTF) 3.771. Source:
  [ar5iv Table 2](https://ar5iv.labs.arxiv.org/html/2211.00895).
- **No objective pitch / chord / onset accuracy is reported** in the paper.
  Later work fills this in (see §1.6).

#### 1.5 License (important caveat)
- The official repo `sweetcocoa/pop2piano` does **not contain a top-level LICENSE
  file** (404 on `/blob/main/LICENSE`); the README mentions none either
  ([repo](https://github.com/sweetcocoa/pop2piano)).
  By default this means **all rights reserved** — you cannot legally use the code
  in a commercial product without explicit permission.
- The Hugging Face port (`sweetcocoa/pop2piano`) and its Transformers integration
  do **not state a model license** on the model card we fetched — Hugging Face
  third-party reports describe the Transformers framework integration as Apache 2.0
  (referring to the *re-implementation*, not the weights).
- **Recommendation:** treat Pop2Piano weights as research-only until Choi & Lee
  publish a license, or contact the authors for a commercial-use grant.

#### 1.6 Robustness / failure cases
- Hard-coded **8th-note quantization** ⇒ no triplets, 16ths, or trills
  (paper §5.4).
- **4-beat context window** during training ⇒ "melody contour or texture of
  accompaniment have less consistency when generating longer than four-beat"
  (paper §5.4). Long-range coherence is weak.
- Hacker News commenters: "10 simultaneous notes being played over 4 octaves,
  which doesn't seem humanly possible"; "the rhythms and chord selections are
  extremely weird and no human would ever play this way" ([HN
  thread](https://news.ycombinator.com/item?id=34205996)).
- Training set is **predominantly K-pop**; cross-genre generalization is
  acknowledged as variable ([HF docs](https://huggingface.co/docs/transformers/model_doc/pop2piano)
  notes "does pretty well" on Western pop/hip-hop).
- **Hip-hop is a documented weak point**, both for Pop2Piano and the follow-up
  PiCoGen, because lead-sheet content is sparse ([PiCoGen
  paper](https://arxiv.org/html/2407.20883v1)).
- Vocal handling is ambiguous: monophonic vocals may collapse to single notes,
  or be voiced as octaves — model has no explicit policy.

#### 1.7 Inference compute
- Demo runs on a single **T4 GPU**
  ([HF Space](https://huggingface.co/spaces/sweetcocoa/pop2piano)). No precise
  latency is published, but the Space is interactive (~tens of seconds per
  3-minute clip), which is consistent with a 59 M-param T5 encoder over ~few
  minutes of audio chunks.
- Dependencies: `pretty-midi==0.2.9`, `essentia==2.1b6.dev1034`, `librosa`,
  `scipy`, `transformers`. The pinned `essentia` build is a common point of pain
  ([repo issue #11](https://github.com/sweetcocoa/pop2piano/issues/11)).
- CPU-only inference is not reported but should work given the model size; GPU
  recommended.

---

### 2. Other Direct / Arrangement Models

#### 2.1 PiCoGen (ICMR 2024)
- Tan et al., <https://arxiv.org/abs/2407.20883>
- **Two-stage**: SheetSage-style audio→lead-sheet, then a CP-Word Transformer
  generates piano bar-by-bar conditioned on the lead sheet.
- Avoids the need for paired (pop, piano) audio. Uses **Pop1k7** (~1,700 piano
  covers).
- Beats Pop2Piano on **subjective** ratings across all 5 dimensions (overall
  preference 3.35 vs 2.55), but **loses on Melody Chroma Accuracy (0.17 vs
  0.25)** — confirming the "tasteful but less faithful" trade-off.
- Hip-hop is flagged as a failure mode (sparse lead sheets).
- License: paper CC BY 4.0; repo not located in our search.

#### 2.2 PiCoGen2 (ISMIR 2025)
- Tan et al., <https://arxiv.org/abs/2408.01551>; review:
  [moonlight](https://www.themoonlight.io/en/review/picogen2-piano-cover-generation-with-transfer-learning-approach-and-weakly-aligned-data).
- Pre-trains on piano-only, fine-tunes on **weakly-aligned** (audio, piano)
  pairs. Drops the discrete lead-sheet bottleneck of v1; uses a continuous lead-
  sheet *encoder* directly.
- Beats baselines on both objective and subjective metrics across 5 pop genres.

#### 2.3 Etude (2025, three-stage)
- <https://arxiv.org/html/2509.16522v1>
- Pipeline: **Extract** (AMT-APC-style dense feature MIDI) → **Structuralize**
  (Beat-Transformer downbeat grid) → **Decode** (25.5 M GPT-NeoX with a
  reduced "Tiny-REMI" tokenization).
- Trained on 4,752 J-pop / K-pop pairs (~500 h).
- Outperforms Pop2Piano and PiCoGen2 on subjective similarity, fluency, dynamic
  expression, overall (3.16/3.73/3.46/3.50 ÷ 5).
- Important admission: performance is **upper-bounded by beat tracking and
  feature extraction**. Single-feature-stream input is an information bottleneck.

#### 2.4 AMT-APC (2024)
- Komiya & Fukuhara, <https://arxiv.org/abs/2409.14086>;
  repo <https://github.com/misya11p/amt-apc> (**MIT license**).
- Fine-tunes the **hFT-Transformer** AMT model into a piano-cover generator
  with a continuous "style vector" condition. Authors claim it "reproduces
  original tracks more accurately than any existing models" — i.e. it leans
  toward the *transcription* end of the trade-off.
- Project page: <https://misya11p.github.io/amt-apc/>
- Practically attractive: clean MIT, single-file inference (`python infer
  input.wav`), YouTube URL support, single-GPU training.

#### 2.5 Symbolic arrangers (need a lead-sheet input)
- **AccoMontage** (Zhao 2021): phrase-retrieval + neural style transfer; folk/pop
  oriented. Repo <https://github.com/zhaojw1998/accomontage>.
- **AccoMontage-3 / Structured-Arrangement-Code** (NeurIPS 2024):
  multi-track full-band, style-prior modelling. Repo
  <https://github.com/zhaojw1998/AccoMontage-3>; paper
  <https://arxiv.org/abs/2310.16334>.
- **PopMAG** (Zhang 2020) and **GETMusic** (Lv 2023, Microsoft Muzic): lead-
  sheet ⇒ multi-track. GETMusic is a **diffusion** transformer that beats PopMAG
  on the same tasks (<https://arxiv.org/abs/2305.10841>). Microsoft repo:
  <https://github.com/microsoft/muzic> — MIT.

#### 2.6 Helpful related models for the humanize step
- **Compound Word Transformer** / **MuseMorphose** / **REMI / REMI+** —
  performance-style tokenization.
- **PerformanceRNN**, **Music Transformer**, **PiCoGen** all model expressive
  velocity/timing.

---

### 3. Audio → Lead-Sheet → Arrange (hybrid stack feasibility)

#### 3.1 Components available *today*
| Stage | Model | Code | License | Notes |
|-------|-------|------|---------|-------|
| Audio → melody+chords | **SheetSage** (Donahue 2022) | <https://github.com/chrisdonahue/sheetsage> | Code MIT, models CC BY-NC-SA 3.0 | Outputs lead-sheet PDF + MIDI; uses Jukebox features. **Non-commercial weights.** |
| Audio → chord labels | **BTC** ([arxiv](https://arxiv.org/abs/1907.02698)) / **ChordFormer** ([arxiv](https://arxiv.org/html/2502.11840v1)) | github | Mostly MIT / Apache | ChordFormer 2025 SOTA: 84.7% Root, 84.1% MajMin, 83.6% MIREX. |
| Audio → vocal F0 | **CREPE** / **RMVPE** ([arxiv](https://arxiv.org/pdf/2306.15412)) | both MIT | OK to ship | RMVPE robust to mixed audio. |
| Audio → polyphonic MIDI | **Basic Pitch** ([repo](https://github.com/spotify/basic-pitch)) / **MT3** ([repo](https://github.com/magenta/mt3)) / **YourMT3+** | mixed | already in Oh Sheet pipeline | MT3 is multi-instrument and Apache-2.0 but heavy. |
| Lead-sheet → piano | **AccoMontage(-3)** / **PopMAG** / **GETMusic** | github | MIT-ish | Symbolic only. |
| Audio → piano (E2E) | **Pop2Piano** / **PiCoGen2** / **Etude** / **AMT-APC** | github / HF | mixed (AMT-APC = MIT) | See §1, §2. |

#### 3.2 Hybrid strategies for Oh Sheet
**A. "Faithful core, tasteful voicing" hybrid**
- Keep Basic Pitch for melody+bass (Oh Sheet already invokes it for transcribe).
- Use **ChordFormer / BTC** to lock harmony.
- Feed (melody MIDI + chord labels) as a synthesized lead sheet into
  **AccoMontage-3** or **GETMusic** for the piano accompaniment voicing.
- Pros: every component MIT/Apache; explainable; bug-fixable per stage.
- Cons: error-prone; transcription artifacts propagate.

**B. SheetSage front + symbolic arranger**
- Use SheetSage for audio→lead-sheet (best public quality), then
  AccoMontage-3.
- Pros: SheetSage quality is high; matches PiCoGen's stage 1.
- Cons: **CC BY-NC-SA on SheetSage models** — not allowed for commercial use
  without retraining a replacement.

**C. End-to-end Pop2Piano-class model**
- Drop arrange entirely; replace transcribe+arrange with one model.
- Pros: minimal pipeline, Korean/Western pop reasonably good, runs on a single
  T4.
- Cons: license unclear; long-context coherence weak; no triplets/16ths;
  can produce unplayable voicings (10-note 4-octave chords).

**D. AMT-APC end-to-end (MIT licensed)**
- Drop arrange; use AMT-APC for audio→piano-cover MIDI directly.
- Pros: clean MIT, claims best objective fidelity in its class; single-GPU.
- Cons: J-pop / piano-cover-Youtube training distribution; less "tasteful
  arranger" feel than Pop2Piano; smaller community.

**E. "Hybrid of hybrids"**
- Use Basic Pitch for the *melody* line only (most reliable).
- Use AMT-APC or Pop2Piano for *accompaniment voicing* only.
- Merge in the engrave step with melody on top staff, generated voicing on
  bottom — playability fix-ups (max 5 notes per hand, octave clamp) applied
  post-hoc.
- This actually matches Oh Sheet's existing 2-hand-piano contract very well.

---

### 4. Recommendation for Oh Sheet

**Short answer: hybrid (E), with AMT-APC as the candidate accompaniment model
and Basic Pitch retained for melody.**

Reasoning:

1. **License risk is the single biggest blocker.** Pop2Piano has no LICENSE file
   on the official repo and SheetSage's models are CC BY-NC-SA. AMT-APC is
   MIT — a clean drop-in for production
   ([repo](https://github.com/misya11p/amt-apc)).

2. **Quality target.** Oh Sheet wants a *playable, pleasant* piano score, not a
   forensic transcription. Direct audio→piano models (Pop2Piano family) are
   subjectively rated higher than transcription-then-arrange pipelines
   (PiCoGen MOS 3.35 > Pop2Piano 2.55 in PiCoGen's eval; Etude beats both).
   Pure transcription (Basic Pitch + naive flatten) consistently sounds
   non-musical.

3. **Reuse what works.** Basic Pitch is already integrated for transcribe and
   gives reasonable melody contour. Don't throw that away — instead, use it as
   one input voice (right-hand top line) and let the direct model generate the
   harmonic / accompaniment voicing for the left hand and inner voices.

4. **Avoid the 4-beat coherence trap.** Pop2Piano (and Pop2Piano-flavored
   layers in Etude) chunk by ~4 beats. Compensate by keeping the structural
   frame from Basic Pitch + a separate beat tracker (e.g. `madmom` already in
   common ML stacks, or **Beat-Transformer** as Etude does).

5. **Stay implementation-friendly.** AMT-APC ships a single-line inference
   script (`python infer input.wav` → MIDI), MIT-licensed, fine-tunable on
   piano-cover YouTube data using their own pipeline. It slots into Oh Sheet
   alongside Basic Pitch with similar shape (audio → MIDI).

#### Proposed pipeline change

```
Today:    ingest → transcribe (Basic Pitch) → arrange (stub) → humanize → engrave
Proposed: ingest → split:
                     ├─ melody  : Basic Pitch (top line, vocal-aware)
                     └─ accomp. : AMT-APC piano-cover MIDI
                   → merge + clamp (≤5 notes/hand, octave-collapse) → humanize → engrave
```

Run a small A/B against:
- Pop2Piano via Hugging Face Transformers (research-only until license is
  confirmed),
- Etude (when code is published — currently arxiv only as of 2025),
- A symbolic baseline: ChordFormer + AccoMontage-3 (B/A hybrid).

Use the existing `evaluate` machinery if you can synthesize a small labelled set.
A 20-song A/B with MOS-style listener votes is enough to pick a winner.

#### What NOT to do

- Don't replace Oh Sheet's `arrange` with **Pop2Piano** unless you (a) get a
  written license grant from Choi & Lee or (b) accept research-only use.
- Don't ship **SheetSage models** in production — `CC BY-NC-SA 3.0` on the model
  weights blocks commercial use. The MIT-licensed *code* is fine to fork if you
  retrain.
- Don't drop transcription entirely. The melody line from a real transcriber is
  what makes the resulting score actually useful for a player.

---

### 5. Open questions / where to validate next

- Does **AMT-APC** satisfy "playable" constraints (≤5 notes per hand) out of the
  box? The paper claims AMT-style fidelity, which usually means too many notes.
  Need a quick probe on a few songs.
- Is there a **reproducible commercial-friendly Pop2Piano replacement**? PiCoGen2
  (paper CC BY 4.0) might be retrainable on POP909 + Pop1k7; check its repo
  status (paper says "model + dataset to be released", repo not located).
- For Oh Sheet's `humanize` stage: **PerformanceRNN** / **Music Transformer**
  output velocity-and-microtiming sequences from quantized MIDI — likely a
  better fit than what's stubbed today.
- Benchmark: There is **no public leaderboard** for "pop audio → piano cover";
  every paper builds its own subjective study. This is a real gap. Etude's
  metrics (Warp Path Deviation, Rhythmic Grid Coherence, IOI Pattern Entropy)
  could be adopted as Oh Sheet's internal eval.

---

## Piano-Specialist Transcription Models: Research Report

**Goal:** identify piano-specific Automatic Music Transcription (AMT) models for use either (a) on a piano stem after source separation in the Oh Sheet pipeline, or (b) as a fine-tuned head replacing/augmenting Basic Pitch.

Date: 2026-04-25.

---

### 1. TL;DR Recommendation

**Use ByteDance/Kong `piano_transcription_inference` (PyPI) as the production default for v1**, with **Aria-AMT (EleutherAI, ICLR 2025) as the v2 robustness upgrade** once a piano stem is reliably extracted by source separation. **Avoid hFT-Transformer for production** despite its SOTA MAESTRO numbers — it is the most MAESTRO-overfit of the modern models, and the academic community has explicitly demonstrated that this kind of model degrades sharply on real-world / non-Disklavier audio. (See §5 "MAESTRO Overfitting" — backed by Edwards et al. 2024 IEEE SPL and the "Towards Musically Informed Evaluation" study.)

**Pedal modeling matters and is rare.** Only Kong 2021 and the new Streaming Piano Transcription model (Niikura 2025) handle sustain pedal explicitly, which is critical for engraving readability. If we lose pedal info, sheet music will look like a blizzard of staccato eighth-notes.

---

### 2. Comparison Table — Piano-Specialist Models

| Model | Year | MAESTRO Note F1 (onset) | MAESTRO Note+Off F1 | MAESTRO N+O+Vel F1 | Pedal F1 | Params | License | Weights | Notes |
|------|------|------------------------|---------------------|--------------------|---------|--------|---------|---------|-------|
| **Onsets & Frames** (Hawthorne 2018) | 2018 | 94.80%¹ | 78.30%¹ | 73.25%¹ | — | ~26M (jongwook PT impl) | Apache 2.0 | Yes (Magenta) | Foundational baseline; no pedal head. Brittle on OOD.¹⁰ |
| **High-Res Piano Transcription** (Kong 2021, ByteDance) | 2021 | **96.72%**² | 82.47%² | 80.92%² | **91.86%**² (sustain pedal) | ~84M (CRNN + regression heads) | Apache 2.0 (training repo); MIT (inference pkg)³ | Yes (Zenodo)⁴ | First pedal benchmark on MAESTRO. Production-grade `pip install piano-transcription-inference`. Repo archived Dec 2025. |
| **HPPNet-sp** (Wei 2022) | 2022 | 97.18%⁵ | 83.80%⁵ | 82.24%⁵ | — | smaller than O&F⁵ | MIT⁶ | No (must train)⁶ | Harmonic dilated conv; small but no pedal, no provided checkpoint. |
| **hFT-Transformer** (Toyama 2023, Sony) | 2023 | **97.44%**⁷ (SOTA) | **90.53%**⁷ | **89.48%**⁷ | — | 5.5M⁷ | MIT (code)⁸ | Yes (release zip)⁸ | SOTA on MAESTRO, no pedal head. Trained only on MAESTRO → high OOD risk. |
| **Onsets & Velocities** (Fernández 2023, EUSIPCO) | 2023 | 96.78%⁹ | — | 94.50% (onset+vel)⁹ | — | ~3.1M⁹ | CC BY 4.0⁹ | Yes (real-time demo + checkpoint)⁹ | Smallest model that holds onset SOTA. CPU real-time. No offset / no pedal. |
| **Maman & Bermano (NoteEM)** | 2022 (ICML) | 89.7 (cross-dataset)¹¹ | — | — | — | — | CC BY-NC-SA 4.0¹¹ | Yes¹¹ | Trained on unaligned in-the-wild data, NOT trained on MAESTRO; surprisingly competitive cross-dataset. **Non-commercial license.** |
| **Edwards et al.** (2024 IEEE SPL) | 2024 | ~96.6¹² | — | — | — | builds on Kong arch¹² | CC BY 4.0¹² | Yes (Zenodo)¹² | Built specifically for **OOD robustness** via heavy aug; F1=88.4 on MAPS without seeing MAPS. **No pedal in checkpoint.**¹² |
| **Aria-AMT** (Bradshaw & Colton, ICLR 2025) | 2025 | competitive on MAESTRO¹³ | — | — | — | seq2seq Whisper-style¹³ | Apache 2.0¹³ | Yes (HuggingFace)¹³ | Designed for **in-the-wild robustness**. Used to transcribe 100k hours → Aria-MIDI dataset. ~131× real-time on H100¹³. |
| **Streaming Piano Transcription** (Niikura 2025) | 2025 | 96.52%¹⁴ | 89.44%¹⁴ | — | yes (sustain)¹⁴ | 16M¹⁴ | research code¹⁴ | n/a | Streaming mode, 380ms latency. Pedal modeled. |
| **Mobile-AMT** (Kusaka & Maezawa 2024 EUSIPCO) | 2024 | comparable to SOTA but 82.9% less compute¹⁵ | — | — | — | mobile-class¹⁵ | — | n/a | Aug improves note F1 by **+14.3pts on realistic audio**¹⁵. On-device feasibility. |
| **Basic Pitch** (Spotify 2022, instrument-agnostic) | 2022 | far below piano specialists¹⁶ | — | — | — | small | MIT | Yes | Currently used in Oh Sheet `transcribe` stage. Generalist, trained on diverse data, NOT a piano specialist. |

¹ Kong 2021 (arXiv 2010.01815) reports OnF baseline MAESTRO numbers in their comparison: onset 94.80%, onset+offset 78.30%, onset+offset+velocity 73.25%.
² Kong et al. 2021 — https://arxiv.org/abs/2010.01815 — "achieves an onset F1 of 96.72% on the MAESTRO dataset, outperforming previous onsets and frames system of 94.80%, … pedal onset F1 score of 91.86%".
³ https://pypi.org/project/piano-transcription-inference/ (MIT). Underlying training code: https://github.com/bytedance/piano_transcription (Apache 2.0).
⁴ Pretrained weights auto-downloaded from https://zenodo.org/record/4034264 on first use.
⁵ HPPNet-sp numbers as reported in hFT-Transformer comparison table (Toyama 2023): "HPPNet-sp scored 93.15% Frame F1, 97.18% Note F1, 83.80% Note+Offset F1, and 82.24% Note+Offset+Velocity F1." See https://arxiv.org/html/2307.04305 .
⁶ https://github.com/WX-Wei/HPPNet — MIT license; README explicitly states no pretrained weights provided, must train via `python train.py`.
⁷ Toyama et al. 2023 (hFT-Transformer) — https://arxiv.org/html/2307.04305 Table 2: Frame 93.24, Note 97.44, Note+Off 90.53, Note+Off+Vel 89.48. Params 5.5M.
⁸ https://github.com/sony/hFT-Transformer — MIT; pretrained checkpoint in releases (`checkpoint.zip` containing `model_016_003.pkl`). Python 3.6.9 environment, NVIDIA RTX 2080 Ti for eval, A100 for training.
⁹ Fernández 2023 — https://arxiv.org/abs/2303.04485 — onset F1 = 96.78% (MAESTRO), onset+velocity F1 = 94.50%, ~3.1M params, 24ms temporal resolution, real-time on commodity hardware. Code: https://github.com/andres-fr/iamusica_training. CC BY 4.0.
¹⁰ Edwards et al. 2024 — https://arxiv.org/html/2402.01424v1 "Kong et al.'s model … pitch shift: 19.2 percentage point F1 drop (82.4 → 72.4); reverb: 10.1 percentage point drop (82.4 → 72.3)".
¹¹ Maman & Bermano 2022 — https://benadar293.github.io/ — "MAESTRO 89.7 note F1, MAPS 87.3 note F1, … not trained on MAESTRO". CC BY-NC-SA 4.0.
¹² Edwards et al. 2024 — https://zenodo.org/records/10610212 — "this checkpoint does not include pedal predictions, and should use the Regress_onset_offset_frame_velocity_CRNN module"; F1=88.4 on MAPS test set; CC BY 4.0; on MAESTRO 96.6 vs Kong's 96.7.
¹³ EleutherAI Aria-AMT — https://github.com/EleutherAI/aria-amt — Apache 2.0; weights at https://huggingface.co/datasets/loubb/aria-midi/resolve/main/piano-medium-double-1.0.safetensors ; per Bradshaw & Colton ICLR 2025 (https://arxiv.org/abs/2504.15071), used for transcribing 100,629 hours at "131x real-time" on H100 with batch size 128.
¹⁴ Niikura et al. 2025 — https://arxiv.org/html/2503.01362 — Note-F1 (onset only) 96.52%, Note-F1 (onset+duration) 89.44%, Frame F1 91.75%; 16M params; 380ms streaming latency; sustain pedal modeled (ablation: removing pedal drops F1 from 89.44 → 87.56).
¹⁵ Kusaka & Maezawa, Mobile-AMT, EUSIPCO 2024 — https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000036.pdf — "augmentation improves note F1 by 14.3 points on realistic audio … reduces compute by 82.9%".
¹⁶ https://engineering.atspotify.com/2022/06/meet-basic-pitch — instrument-agnostic.

---

### 3. Detailed Write-Ups: Top 3 Candidates

#### 3.1 ByteDance / Kong "High-Resolution Piano Transcription" (2021) — current production default

- **Paper:** "High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times", Kong et al., IEEE/ACM TASLP 2021. https://arxiv.org/abs/2010.01815
- **Architecture:** CRNN with regression heads that *analytically* compute precise onset and offset times rather than picking the nearest frame — the source of its name "high-resolution". Joint heads for onset, offset, frame, velocity, and **sustain pedal** (still rare).
- **MAESTRO test:** Note onset F1 96.72%, note onset+offset F1 82.47%, note onset+offset+velocity F1 80.92%, **pedal onset F1 91.86%** (https://arxiv.org/abs/2010.01815).
- **Weights:** auto-downloaded from Zenodo by `piano_transcription_inference` package. https://zenodo.org/record/4034264
- **License:** Apache 2.0 (training repo) / MIT (inference pkg). Both commercial-friendly.
- **Integration:** trivial.
  ```python
  pip install piano-transcription-inference  # 0.0.6 latest, MIT
  from piano_transcription_inference import PianoTranscription
  pt = PianoTranscription(device='cuda')  # or 'cpu'
  pt.transcribe(audio_array, 'output.mid')
  ```
- **Inference cost:** Repo's training code targets PyTorch 1.4+; inference works on CPU but several minutes of audio takes minutes on CPU; on a single consumer GPU it's faster than real-time (anecdotal, the repo doesn't report RTF). The model runs frame-by-frame so memory is bounded.
- **Robustness:** **Mediocre.** Edwards et al. 2024 (https://arxiv.org/html/2402.01424v1) tested it: pitch-shift augmentation knocks Kong from 82.4 F1 → 72.4 F1 on MAPS (-19.2 pts); reverb -10.1 pts; background noise approximately neutral. So it is fine on Disklavier-like sources, fragile on YouTube rips.
- **Why pick it for v1:** It's the only mature, pip-installable, commercially-licensed piano transcriber that **outputs sustain-pedal events**. For sheet-music engraving, pedal info is what makes the score readable. None of hFT-Transformer, OnF, Onsets & Velocities, HPPNet, NoteEM expose pedal at all.
- **Status warning:** the github repo `bytedance/piano_transcription` was archived (read-only) on Dec 8, 2025. The PyPI package is still maintained by Kong (latest 0.0.6 released Jan 2025). Treat as stable but unmaintained.

#### 3.2 Aria-AMT (EleutherAI, ICLR 2025) — best v2 candidate for noisy stems

- **Paper:** Bradshaw & Colton, "Aria-MIDI: A Dataset of Piano MIDI Files for Symbolic Music Modeling", ICLR 2025. https://arxiv.org/abs/2504.15071 (the dataset paper; AMT model also detailed here)
- **Companion preprint on the AMT model:** https://www.alexander-spangher.com/papers/aria_amt.pdf
- **Architecture:** Whisper-style sequence-to-sequence (encoder-decoder Transformer) trained to emit MIDI-like tokens. "Maintains architectural cohesion with Whisper, making it relatively straightforward to integrate into pre-existing Whisper run-times." Per https://arxiv.org/html/2504.15071v1 .
- **Robustness:** **The killer feature.** The authors *chose* Aria-AMT over hFT-Transformer / Kong for the Aria-MIDI build because of its robustness to "diverse timbres and recording qualities" — they then ran it on 1.7M YouTube videos and got a 1M-MIDI corpus they're confident in. Per https://arxiv.org/html/2504.15071v1 : "Aria-AMT, a piano transcription model designed to handle diverse timbres and recording qualities, was chosen for its robustness in transcribing audio from a diverse set of recording environments, compared to models used in previous work."
- **Throughput:** 100,629 hours of audio transcribed in 765 H100 hours at batch=128 → ~131× real-time. https://arxiv.org/html/2504.15071v1 . Per-stream latency (batch=1) is much higher because it's seq2seq with autoregressive decoding; the github repo provides `-compile` and Int8 quant for that case.
- **Weights:** publicly hosted on HuggingFace, https://huggingface.co/datasets/loubb/aria-midi/resolve/main/piano-medium-double-1.0.safetensors . Apache 2.0 license.
- **Integration:**
  ```bash
  git clone https://github.com/EleutherAI/aria-amt
  pip install -e .
  aria-amt transcribe medium-double <ckpt> -load_path piano.wav -save_dir out -bs 1 -compile
  ```
- **Caveats:**
  - Needs Python 3.11.
  - Int8 quantization requires GPU with BF16 (Ampere/Hopper).
  - Pedal modeling **not explicitly documented** in repo README (verify). The paper trained on Aria-MIDI which itself has pedal info, so the model likely emits pedal tokens.
- **Why not v1:** model is large, repo is recent (67 stars at time of research), Python 3.11 + GPU required, and seq2seq batch=1 latency is high. Use it as v2 once the source separator output is known to be noisy.

#### 3.3 hFT-Transformer (Sony, ISMIR 2023) — best raw MAESTRO numbers, **avoid for v1**

- **Paper:** Toyama, Akama, Ikemiya, Takida et al., ISMIR 2023. https://arxiv.org/abs/2307.04305
- **Repo:** https://github.com/sony/hFT-Transformer (MIT license, weights in release `checkpoint.zip`).
- **MAESTRO results (Table 2 of the paper, half-stride variant):**
  - Frame F1 93.24
  - Note F1 (onset only) **97.44** (SOTA)
  - Note+Offset F1 **90.53** (SOTA)
  - Note+Offset+Velocity F1 **89.48** (SOTA)
- **MAPS results:** Frame 82.89, Note 85.14, Note+Offset 66.34, Note+Offset+Velocity 48.20. Note the dramatic drop on velocity F1 (48.2 vs 89.5 on MAESTRO) — this is the textbook "MAESTRO velocity calibration doesn't transfer to MAPS Disklavier".
- **Architecture:** two-level hierarchical Transformer. First level: 1-D conv on time, Transformer encoder on frequency, Transformer decoder converting frequency. Second level: Transformer encoder on time. 256 emb / 4 heads / 3 layers / 512 FFN. Total **5.5M params** — surprisingly small.
- **Integration cost:** the repo is research-grade; eval scripts assume Python 3.6.9 and a specific MAESTRO directory layout. No pip package. No `piano_transcription_inference`-style ergonomic wrapper.
- **No pedal head.** This alone disqualifies it for sheet engraving v1.
- **OOD risk:** the Edwards et al. 2024 robust-transcription study tested Toyama on MAPS without seeing MAPS at training time and got 85.1 F1 (https://arxiv.org/html/2402.01424v1 Table); this is a 12-point drop from its MAESTRO numbers. The "Towards Musically Informed Evaluation" study (https://arxiv.org/html/2406.08454v2) further showed all three of Hawthorne / Kong / T5-style models drop sharply under noise+reverb on Disklavier re-recordings.
- **Bottom line:** the SOTA F1 is real, but the conditions are MAESTRO-clean. For Oh Sheet's input distribution (user MP3s, YouTube rips), this is not the right starting point.

---

### 4. Specialist Mentions Worth Knowing

- **Onsets & Velocities (Fernández 2023, EUSIPCO).** 3.1M params, 96.78 onset F1 on MAESTRO, real-time on a laptop CPU. https://arxiv.org/abs/2303.04485 . Useful for an on-device fallback or a low-latency progress-bar transcription. **No offset, no pedal.**
- **HPPNet-sp (Wei 2022, ISMIR).** Highest-F1 Note-only number until hFT-Transformer dethroned it. Smaller than O&F via harmonic dilated conv. **No pedal, no shipped weights** — would need to train. https://arxiv.org/abs/2208.14339 / https://github.com/WX-Wei/HPPNet
- **Edwards et al. 2024 IEEE SPL** — explicitly built for OOD robustness; gets MAPS 88.4 F1 cross-dataset. Weights on https://zenodo.org/records/10610212 (CC BY 4.0). **No pedal head in checkpoint.** Could be useful as the "robust generalist" until Aria-AMT integration is done.
- **NoteEM / Maman & Bermano 2022 (ICML)** — unaligned-supervision EM training on real recordings; cross-dataset MAESTRO 89.7 / MAPS 87.3 *without training on either*. **CC BY-NC-SA 4.0 — non-commercial only**, so unsuitable for Oh Sheet if you ever charge users. https://benadar293.github.io/
- **Streaming Piano Transcription with sustain pedal (Niikura 2025).** 16M params, 380ms latency, models pedal. https://arxiv.org/html/2503.01362 — useful if Oh Sheet wants live-progress-bar transcription with pedal info.
- **Mobile-AMT (Kusaka & Maezawa, EUSIPCO 2024)** — designed for mobile; reports +14.3 F1 points on realistic audio with their aug pipeline. https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000036.pdf
- **MR-MT3 (2024)** — multi-instrument, not piano-specialist; ~62.5 MIDI-class F1 on Slakh2100. Better seen as alternative to MT3 for the multi-instrument transcribe-by-instrument idea, not as a piano specialist. Note: **CC BY-NC-SA 4.0** (non-commercial). https://arxiv.org/html/2403.10024v1
- **Commercial baselines** — Klangio (https://klang.io/transcription-studio/) and AnthemScore. Both polished, both closed-source / paid. Not realistic to integrate, but useful as accuracy *ceilings* for benchmarking against our own pipeline.
- **Apple "Piano Transcription"** — no public model exists as of April 2026. The "Audio to MIDI" feature in Logic Pro is closed-source and Apple has not published a model card.

---

### 5. The MAESTRO Overfitting Problem — back this with citations

**The single most important production risk.** Every model in §2 that is trained *only* on MAESTRO has been shown to crater on out-of-distribution piano audio. The Edwards et al. 2024 study and the "Musically Informed Evaluation" study converge on the same finding.

Concrete numbers:

- **Kong 2021** (https://arxiv.org/html/2402.01424v1): pitch-shift aug at eval time → −19.2 F1 pts on MAPS. Reverb → −10.1 pts. Adding noise → roughly neutral. *MAESTRO-clean accuracy hides catastrophic failure modes.*
- **All three models tested in "Towards Musically Informed Evaluation"** (https://arxiv.org/html/2406.08454v2 — OnF, Kong, T5 sequence-to-sequence): "When tested on re-recorded piano performances and degraded audio (with noise/reverberation), all models showed substantial performance drops. Statistical testing confirmed significant differences between MAESTRO and real-world Disklavier recordings across all models."
- **Edwards et al.** (https://arxiv.org/html/2402.01424v1) summary: standard MAESTRO-trained models "can severely overfit to acoustic properties of training data when measured on out-of-distribution annotated piano data." Their fix: weighted-mixture training over MAESTRO + Studio-MAESTRO + 6 Pianoteq synth versions, plus aug — random 7-band EQ, random background noise, random pitch shift ±0.1 semitone, reverb, each at p=0.5. Result: their model only drops 2.5 F1 pts under aug-eval vs Kong's 19.2 pts.

**What this means for Oh Sheet.** A user uploading an MP3 of a YouTube recording of a piano cover, possibly with phone-mic background noise or YouTube's compression, is *exactly* the OOD condition that breaks Kong/hFT-Transformer/HPPNet. The mitigations:

1. **Source separation upstream.** A dedicated piano-stem extractor (Demucs piano, MDX-Net) before the AMT will move audio closer to the MAESTRO clean-piano distribution. Most of the win comes from this step.
2. **Pick a robustness-aware model for the AMT itself.** Aria-AMT (designed for diverse-timbre robustness, used to scrape 1M YouTube performances) and Edwards et al. 2024 (explicit aug-trained Kong variant) are the two production-realistic options. Mobile-AMT's aug pipeline (+14.3 pts on realistic audio) is the third.
3. **Always evaluate on a non-MAESTRO held-out set** before shipping. The paper community now routinely reports MAESTRO + Disklavier OOD numbers; we should adopt the same internal practice.

---

### 6. Recommendation for Oh Sheet

#### Phase 1 (next sprint): drop-in upgrade behind a feature flag

- **Keep Basic Pitch as the default** for the `transcribe` stage to avoid regressions for non-piano inputs.
- **Add a `piano_specialist=true` path** in `backend/workers/transcribe` that, when the upstream `decompose` stage flags the source as predominantly piano, routes to **`piano_transcription_inference` (Kong 2021)**. This:
  - Adds pedal info to the `TranscriptionResult` (extend the contract — Schema bump).
  - Adds `~84M` model download (Zenodo, cached).
  - Apache-2.0/MIT, commercial-safe.
  - Trivial integration:
    ```python
    # backend/workers/transcribe/piano_specialist.py
    from piano_transcription_inference import PianoTranscription, sample_rate
    import librosa
    audio, _ = librosa.load(in_path, sr=sample_rate, mono=True)
    pt = PianoTranscription(device=device)
    pt.transcribe(audio, out_midi_path)
    # Note: 'pt' bundle includes pedal events in MIDI control changes
    ```
- **Risks:** Kong is MAESTRO-trained → noisy inputs may still produce garbage. Mitigate with a quality gate that compares Basic Pitch and Kong outputs and falls back if Kong's note count is anomalous.
- **Pedal-data handoff:** Update `backend/contracts.py` `TranscriptionResult` to include `pedal_events: list[PedalEvent]`. The Engrave stage (LilyPond / Verovio / etc.) needs this to render Ped./*.

#### Phase 2 (after source-separation lands): swap to Aria-AMT

Once `svc-decomposer` gives us a clean piano stem:
- Keep Kong as the fast path.
- Add **Aria-AMT** as the robustness path for stems where the separator's confidence is low (genre = pop/jazz/cover, presence of vocal residue, etc.).
- Aria-AMT's HF weights are ~few-hundred-MB; deploy on the same Cloud Run worker as the GPU-bearing Transcribe service, or keep a lightweight queue → GPU spot pool.
- Apache-2.0, commercially safe.
- Verify pedal-token coverage in Aria-AMT outputs (the README is silent; the dataset has pedal so the model likely does too).

#### Phase 3 (research): fine-tune

- Once we have a labelled in-house piano dataset (logs of corrected user output), fine-tune Edwards et al.'s checkpoint or Kong's checkpoint with realistic-audio aug — basically replicating the Edwards / Mobile-AMT recipe. Their aug pipelines (pitch ±0.1, reverb, background noise, 7-band EQ) are well-documented and known to recover ~+15 F1 on realistic audio.

#### Hard "no" list

- **hFT-Transformer** — MAESTRO SOTA but no pedal, brittle on OOD, awkward integration. Save for the leaderboard headline only.
- **NoteEM (Maman & Bermano)** — CC BY-NC-SA 4.0. Non-commercial only.
- **MR-MT3** — multi-instrument, not piano-specialist; also CC BY-NC-SA 4.0.
- **HPPNet-sp** — no pretrained weights; would need to train from scratch on MAESTRO ourselves.

---

### 7. Open Questions / Followups

1. Verify whether **Aria-AMT actually emits pedal events** (read the AMT-only preprint at https://www.alexander-spangher.com/papers/aria_amt.pdf or test on a known piece).
2. Benchmark **Kong vs Aria-AMT on our own held-out user-uploaded MP3 set** (10–20 examples). The literature predicts Aria-AMT wins on YouTube rips and Kong wins on Disklavier-like recordings; confirm.
3. **Latency budget per minute of audio**: Kong CPU is on the order of minutes-per-minute-audio. Aria-AMT batched on a single A100 is reportedly 131× real-time; batch=1 is much slower. This drives the Cloud Run worker shape.
4. **Verovio / engraving stage** must learn to read the new pedal events. This is the integration bottleneck — most piano sheets the user expects will look wrong without it.

---

### Sources

- hFT-Transformer paper — https://arxiv.org/abs/2307.04305
- hFT-Transformer HTML — https://arxiv.org/html/2307.04305
- hFT-Transformer code (MIT) — https://github.com/sony/hFT-Transformer
- Onsets & Frames paper — https://arxiv.org/abs/1710.11153
- Onsets & Frames PyTorch — https://github.com/jongwook/onsets-and-frames
- Kong 2021 paper — https://arxiv.org/abs/2010.01815
- ByteDance training repo — https://github.com/bytedance/piano_transcription
- piano-transcription-inference PyPI — https://pypi.org/project/piano-transcription-inference/
- piano-transcription-inference repo — https://github.com/qiuqiangkong/piano_transcription_inference
- Kong weights on Zenodo — https://zenodo.org/record/4034264
- HPPNet paper — https://arxiv.org/abs/2208.14339
- HPPNet repo — https://github.com/WX-Wei/HPPNet
- Onsets & Velocities (Fernández 2023) — https://arxiv.org/abs/2303.04485
- Onsets & Velocities code — https://github.com/andres-fr/iamusica_training
- Sequence-to-sequence (Hawthorne 2021 ISMIR) — https://archives.ismir.net/ismir2021/paper/000030.pdf
- Edwards et al. 2024 robust transcription — https://arxiv.org/abs/2402.01424
- Edwards et al. 2024 HTML — https://arxiv.org/html/2402.01424v1
- Edwards et al. checkpoint (Zenodo, CC BY 4.0) — https://zenodo.org/records/10610212
- "Towards Musically Informed Evaluation" — https://arxiv.org/html/2406.08454v2
- NoteEM project — https://benadar293.github.io/
- NoteEM paper — https://proceedings.mlr.press/v162/maman22a/maman22a.pdf
- Aria-AMT repo (Apache 2.0) — https://github.com/EleutherAI/aria-amt
- Aria-AMT weights (HuggingFace) — https://huggingface.co/datasets/loubb/aria-midi/resolve/main/piano-medium-double-1.0.safetensors
- Aria-MIDI ICLR 2025 paper — https://arxiv.org/abs/2504.15071
- Aria-MIDI HTML — https://arxiv.org/html/2504.15071v1
- Aria-AMT preprint — https://www.alexander-spangher.com/papers/aria_amt.pdf
- Streaming Piano Transcription (Niikura 2025) — https://arxiv.org/html/2503.01362
- Mobile-AMT (Kusaka & Maezawa, EUSIPCO 2024) — https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000036.pdf
- Min-latency real-time AMT — https://arxiv.org/html/2509.07586
- MR-MT3 paper — https://arxiv.org/html/2403.10024v1
- MR-MT3 repo — https://github.com/gudgud96/MR-MT3
- Magenta Onsets & Frames — https://magenta.tensorflow.org/onsets-frames
- Basic Pitch Spotify — https://engineering.atspotify.com/2022/06/meet-basic-pitch
- Klangio — https://klang.io/transcription-studio/
- AnthemScore vs Klangio — https://eliteai.tools/comparison/anthemscore/vs/klangio
- PianoTrans GUI — https://github.com/azuwis/pianotrans
- Efficient sparse-attention piano transcription (2025) — https://arxiv.org/html/2509.09318

---

## Evaluation Metrics for End-to-End Audio→Sheet Music Transcription

**Research target:** A metrics ladder for Oh Sheet — a pop-to-piano pipeline that takes MP3/MIDI/YouTube and emits PDF + MusicXML + MIDI piano sheet music. Existing AMT literature stops at note-level F1 on classical-piano test sets; we need metrics that reach all the way to "is this sheet music actually playable and recognizable."

This report is organised as a **five-tier ladder**, from low-level acoustic accuracy up to human perception. For each metric: definition, library/code link, when to apply, expected ranges, gotchas, and whether it is **reference-free** (RF) — i.e., usable when no ground-truth piano cover exists.

---

### Tier 1 — Classic AMT metrics (note/frame F1 on a parallel reference)

These are the bread-and-butter numbers from MIREX / MAESTRO / MAPS. Apply them when the input has a clean reference transcription (e.g. solo piano with MIDI ground truth, or when comparing against a known cover). They are **reference-required** and break down on full-mix pop input.

#### 1.1 Note-onset F1 (mir_eval.transcription)
- **Definition:** Match estimated notes to reference notes if the onset is within a tolerance window (default 50 ms) and pitch is exact. Compute precision, recall, F1.
- **Library:** [`mir_eval.transcription.onset_precision_recall_f1`](https://mir-eval.readthedocs.io/latest/api/transcription.html), originally Bay 2009 / Raffel 2014 ([mir_eval paper](https://colinraffel.com/posters/ismir2014mir_eval.pdf)).
- **When:** Solo-piano transcription with parallel MIDI ground truth (MAESTRO, MAPS, SMD).
- **Expected range:** SOTA on MAESTRO ~96–98% F1, ~88% on MAPS (out-of-distribution).
- **Gotchas:** Insensitive to offset, velocity, voicing, rhythm. Two reasonable transcriptions of the same audio can both be "correct" with low cross-F1. Highly distribution-shifted: ~14% degradation on different piano sound, ~52% on non-piano domain ([Riley et al. 2024](https://arxiv.org/pdf/2402.01424)).
- **Reference-free?** No.

#### 1.2 Note onset+offset F1 (mir_eval.transcription)
- **Definition:** As 1.1 but additionally requires offset within max(50 ms, 20% of note duration).
- **Library:** [`mir_eval.transcription.precision_recall_f1_overlap`](https://mir-eval.readthedocs.io/latest/api/transcription.html).
- **Gotchas:** Pedal makes offsets ill-defined; sustain releases drift. Tightening offset tolerance below 80 ms is rarely meaningful for piano because of damper physics.

#### 1.3 Note onset+offset+velocity F1
- **Definition:** As 1.2, additionally normalised MIDI velocity must agree within 10%.
- **Library:** [`mir_eval.transcription_velocity`](https://github.com/mir-evaluation/mir_eval/blob/main/mir_eval/transcription_velocity.py).
- **When:** Performance fidelity matters (humanisation evaluation on MAESTRO).
- **Gotchas:** Pop covers of the same song will have wildly different dynamic shaping; never compute against a different cover.

#### 1.4 Frame-level multi-pitch F1 (mir_eval.multipitch)
- **Definition:** Discretise pianoroll at fixed hop (typically 10 ms), compute per-frame precision/recall/F1 over active pitches.
- **Library:** [`mir_eval.multipitch`](https://mir-eval.readthedocs.io/latest/api/multipitch.html); see [`mir_eval` paper](https://colinraffel.com/posters/ismir2014mir_eval.pdf).
- **Expected range:** SOTA piano frame F1 >90% on MAESTRO.
- **Gotchas:** Heavily inflated by sustained notes — a single 2-second held chord dominates the score. Hop size matters: 10 ms is convention. Always report alongside note-level metrics.

#### 1.5 f0 raw pitch accuracy (CREPE / pYIN style)
- **Definition:** Continuous fundamental-frequency tracking accuracy at a tolerance (typically 50 cents).
- **Library:** [`mir_eval.melody`](https://mir-eval.readthedocs.io/latest/api/melody.html), [`crepe`](https://github.com/marl/crepe), [`pYIN` / librosa](https://librosa.org/doc/main/generated/librosa.pyin.html).
- **When:** Lead melody only — useful for pop where the vocal line is a stand-in target.
- **Reference-free?** No.

#### 1.6 Onset detection F1 (mir_eval.onset)
- **Definition:** Pitch-agnostic onset accuracy; tolerance ~50 ms.
- **Library:** [`mir_eval.onset`](https://mir-eval.readthedocs.io/latest/api/onset.html).
- **When:** As a sanity check decoupled from pitch; useful if Basic Pitch is producing onsets but pitch is off.

---

### Tier 2 — Structural music-information metrics

These evaluate whether the transcription gets the **musical bones** right: key, tempo, downbeat, time signature, chord progression. They tolerate inversion/voicing differences. Many can be computed reference-free (against the original audio's structural analysis), giving partial evaluation when no ground-truth piano cover exists.

#### 2.1 Key detection accuracy (mir_eval.key)
- **Definition:** MIREX weighted score: 1.0 exact match, 0.5 perfect-fifth error, 0.3 relative major/minor, 0.2 parallel major/minor, else 0.
- **Library:** [`mir_eval.key.weighted_score`](https://mir-eval.readthedocs.io/latest/api/key.html).
- **When:** Compare key inferred from transcription vs key inferred from original audio (madmom / Krumhansl-Schmuckler).
- **Expected:** ~70–85% on pop.
- **Reference-free?** Yes — compare transcription's key against original-audio key estimator.

#### 2.2 Tempo accuracy (mir_eval.tempo)
- **Definition:** Tempo estimate within ±4% (or 8% for relaxed). MIREX uses two tempi (P-score).
- **Library:** [`mir_eval.tempo`](https://mir-eval.readthedocs.io/latest/api/tempo.html).
- **Reference-free?** Yes (transcription tempo vs audio-side tempo).
- **Gotchas:** Tempo octave errors (half/double) very common; report octave-relaxed score too.

#### 2.3 Beat / downbeat F-measure (mir_eval.beat, madmom)
- **Definition:** F1 of beat/downbeat positions within ±70 ms of references. madmom RNN-based downbeat tracker is SOTA (~80–85% on pop).
- **Library:** [`mir_eval.beat`](https://mir-eval.readthedocs.io/latest/api/beat.html), [`madmom.features.downbeats`](https://madmom.readthedocs.io/) ([Böck et al. 2016](https://arxiv.org/abs/1605.07008)).
- **Reference-free?** Yes — transcription's downbeat grid vs madmom on original audio.
- **Gotchas:** mir_eval.beat ships **multiple metrics** (F-measure, Cemgil, Goto, P-score, CMLc/CMLt, AMLc/AMLt, Information Gain) — pick CMLt (correct-metrical-level total) as a single robust number, plus F1 ± 70 ms. See [Davies et al. 2014](https://archives.ismir.net/ismir2014/paper/000238.pdf).

#### 2.4 Time-signature accuracy
- **Definition:** Exact match of meter (4/4, 3/4, 6/8…). Confusion matrix more useful than scalar.
- **Library:** No standard library function in mir_eval; check the time signature on transcription's MusicXML against madmom-derived meter or human label. Survey: [Time Signature Detection: A Survey, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512143/).
- **Reference-free?** Partial.

#### 2.5 Chord-sequence accuracy (mir_eval.chord)
- **Definition:** Frame-weighted accuracy with multiple comparison schemes: `root` (root match only), `triads`, `tetrads`, `mirex` (≥3 shared pitches), `majmin`, `sevenths`. Plus segmentation metrics (`overseg`, `underseg`, `seg`).
- **Library:** [`mir_eval.chord`](https://mir-eval.readthedocs.io/latest/api/chord.html), [`madmom.evaluation.chords`](https://madmom.readthedocs.io/en/v0.16/modules/evaluation/chords.html).
- **Reference-free?** Yes — chord-recognise both transcription and original audio, compare.
- **When:** Crucial for pop — chord progression is most of what listeners recognise.
- **Gotchas:** Inversions count as different in `tetrads` but same in `mirex`. Pick one and stick to it.

#### 2.6 Section-boundary F-measure (mir_eval.segment / hierarchy)
- **Definition:** Boundary detection F1 at strict (±0.5 s, HR.5F) and relaxed (±3 s, HR3F) tolerances. Plus pair-wise frame clustering F1 (PWF) and normalised conditional entropy (Sf) for label evaluation.
- **Library:** [`mir_eval.segment`](https://mir-eval.readthedocs.io/latest/api/segment.html), [MSAF](https://github.com/urinieto/msaf) ([Nieto & Bello 2015](https://ccrma.stanford.edu/~urinieto/MARL/publications/NietoBello-ISMIR2015.pdf)).
- **Reference-free?** Yes — compare against MSAF-segmented original audio.
- **Expected:** Inter-annotator F1 on SALAMI is ~0.76 at ±3 s, ~0.67 at ±0.5 s — that's the practical ceiling.

#### 2.7 Voice-separation evaluation (MV2H Voice component)
- **Definition:** F1 over voice-assignment edges in a polyphonic graph.
- **Library:** [MV2H, McLeod 2018](https://github.com/apmcleod/MV2H).
- **Reference-free?** No.

---

### Tier 3 — Arrangement-quality and engraving metrics

These ask whether the **sheet music itself** is a competent piano arrangement: voice leading, hand reachability, rhythmic complexity, and notational legibility. Most are RF — they evaluate the score in isolation.

#### 3.1 MV2H — joint multi-pitch / voice / meter / value / harmony (McLeod & Steedman 2018)
- **Definition:** Composite score combining five sub-metrics:
  1. **Multi-pitch** — note F1 (pitch only)
  2. **Voice** — voice-assignment edge F1
  3. **Meter** — metrical-grid alignment F1
  4. **Value** — note-duration accuracy (rhythmic notation)
  5. **Harmony** — chord-symbol accuracy
  Final MV2H is the unweighted mean of the five. Score in [0, 1].
- **Library:** [github.com/apmcleod/MV2H](https://github.com/apmcleod/MV2H) (Java canonical, Python port slower). Paper: [McLeod & Steedman 2018, ISMIR](https://ismir2018.ismir.net/doc/pdfs/148_Paper.pdf).
- **When:** When you have a parallel reference score (MusicXML/MIDI). Best single composite score for "good transcription" beyond F1.
- **Expected:** Strong systems on classical land in 0.65–0.80; perfect is 1.0.
- **Reference-free?** No — needs reference.
- **Gotchas:** Sensitive to voice-assignment heuristics. Has a **non-aligned** mode (`-a` flag) that handles cases where transcription uses a different time-base — important for our case if engrave step quantises differently.

#### 3.2 Cogliati & Duan score-similarity metric (ISMIR 2017)
- **Definition:** Levenshtein-style edit distance over a score-aware token sequence (notes, rests, beams, ties), then a regression model maps edit distance → predicted human quality rating.
- **Library:** [github.com/AndreaCogliati/MetricForScoreSimilarity](https://github.com/AndreaCogliati/MetricForScoreSimilarity). Paper: [Cogliati & Duan 2017](https://archives.ismir.net/ismir2017/paper/000131.pdf).
- **When:** Direct evaluation of music notation output (not just MIDI).
- **Reference-free?** No.

#### 3.3 musicdiff — score-tree edit distance over MusicXML
- **Definition:** Combined sequence- and tree-edit distance over an intermediate notation tree. Visually-aware (cares about beaming, stem direction, etc.).
- **Library:** [`musicdiff` PyPI](https://pypi.org/project/musicdiff/) (depends on music21≥9.7). Paper: [Foscarin et al. 2019](https://inria.hal.science/hal-02267454v2/document).
- **When:** Engrave-stage QA — diff a generated MusicXML against a reference cover's MusicXML.
- **Reference-free?** No.

#### 3.4 OMR-NED (OMR Normalised Edit Distance, ISMIR 2025)
- **Definition:** Per-measure set-edit-distance over symbol categories (notes, beams, accidentals, articulations…), with category-specific insertion/deletion costs.
- **Library:** Sheet Music Benchmark dataset + code, [arXiv:2506.10488](https://arxiv.org/abs/2506.10488).
- **When:** When you have **rendered-page ground truth** (PDF/PNG) and want a graphical-fidelity score. Originally for OMR but applicable inversely: render and compare.
- **Reference-free?** No.

#### 3.5 Hand reachability / playability score (custom)
- **Definition:** Per-chord, score = 1 if (max-pitch − min-pitch) ≤ 14 semitones AND ≤ 5 simultaneous notes per hand AND no impossible voice crossings; else weighted penalty. Aggregate as fraction of playable chords.
- **Library:** Custom over music21 chord pitch-spans; [music21 chord module](https://music21.org/music21docs/moduleReference/moduleChord.html). Reference for thresholds: [Nakamura & Sagayama, "Statistical Piano Reduction Controlling Performance Difficulty"](https://arxiv.org/pdf/1808.05006).
- **When:** Critical for pop-to-piano output. Caveat: thresholds depend on tempo (broken chords playable slowly are not at speed) and skill level.
- **Reference-free?** Yes.
- **Gotchas:** A 10th is the practical ceiling for most amateurs; some pieces use intentional 11th/12th spans played as broken chords. Decide whether you penalise these.

#### 3.6 Voice-leading smoothness
- **Definition:** Mean semitone displacement per voice across consecutive chords (Tonnetz-based parsimonious-voice-leading metric). Lower is smoother.
- **Library:** Custom over music21 / Tonnetz. Theory: [Tonnetz, Wikipedia](https://en.wikipedia.org/wiki/Tonnetz); [Lerdahl, Tonal Pitch Space](https://global.oup.com/academic/product/tonal-pitch-space-9780195178296).
- **Reference-free?** Yes.
- **When:** Detects choppy arrangements where each chord is in a different inversion than the last.

#### 3.7 Rhythmic complexity / sight-readability
- **Definition:** Combine note-density (notes per beat), syncopation index, smallest-tatum subdivision, and hand-displacement frequency. Map to a difficulty grade (Henle 1–9 or ABRSM 1–8).
- **Library:** [RubricNet](https://arxiv.org/html/2509.16913) descriptors (interpretable difficulty); [Ramoneda et al. 2023, "Combining piano performance dimensions for score difficulty classification"](https://arxiv.org/abs/2306.08480); CIPI dataset for training.
- **Reference-free?** Yes.
- **When:** Filter outputs that are technically correct but unreadable for the target user (intermediate amateur).

#### 3.8 Texture / density features
- **Definition:** Mean polyphony per beat, gap between hands, density-1/2/3 layer count (Couvreur & Lartillot taxonomy).
- **Library:** Custom over music21 / partitura. Theory: [Annotating Symbolic Texture in Piano Music: a Formal Syntax](https://hal.science/hal-03631151/file/main.pdf).
- **Reference-free?** Yes.

#### 3.9 Engraving quality (heuristic)
- **Definition:** Penalties for: ledger lines >3 (use 8va), enharmonic spelling mismatched to key, beaming violating beat groupings, voice crossings, stems wrong direction. Compute against LilyPond / Verovio output.
- **Library:** None standard — write rule checks over MusicXML. Inspect with `musicdiff` against a clean reference.
- **Reference-free?** Yes.

---

### Tier 4 — End-to-end perceptual / re-synthesis / embedding metrics

When you can't define a parallel reference (most pop songs!), measure the transcription by re-synthesising it and comparing against the original audio in a perceptual embedding space. **All Tier 4 metrics are reference-free in our sense — they use the input audio as the reference, not a separate piano cover.**

#### 4.1 Frechet Audio Distance (FAD)
- **Definition:** Distance between Gaussian fits of embeddings (VGGish / CLAP / PANNs) over a *set* of generated vs reference clips.
- **Library:** [microsoft/fadtk](https://github.com/microsoft/fadtk). Original: [Kilgour et al. 2019](https://www.isca-archive.org/interspeech_2019/kilgour19_interspeech.pdf); generative-music adaptation: [Gui et al. 2023](https://arxiv.org/abs/2311.01616).
- **When:** Aggregate quality over a corpus of transcriptions — not per-song.
- **Reference-free?** Set-level (no per-clip ground truth needed).
- **Gotchas:** Requires hundreds of samples for stable estimates; sample-size bias is severe; embedding choice matters (CLAP-music > VGGish for music). Don't trust below ~500 samples.

#### 4.2 CLAP cosine similarity (per-clip semantic similarity)
- **Definition:** Cosine similarity between CLAP-embedded original audio and CLAP-embedded re-synthesised piano transcription.
- **Library:** [LAION-CLAP](https://github.com/LAION-AI/CLAP); [`audiocraft.metrics.clap_consistency`](https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/metrics/clap_consistency.html).
- **When:** Per-song, no reference cover needed.
- **Expected:** ~0.5–0.85 cosine for plausible piano renderings of pop.
- **Reference-free?** Yes (the original audio is the anchor).
- **Gotchas:** Weak correlation with subjective MOS; use as a screen, not a verdict. CLAP-music checkpoint outperforms general CLAP.

#### 4.3 MERT embedding similarity
- **Definition:** As 4.2 but using MERT (music-trained masked-acoustic Transformer).
- **Library:** [m-a-p/MERT](https://huggingface.co/m-a-p/MERT-v1-330M). [Survey: Kader 2025](https://arxiv.org/html/2509.00051v1).
- **Reference-free?** Yes.
- **When:** Music-specialist alternative to CLAP; better for fine-grained timbral and harmonic comparison.

#### 4.4 Chroma cosine similarity (low-level harmonic)
- **Definition:** Cosine similarity between beat-aligned chromagrams of original audio and re-synthesised transcription.
- **Library:** [librosa.feature.chroma_cqt](https://librosa.org/doc/main/generated/librosa.feature.chroma_cqt.html), then bar-wise cosine.
- **Reference-free?** Yes.
- **When:** Cheap continuous proxy for "are the same chords sounding at the same time?"

#### 4.5 Tonnetz / tonal-centroid distance
- **Definition:** L2 distance between tonal-centroid trajectories (chroma → 6-D tonal centroid via perfect-fifth/minor-third/major-third basis).
- **Library:** [`librosa.feature.tonnetz`](https://librosa.org/doc/main/generated/librosa.feature.tonnetz.html).
- **Reference-free?** Yes.

#### 4.6 Grooving / rhythm similarity
- **Definition:** Bar-wise cosine between binary onset patterns (grooving vectors) of original audio and resynthesised transcription.
- **Library:** Custom over librosa onsets. See [Wang et al. 2020 (POP909)](https://arxiv.org/pdf/2008.07142v1) and [Survey: Kader 2025](https://arxiv.org/pdf/2509.00051) for canonical formulations.
- **Reference-free?** Yes.

#### 4.7 Round-trip (transcribe → engrave → re-synth → re-transcribe) consistency
- **Definition:** Agreement of two-stage transcription against the engrave-stage MIDI. F1 of (audio→MIDI₁) vs (audio→engrave→synth→MIDI₂) measures how lossy the engrave step is. A drop indicates quantisation, voicing, or simplification artefacts hurt fidelity.
- **Library:** Pipeline-internal — compose mir_eval.transcription with FluidSynth. Concept aligned with [Simonetta et al. 2022, "A perceptual measure for evaluating the resynthesis of automatic music transcriptions"](https://arxiv.org/abs/2202.12257).
- **Reference-free?** Yes.
- **When:** Diagnoses *which stage* (transcribe vs arrange vs engrave) is lossy.

#### 4.8 Perceptual resynthesis measure (Simonetta et al. 2022)
- **Definition:** Trained model that predicts a subjective MOS-like score from the resynthesised MIDI vs original-audio pair.
- **Library / paper:** [Simonetta et al. 2022, Multimedia Tools & Applications](https://link.springer.com/article/10.1007/s11042-022-12476-0).
- **Reference-free?** Yes.

#### 4.9 Note-level DTW alignment cost
- **Definition:** DTW cost to align transcription's MIDI to original-audio onset envelope (or to mir_eval-aligned reference). Lower = tighter timing.
- **Library:** [`librosa.sequence.dtw`](https://librosa.org/doc/main/generated/librosa.sequence.dtw.html); [parangonar](https://pypi.org/project/parangonar/) for symbolic.
- **Reference-free?** Yes for audio-anchor; no for symbolic-vs-symbolic.

---

### Tier 5 — Human evaluation protocols

Automated metrics never close the loop alone. The Ycart et al. 2020 paper ([TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.57)) and the Pop2Piano user study ([Choi et al. 2022](https://arxiv.org/abs/2211.00895)) explicitly find that mir_eval F1 has only modest correlation with subjective judgement. Always include a human study in any release-blocking eval.

#### 5.1 MOS (Mean Opinion Score)
- **Definition:** N≥20 listeners rate each item 1–5 on each axis. Standard axes: Overall Quality, Faithfulness to Original, Playability/Readability, Musicality.
- **Best practice:** Mix listener pool (musicians + non-musicians); blind random ordering; include hidden-reference upper bound and trivial-baseline lower bound; report 95% CIs (bootstrap).
- **References:** [How to Measure AI-Generated Music Quality, Mureka](https://www.mureka.ai/hub/aimusic/how-to-measure-the-real-sound-quality/); [MMMOS, arXiv:2507.04094](https://arxiv.org/html/2507.04094v2).
- **Reference-free?** Yes (rating is on the artefact alone).

#### 5.2 A/B preference (paired comparison)
- **Definition:** Listener picks Oh-Sheet vs baseline (or vs human cover) head-to-head; report win-rate with binomial CI.
- **Best practice:** ≥3 listeners per item, balanced presentation order, attention checks.
- **When:** More sensitive than MOS for ordering similar systems. Used by Pop2Piano.

#### 5.3 Multi-axis MOS (MMMOS-style)
- **Definition:** Separate ratings for *Production Quality*, *Production Complexity*, *Content Enjoyment*, *Content Usefulness*. We add: *Faithfulness*, *Playability*, *Notational correctness*.
- **Library/paper:** [MMMOS arXiv:2507.04094](https://arxiv.org/html/2507.04094v2).

#### 5.4 Pianist rubric (expert eval)
- **Definition:** ≥3 pianists score the engraved output on: enharmonic spelling, beam grouping, voice separation, fingering implications, stem direction, rest placement. 5-point Likert per axis.
- **Best practice:** Calibrate inter-rater agreement (Krippendorff's α ≥ 0.6 acceptable for subjective). Ground truth in [Henle difficulty levels](https://www.henle.de/Levels-of-Difficulty/) for difficulty validation.

#### 5.5 Sight-readability test (in-the-wild)
- **Definition:** Show transcription to N pianists at the target skill level, ask them to play through; measure (a) error rate, (b) time-to-fluency, (c) self-reported readability 1–5.
- **Reference-free?** Yes.
- **When:** The decisive end-game test for "is this sheet music useful?"

#### 5.6 Holzapfel ethnomusicology user-study protocol
- **Definition:** Domain experts rate transcription utility for their actual use case (study, performance, cover, learning).
- **Reference:** [Holzapfel et al. 2019, ISMIR](https://archives.ismir.net/ismir2019/paper/000082.pdf).

---

### Recommended metric ladder for Oh Sheet (300-word summary)

Oh Sheet's pop-mix → piano-sheet pipeline does not have parallel ground truth in the wild — there is no canonical "correct" piano cover of any given pop song. The metric system therefore needs four characteristics: (1) **graceful degradation** when no cover exists, (2) **stage-level diagnosability** so we know whether transcribe / arrange / engrave is to blame, (3) **alignment with what users care about** (recognisable, playable, readable), and (4) **affordable enough to run in CI**.

**For CI on every PR (cheap, automated, reference-free):**
1. **Tier 2 structural triplet** — key (`mir_eval.key`), tempo (`mir_eval.tempo`), chord progression (`mir_eval.chord` or `madmom`) computed on input audio vs on resynthesised transcription. These are robust under voicing differences.
2. **Tier 3 playability gate** — fraction-playable chords (hand-span ≤ 14 semitones, ≤ 5 notes/hand) plus voice-leading smoothness median. Hard-fail PRs that regress this.
3. **Tier 4 audio-embedding similarity** — CLAP-music cosine and chroma cosine between original audio and synthesised transcription. Cheap, ~per-song, no reference.

**For weekly regression suite (slow but stable):**
4. **Tier 1 + MV2H** on a curated 50-song held-out set with hand-made piano-cover MIDI references — full mir_eval transcription + MV2H.5 sub-scores, FAD over the corpus.

**For release gates (decisive):**
5. **Tier 5 human evaluation** — multi-axis MOS (Faithfulness / Playability / Musicality) with N≥30 listeners + ≥3 expert pianists doing rubric and sight-readability scoring. Pre-register a minimum win-rate (e.g. ≥55%) vs the previous release.

Three numbers always reported together: chord-accuracy (recognisable), playability fraction (playable), CLAP-music cosine (perceptually similar). One number is misleading; this triple is the smallest set that captures Oh Sheet's actual product surface.

---

### Sources

- [mir_eval documentation](https://mir-eval.readthedocs.io/latest/)
- [mir_eval GitHub](https://github.com/mir-evaluation/mir_eval)
- [mir_eval paper, Raffel et al. 2014](https://colinraffel.com/posters/ismir2014mir_eval.pdf)
- [MV2H, McLeod 2018, GitHub](https://github.com/apmcleod/MV2H)
- [McLeod & Steedman 2018, ISMIR](https://ismir2018.ismir.net/doc/pdfs/148_Paper.pdf)
- [Cogliati & Duan, ISMIR 2017](https://archives.ismir.net/ismir2017/paper/000131.pdf)
- [musicdiff PyPI](https://pypi.org/project/musicdiff/) and [Foscarin et al. 2019](https://inria.hal.science/hal-02267454v2/document)
- [Sheet Music Benchmark / OMR-NED, arXiv:2506.10488](https://arxiv.org/abs/2506.10488)
- [Calvo-Zaragoza, Hajič & Pacha — Understanding OMR](https://arxiv.org/pdf/1908.03608)
- [Towards Musically Informed Evaluation of Piano Transcription Models, arXiv:2406.08454](https://arxiv.org/html/2406.08454v2)
- [Ycart, Liu, Benetos, Pearce — Investigating the Perceptual Validity of Evaluation Metrics for Automatic Piano Music Transcription, TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.57)
- [Simonetta et al. 2022 — Perceptual resynthesis measure](https://link.springer.com/article/10.1007/s11042-022-12476-0)
- [Frechet Audio Distance, Kilgour et al. 2019](https://www.isca-archive.org/interspeech_2019/kilgour19_interspeech.pdf)
- [microsoft/fadtk](https://github.com/microsoft/fadtk)
- [Adapting FAD for Generative Music Evaluation, arXiv:2311.01616](https://arxiv.org/abs/2311.01616)
- [LAION-CLAP](https://github.com/LAION-AI/CLAP)
- [audiocraft.metrics.clap_consistency](https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/metrics/clap_consistency.html)
- [MERT, Hugging Face](https://huggingface.co/m-a-p/MERT-v1-330M)
- [madmom paper, arXiv:1605.07008](https://arxiv.org/abs/1605.07008)
- [madmom docs](https://madmom.readthedocs.io/)
- [MSAF, Nieto & Bello 2015](https://ccrma.stanford.edu/~urinieto/MARL/publications/NietoBello-ISMIR2015.pdf)
- [Davies et al. 2014 — Evaluating Beat Tracking Measures, ISMIR](https://archives.ismir.net/ismir2014/paper/000238.pdf)
- [CREPE, Kim et al. 2018](https://arxiv.org/abs/1802.06182)
- [Lerdahl, Tonal Pitch Space](https://global.oup.com/academic/product/tonal-pitch-space-9780195178296)
- [Tonnetz, Wikipedia](https://en.wikipedia.org/wiki/Tonnetz)
- [librosa.feature.tonnetz](https://librosa.org/doc/main/generated/librosa.feature.tonnetz.html)
- [Pop2Piano, Choi et al. 2022, arXiv:2211.00895](https://arxiv.org/abs/2211.00895)
- [POP909, Wang et al. 2020](https://arxiv.org/pdf/2008.07142v1)
- [Nakamura & Sagayama — Statistical Piano Reduction, arXiv:1808.05006](https://arxiv.org/pdf/1808.05006)
- [Ramoneda et al. 2023, Score Difficulty, arXiv:2306.08480](https://arxiv.org/abs/2306.08480)
- [Difficulty-Aware Score Generation for Piano Sight-Reading, arXiv:2509.16913](https://arxiv.org/html/2509.16913)
- [Annotating Symbolic Texture in Piano Music: a Formal Syntax](https://hal.science/hal-03631151/file/main.pdf)
- [music21](https://music21.org/)
- [parangonar PyPI](https://pypi.org/project/parangonar/)
- [Survey on Evaluation Metrics for Music Generation, Kader 2025, arXiv:2509.00051](https://arxiv.org/html/2509.00051v1)
- [MMMOS, arXiv:2507.04094](https://arxiv.org/html/2507.04094v2)
- [Holzapfel et al. 2019 — AMT user study, ISMIR](https://archives.ismir.net/ismir2019/paper/000082.pdf)
- [Riley et al. 2024 — Robust AMT analysis, arXiv:2402.01424](https://arxiv.org/pdf/2402.01424)
- [MAESTRO dataset](https://magenta.tensorflow.org/maestro-wave2midi2wave)
- [Henle Difficulty Levels](https://www.henle.de/Levels-of-Difficulty/)

---

## Datasets for Audio→Piano-Sheet Systems (Pop-Music Focus)

**Author:** research agent for Oh Sheet
**Date:** 2026-04-25
**Scope:** Datasets relevant to (a) training, (b) finetuning, and (c) evaluating an audio→piano-sheet system, with emphasis on pop music applicability.

---

### TL;DR

Solo-piano data is plentiful and well-licensed for research (MAESTRO, MAPS, ASAP, SMD, GiantMIDI, Aria-MIDI, PiJAMA). **Paired pop-audio + piano-cover data is the bottleneck.** The only sizable public corpus that targets this exact task is the Pop2Piano paper's PSP corpus — and only the *model weights* and a tiny CI subset are released; the raw paired audio is not redistributed because of YouTube/copyright. POP909 is widely cited as "pop dataset" but is misleadingly named: it is **MIDI-only piano arrangements, not paired pop audio**. For evaluation today, the most defensible path is a hybrid: MAESTRO/ASAP for in-distribution piano AMT correctness, plus a small (~30-track) hand-curated YouTube pop-cover eval set held in-house under fair-use research, scored against publicly available human transcriptions where possible.

---

### 1. Comprehensive Comparison Table

| # | Dataset | Year | Size | Genre | Paired (audio+MIDI) | Audio quality | Alignment quality | License | Commercial OK? | Hosting | Pop-piano relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **MAESTRO v3** | 2018/2020 | ~200 hrs / ~1276 perf. / ~430 compositions | Classical solo piano | Yes (Disklavier ~3 ms) | Real acoustic, lossless | Excellent (Disklavier MIDI) | CC-BY-NC-SA 4.0 | No | magenta.tensorflow.org / GCS | **Low** (no pop) — but gold for piano AMT training |
| 2 | **MAPS** | 2010 | ~31 hrs / 270 pieces | Classical | Yes (synth + Disklavier subset) | Mixed (synth + real) | Excellent | CC BY-NC-SA 2.0 FR | No | INRIA (telecom-paris.fr) | **Low** — classical reference |
| 3 | **GiantMIDI-Piano** | 2020 | 10,855 unique works / ~1,237 hrs / 38.7M notes | Classical solo piano | MIDI only (transcribed by O&F) | N/A (audio not in dataset) | Noisy (auto-transcribed) | "Disclaimer" — research only; symbolic only redistributed | Murky | github.com/bytedance/GiantMIDI-Piano | **Low** — classical, MIDI-only |
| 4 | **Aria-MIDI** | 2025 | 1,186,253 MIDI files / ~100,629 hrs | Mixed (incl. some pop) | MIDI only (auto-transcribed via Aria-AMT) | N/A (audio not in dataset) | Noisy | Open release via HF | Likely yes (MIDI = facts) | huggingface.co/datasets/loubb/aria-midi | **Medium** — has multi-genre piano MIDI inc. some pop |
| 5 | **PiJAMA** | 2023 | 219.4 hrs / 2,777 perf. / 120 pianists / 244 albums | Solo jazz piano | MIDI only (auto-transcribed) | N/A | Noisy | Audio not redistributed; MIDI on Zenodo | Murky | almostimplemented.github.io/PiJAMA | **Low** (jazz) — but useful for stylistic generalization |
| 6 | **POP909** | 2020 | 909 songs / MIDI only | Chinese/Western pop | **MIDI only — no audio** | N/A | Hand-aligned beat/tempo, MIR-derived chords | **MIT** (per repo LICENSE) | **Yes** | github.com/music-x-lab/POP909-Dataset | **High (symbolic)** — pop melody+piano-arrangement structure |
| 7 | **Pop2Piano PSP corpus** | 2023 | ~5,503 paired songs (filtered from 5,844) / ~180 hrs | K-pop dominant + Western pop + hip hop | **YES — pop audio + piano cover MIDI** | YouTube (lossy) | Synchronization-pipeline, weak | Code MIT; **raw audio NOT released** (only CI test = 2 clips); model weights MIT | Model yes, data no | github.com/sweetcocoa/pop2piano | **Critical — but unavailable as data** |
| 8 | **PiCoGen / PiCoGen2** | 2024 | "weakly-aligned" pop/piano pairs + piano-only pretrain | Pop (5 sub-genres) | **YES** (weakly-aligned) | YouTube | Loose (intentional) | Code MIT; data redistribution restricted | Model yes, data no | github.com/tanchihpin0517/PiCoGen | **Critical — same gap as Pop2Piano** |
| 9 | **Slakh2100** | 2019 | 2,100 mixes / ~145 hrs / 187 instrument patches | Synthetic (rendered Lakh) | Yes (synth audio + MIDI) | Synth (high-quality VST) | Perfect (synthetic) | **CC-BY 4.0** | **Yes** | zenodo (record/4599666) | **Medium** — multi-track AMT training; pop-ish style but synthetic |
| 10 | **MUSDB18 / MUSDB18-HQ** | 2017/2019 | 150 tracks / ~10 hrs | Pop/rock primarily | Audio stems only — drums/bass/vocals/other (piano lumped in "other") | Real | N/A (no MIDI) | Educational only; tracks mostly CC BY-NC-SA | **No** | sigsep.github.io / zenodo | **Low** for transcription (no symbolic), useful for separation pre-step |
| 11 | **MoisesDB** | 2023 | 240 tracks / ~14h 24m / 47 artists / 12 genres | Multi-genre incl. pop | Audio stems only, 11-stem hierarchy (incl. piano) | Real | N/A | **CC-BY-NC-SA 4.0** | **No** | github.com/moises-ai/moises-db | **Medium** — supplies isolated piano stems from real pop tracks (no MIDI labels) |
| 12 | **MedleyDB v2** | 2014/2016 | 196 multitrack songs | Mixed, ~half pop/rock | Multitrack audio + melody f0 | Real | f0 annotations only | **CC BY-NC-SA 3.0** | **No** | medleydb.weebly.com | **Medium** — real pop multitrack but no piano-MIDI labels |
| 13 | **MusicNet** | 2017 | 330 recordings / ~34 hrs / >1M notes | Classical (incl. piano) | Yes (DTW-aligned) | Real (CC/Public Domain) | Good (DTW, ~4% error) | CC by source (CC0/CC-BY/PD mix); aggregate: research-permissive | Mixed | zenodo (record/5120004) | **Low** — classical only |
| 14 | **RWC Music Database (Popular)** | 2002+ | 100 songs (popular set) | Japanese pop (80) + American-style (20) | Audio + ground-truth chord/melody (no piano-roll) | Real (purpose-recorded) | High | **CC BY-NC 4.0** (post-2024 release) | **No** | staff.aist.go.jp/m.goto/RWC-MDB; zenodo | **Medium** for chord/melody eval, not piano transcription |
| 15 | **ASAP** | 2020 | 222 scores / 1,068 perf. / >92 hrs | Western classical piano | MIDI + MusicXML score-aligned, audio for subset | Real | Excellent (manual beat alignment) | Custom academic (per repo) | No | github.com/fosfrancesco/asap-dataset | **Low** for pop, **High** for AMT-with-score eval |
| 16 | **SMD** | 2011 | 50 pieces / ~261 min | Classical piano | Disklavier (audio+MIDI ~10 ms onset) | Real | Excellent | CC BY-NC-SA 3.0 | No | resources.mpi-inf.mpg.de/SMD | **Low** (classical) |
| 17 | **Lakh MIDI Dataset (LMD)** | 2016 | 176,581 MIDI / 45,129 matched to MSD | Mixed pop/rock heavy | MIDI only | N/A | N/A (varies) | **CC-BY 4.0** | **Yes** | colinraffel.com/projects/lmd | **High** (symbolic) — vast pop MIDI for symbolic-side training |
| 18 | **Million Song Dataset** | 2011 | 1M tracks (features + metadata only) | Pop-heavy | **No audio**, no MIDI | N/A | N/A | Free, mixed | Yes | millionsongdataset.com | Negligible for transcription |
| 19 | **FMA (Free Music Archive)** | 2017 | 106,574 tracks / ~343 days | Mixed; 161 genres | Audio only | Real, CC | N/A | Per-track CC; commercial-filterable subset available | **Yes (subset)** | github.com/mdeff/fma | Low for piano AMT but useful as legal pop-style audio source |
| 20 | **JSB Chorales** | (multiple) | ~389 chorales | Baroque 4-voice | MIDI/symbolic | N/A | Manual/score | Mixed (UCI: CC-BY 4.0) | Yes | UCI / github.com/czhuang | Negligible for pop |
| 21 | **GuitarSet** | 2018 | 360 excerpts / ~3 hrs | Acoustic guitar | Audio + hexaphonic + MIDI | Real | High (hex-pickup) | CC-BY 4.0 (per Zenodo) | Yes | github.com/marl/GuitarSet | Low for piano |
| 22 | **NSynth** | 2017 | 305,979 individual notes | Single-note instruments | Audio + pitch labels | Synth | Per-note | CC-BY 4.0 | Yes | magenta/tensorflow.org/datasets/nsynth | Negligible for AMT |
| 23 | **Bach10** | (older) | 10 chorales | Multi-instrument classical | Audio + per-instrument MIDI | Real | Good | CC BY-NC 4.0 (mf0-synth variant) | No | synthdatasets.weebly.com | Negligible |
| 24 | **MIR-1K** | 2009 | 1,000 clips / 110 karaoke songs | Chinese pop (singing) | Stereo split (vocal/accomp.) | Real | Pitch contour annotations | Free for research (mirlab.org) | No | zenodo/3532216 | Low (no MIDI labels) |
| 25 | **OpenMIC-2018** | 2018 | 20,000 × 10s clips / 20 instr. labels | FMA-derived | Audio + instrument tags | Real | Tags only (no time-aligned MIDI) | **CC BY 4.0** | **Yes** | github.com/cosmir/openmic-2018 | Low — instrument detection only |
| 26 | **NES-MDB** | 2018 | 5,278 songs / 397 NES games / 296 composers | Chiptune | MIDI w/ NES audio | Synth | Sample-accurate | Public domain (US) for MIDI; audio synthesized at runtime | Murky | github.com/chrisdonahue/nesmdb | Negligible |
| 27 | **PrIMuS** | 2018 | 87,678 sequences | Monophonic mensural/CWMN images | PNG + MIDI + MEI | N/A | Synthetic | Free for research | Murky | grfia.dlsi.ua.es/primus | OMR-as-eval only |
| 28 | **DeepScores** | 2018 | ~300,000 synthetic images | Multi-style print | MusicXML→PNG | N/A | Synthetic | CC by source | Murky | tuggeluk.github.io/deepscores | OMR-as-eval only |
| 29 | **MUSCIMA++** | 2017 | 91,255 symbols / 140 pages | Handwritten | Symbol-bbox annotations | N/A | Manual | "Permissive" (annotations); base CVC-MUSCIMA license | Murky | github.com/OMR-Research/muscima-pp | OMR handwriting eval |
| 30 | **HookTheory HLSD** | (ongoing) | 18,843 sections / metadata only | Pop heavy | Lead-sheet metadata only — no audio | N/A | Manual | TheoryTab ToS; metadata-only redistribution | No (TOS) | hooktheory.com | Symbolic harmony reference |
| 31 | **Bach Chorales (UCI)** | older | ~100 chorales | Baroque | Symbolic | N/A | Manual | CC-BY 4.0 | Yes | archive.ics.uci.edu/dataset/25 | Negligible |

#### Sources
- MAESTRO: Hawthorne et al. 2019 ICLR; magenta.tensorflow.org/maestro-wave2midi2wave; mirdata.readthedocs.io
- MAPS: Emiya & Bertin 2010 INRIA HAL inria-00544155; adasp.telecom-paris.fr/resources/2010-07-08-maps-database
- GiantMIDI-Piano: Kong et al. 2020/2022 ISMIR transactions/tismir.80; github.com/bytedance/GiantMIDI-Piano
- Aria-MIDI: Bradshaw & Colton ICLR 2025 (arxiv 2504.15071); huggingface.co/datasets/loubb/aria-midi
- PiJAMA: Edwards, Dixon, Benetos 2023 transactions/tismir.162
- POP909: Wang et al. ISMIR 2020 (arxiv 2008.07142); github.com/music-x-lab/POP909-Dataset (LICENSE: MIT)
- Pop2Piano: Choi & Lee ICASSP 2023 (arxiv 2211.00895); github.com/sweetcocoa/pop2piano; huggingface.co/datasets/sweetcocoa/pop2piano_ci (CI subset only)
- PiCoGen2: Tan et al. 2024 (arxiv 2408.01551); github.com/tanchihpin0517/PiCoGen
- Slakh2100: Manilow et al. WASPAA 2019 (arxiv 1909.08494); zenodo.org/records/4599666
- MUSDB18-HQ: zenodo.org/records/3338373; sigsep.github.io/datasets/musdb.html
- MoisesDB: Pereira et al. ISMIR 2023 (arxiv 2307.15913); github.com/moises-ai/moises-db
- MedleyDB v2: Bittner et al. ISMIR 2016 LBD; medleydb.weebly.com
- MusicNet: Thickstun et al. 2017; zenodo.org/records/5120004
- RWC: Goto et al. ISMIR 2002; staff.aist.go.jp/m.goto/RWC-MDB; transactions.ismir.net/articles/10.5334/tismir.326
- ASAP: Foscarin et al. ISMIR 2020; github.com/fosfrancesco/asap-dataset
- SMD: Müller et al. ISMIR 2011 LBD; resources.mpi-inf.mpg.de/SMD
- LMD: Raffel 2016 thesis; colinraffel.com/projects/lmd
- MSD: Bertin-Mahieux et al. ISMIR 2011; millionsongdataset.com
- FMA: Defferrard et al. ISMIR 2017 (arxiv 1612.01840); github.com/mdeff/fma
- GuitarSet: Xi et al. ISMIR 2018; zenodo.org/records/1492449
- NSynth: Engel et al. 2017; magenta.tensorflow.org/datasets/nsynth
- Bach10: Duan et al.; synthdatasets.weebly.com/bach10-mf0-synth.html
- MIR-1K: Hsu & Jang 2010; zenodo.org/records/3532216
- OpenMIC-2018: Humphrey, Durand, McFee ISMIR 2018; zenodo.org/records/1432913
- NES-MDB: Donahue et al. ISMIR 2018; github.com/chrisdonahue/nesmdb
- PrIMuS / DeepScores / MUSCIMA++: see github.com/apacha/OMR-Datasets
- HookTheory HLSD: hooktheory.com/theorytab; emergentmind.com/topics/hooktheory-dataset

---

### 2. Critical Gap: Pop-Audio + Piano-Cover Paired Data

**This is the dataset that would most improve Oh Sheet's pop transcription quality, and it does not exist as a redistributable corpus.**

#### What's been published
| Effort | Pairs | Released? |
|---|---|---|
| Pop2Piano (Choi & Lee, ICASSP 2023) | 5,844 collected → 5,503 after filtering → ~3,000 used in main results, ~180 hrs total | **Model weights MIT, raw paired audio NOT released**. Only a 2-clip CI test subset is on HF (`sweetcocoa/pop2piano_ci`). |
| PiCoGen2 (Tan et al., 2024) | "weakly-aligned" — exact count not in abstract | Code MIT; data redistribution restricted by source-track copyright |
| Yamaha Print Gakufu (commercial) | Unknown; J-pop YouTube + transcriptions | Proprietary |
| MuseScore community | ~340,000 user-uploaded songs with audio + MusicXML | Murky — user uploads with mixed claims; ToS restricts bulk scraping |
| AnthemScore / Klangio / Songscription | Proprietary | Not released; Songscription explicitly says "majority synthetic + licensed-from-musicians" |

#### Why pop pairs are scarce
1. **Audio is third-party copyrighted.** A "Despacito → piano cover" pair contains both Universal Music's master and the YouTuber's derivative work. Neither is the researcher's to redistribute.
2. **Piano covers are derivative works** that, while uploaded under YouTube ToS, are not relicensed CC.
3. **Alignment is hard.** Cover pianists rephrase, transpose, and shift bars; only weak/synchronization alignment is feasible without manual labor.
4. **DMCA + ToS risk.** Courts have begun to allow anti-circumvention claims (e.g., Panda-70M lawsuit, 2025) targeting *the act* of scraping rather than fair-use of contents.

#### Could Oh Sheet build its own?
- **Yes, technically.** Open scrape pipeline: query YouTube `<song name> piano cover` → `yt-dlp` → audio fingerprint match against original (`audfprint` Columbia, Dejavu Python, Panako) to verify pairing → run Basic Pitch / Aria-AMT on the cover → DTW-align cover MIDI to original audio chroma. This is essentially the Pop2Piano automated pipeline (Choi & Lee 2023, §3) and the PiCoGen2 weakly-aligned pipeline.
- **Realistic scale:** Pop2Piano scraped 5,844 raw → 5,503 usable in months. With one engineer ~2k–5k high-quality pairs in ~4–8 weeks is plausible.
- **Legal posture for a US-based commercial product:** Internal training-only use under fair-use research has precedent (Authors Guild v. Google, Bartz v. Anthropic 2025), but **publishing or redistributing** the corpus is high risk. A reasonable strategy: keep the corpus internal; release model + small CC eval set + open-source the scrape pipeline (so others can rebuild).
- **Mitigations:** prefer FMA-licensed source tracks (commercial-OK CC subset) when possible — smaller pop subset but legally clean; explicitly seek out CC-BY-licensed YouTube pianists (a few channels do this).

#### Open piano-cover scrapers / fingerprint tools that exist
- `audfprint` (Columbia / Dan Ellis): `scrape-yt-match-fprint.sh` does almost exactly the right thing.
- `worldveil/dejavu` — Python audio fingerprinting.
- `JorenSix/Panako` — robust to time-stretch and pitch shift (useful for pianist arrangements).
- Pop2Piano repo's `download/download.py` lists track IDs only.
- Synthesia-video → MIDI: `emilamaj/SynToMid`, `Adelost/piano-video-2-midi` (extract piano from falling-bar tutorials directly).

---

### 3. Best for Evaluating Oh Sheet (Top 3)

#### #1 — Hand-curated 30–50 song internal pop eval set + MAESTRO test split
- **Why:** Oh Sheet's product target is "pop song MP3/YouTube → playable sheet". No public dataset measures *that* end-to-end. A small, internally controlled eval set fixes that.
- **Build it:** 30–50 songs across genres (mainstream pop, hip-hop with piano, ballad, K-pop, indie). For each: (a) source audio file or YouTube URL; (b) human-verified reference MIDI (purchase from Hooktheory premium, MuseScore.com Pro, or contract a transcriber for ~$30/song = $1.5k); (c) reference PDF sheet. Score with mir_eval onset/offset F1, plus a downstream "playability" metric (e.g., max simultaneous notes per hand, hand-stretch violations).
- **Cite as supplement:** MAESTRO test split (CC-BY-NC-SA, ~20 hrs) for in-domain piano AMT regression — ensures changes don't degrade classical baseline.

#### #2 — POP909 (symbolic side) + selected MUSDB18-HQ tracks (audio side)
- **Why POP909:** 909 pop songs, MIT-licensed, melody+piano-arrangement MIDI. Use as a *symbolic-side* evaluation: given POP909 audio (you'd render or synth-source it), can the system reconstruct the canonical melody+piano arrangement? Cited very widely so results are comparable.
- **Why MUSDB18-HQ:** 150 tracks of real pop/rock with isolated stems (drums/bass/other/vocals). Even though piano is in "other", you can use the *separation* as a sanity input — feeding pre-separated mixes to test robustness.
- **Caveat:** POP909 ships only MIDI, not audio. Either render with high-quality VST (acceptable for pipeline eval; risks domain mismatch) or pair MIDI with the original copyrighted audio internally (not redistributable).

#### #3 — ASAP + SMD as alignment/notation correctness eval
- **Why:** Oh Sheet's last mile is *engraving* — turning MIDI into readable sheet. ASAP (CPJKU mirror) and SMD provide score-aligned ground truth so you can measure not just note F1 but *score correctness*: time signature, beat detection, voicing, hand-split. Both are CC-BY-NC-SA → fine for internal eval, not redistribution.
- **Caveat:** Classical only. Use only for engraving-side quality, not pop-style applicability.

---

### 4. Best for Finetuning a Pop→Piano Model (Top 3)

#### #1 — Pop2Piano model + self-collected pop-cover pairs (~2k–5k)
- Start from `sweetcocoa/pop2piano` weights (MIT, on HF). Replicate Pop2Piano's automated scrape/align pipeline on a fresh YouTube crawl. Filter via Pop2Piano's published thresholds (chroma accuracy ≥0.05, length-mismatch ≤15%). Keep corpus internal.
- **Why this and not training from scratch:** Pop2Piano weights have already absorbed the K-pop heavy distribution; finetuning on Western pop closes the genre gap.
- **Risk:** copyright (training-only fair-use posture); DMCA anti-circumvention risk if scraping bypasses controls.

#### #2 — Slakh2100 (CC-BY 4.0) for multi-instrument pretraining + POP909 (MIT) for arrangement structure
- **Slakh:** 145 hrs of synthetic multitrack (rendered Lakh MIDI) — clean license, perfect alignment, lots of pop-style instruments. Excellent for pretraining a multi-instrument decomposer (transcribe drums/bass/keys jointly) before specializing to piano-cover style.
- **POP909:** Use the melody+bridge+accompaniment structure as *symbolic supervision* — train a reduction head that maps a multi-instrument transcription to a 2-handed piano arrangement.
- **Both fully redistributable.**

#### #3 — MoisesDB-derived isolated-piano stems + Aria-MIDI/GiantMIDI piano performance prior
- **MoisesDB:** 240 real songs, 11-stem hierarchy *including* isolated piano. CC-BY-NC-SA 4.0 (research-only). Use isolated piano stems as an "in-the-wild pop-piano performance" target for self-supervised pretraining or as a domain-adaptation set for an audio→MIDI piano transcriber.
- **Aria-MIDI (1.18M files, ~100k hrs) or GiantMIDI:** symbolic pretraining for the piano-cover decoder so it learns plausible 2-handed pianistic patterns.
- **Caveat:** MoisesDB has no piano-MIDI ground truth — pseudo-labels via Aria-AMT or Onsets-and-Frames will be noisy.

---

### 5. Recommendations: Self-collected Data

If Oh Sheet is serious about pop-piano accuracy, building an internal corpus is unavoidable. A pragmatic plan:

#### Process (Pop2Piano-style pipeline, refined)
1. **Source-list curation.** Pull Billboard Hot-100, Spotify Top-50 (multiple regions), and any niches (anime, video-game, K-pop) that overlap user demand. Target 5,000 songs.
2. **Per song, scrape candidate piano covers.** YouTube query `"<title>" "<artist>" piano cover OR piano arrangement`. Take top-3 by view count.
3. **Audio-fingerprint match** the cover against the original master to confirm pairing (use `audfprint` or Panako; Panako is more robust to pitch/tempo shift).
4. **Auto-transcribe the cover** using Aria-AMT (best public piano AMT as of 2025) or Spotify `basic-pitch`.
5. **Synchronize** cover MIDI to original audio via DTW on chroma + onset features. (Pop2Piano calls this their PSP synchronization stage.)
6. **Filter**: chroma accuracy threshold, length difference ≤15%, instrument-purity (solo-piano detector e.g., GiantMIDI's CNN).
7. **Manual spot-check** ~5% to estimate label noise rate.

#### Scale targets
- **MVP** (4 weeks, 1 engineer): 1,000 pairs; sufficient to finetune Pop2Piano on Western pop and measurably beat zero-shot.
- **Production** (3 months): 5,000–10,000 pairs across genres + balanced demographic of pianists.
- **Stretch** (12 months, with paid annotators): 1,000 manually-verified gold pairs as eval set.

#### Legal posture
- **Training only**: keep corpus internal, never redistribute. US fair-use case law (Authors Guild v. Google; Bartz v. Anthropic 2025) is generally favorable to ML training on copyrighted material when the use is transformative. Risk is non-zero.
- **Avoid DMCA circumvention.** Use yt-dlp on public videos; do not bypass age gates, paywalls, or rate limits aggressively.
- **Honor robots.txt and ToS to the extent practical.** Document the lawful basis (research / fair use) per ingestion batch.
- **Eval set redistribution**: contract with transcribers for original work-for-hire transcriptions (you own the MIDI); pair them with public-domain or CC-licensed tracks where possible (FMA `commercial=true` subset). This gives a small (30–50 song) redistributable eval set that avoids licensing landmines.
- **Public artifacts:** model weights (likely OK), open-source scrape/align pipeline (low risk), the small CC eval set (low risk). Do **not** publish the scraped audio corpus.

#### Cost estimate
- 1 engineer × 2 months × ~$15k all-in = ~$30k for the pipeline + 5k pairs.
- 30 manually-verified eval transcriptions × ~$30/song = ~$900.
- GPU compute for Aria-AMT inference on 5k songs × ~5 min average × A100 = ~$200 on cloud.
- **Total ~$31k for an MVP with sane license hygiene.**

---

### 6. Red Flags (URLs/availability)

- **MAPS** original SFTP at INRIA Rennes has been flaky historically; current canonical pointer is `adasp.telecom-paris.fr/resources/2010-07-08-maps-database`.
- **MIR-1K** original `mirlab.org/dataset/public/MIR-1K.rar` URL is intermittent; Zenodo mirror at `zenodo.org/records/3532216` is the stable one.
- **GiantMIDI-Piano** — only the *symbolic* MIDI is redistributed (`193 MB stable version`); the audio-collection pipeline points at YouTube IDs and you must reconstruct audio yourself.
- **Pop2Piano** — the public HF dataset `sweetcocoa/pop2piano_ci` is **2 audio clips of ~10 s each**, not the training set. Do not assume it is usable as data.
- **HookTheory HLSD** — only metadata is distributed; the matched audio recordings must be procured separately and ToS restricts bulk download.
- **MAESTRO V2 → V3 incompatibility:** V3 dropped 6 erroneously-included recordings; downstream test sets pinned to V2 must be re-checked.

---

### 7. License Cheat-sheet (commercial product perspective)

Commercial-use friendly: **Lakh MIDI (CC-BY 4.0), Slakh2100 (CC-BY 4.0), POP909 (MIT), GuitarSet (CC-BY 4.0), NSynth (CC-BY 4.0), OpenMIC-2018 (CC-BY 4.0), MusicNet (mostly CC/PD), FMA (per-track, filterable subset OK).**

**Research-only (NC):** MAESTRO, MAPS, SMD, MedleyDB, MoisesDB, MUSDB18, RWC, PiJAMA, ASAP.

**Training-only ambiguous:** GiantMIDI-Piano (disclaimer-based), Aria-MIDI (HF-released, MIDI-as-facts argument), Pop2Piano (model MIT, data not released).

**Do not assume any public pop-paired-piano dataset has redistributable raw audio.** None do.

---

## Emerging & Experimental Techniques for Audio→Piano-Sheet Transcription (2024-2026)

**Audience:** Oh Sheet maintainers (current pipeline = Basic Pitch + heuristic arrangement)
**Goal:** Survey what is *new and interesting* (not what is already conservative SOTA), with honest maturity calls.

---

### Top 10 Emerging Techniques

#### 1. Aria-AMT + Aria-MIDI (synthetic-pretraining + scale)
- **Description:** EleutherAI's Whisper-style seq-to-seq piano transcriber, trained on the 1.18M-file/100K-hour Aria-MIDI dataset (which itself was bootstrap-transcribed from YouTube audio). The combination of large synthetic pretraining + DTW filtering yields a transcriber that's highly robust to "in-the-wild" recording conditions — the exact regime where Basic Pitch struggles. Inference engine is ~131x real-time on H100, batched and multi-GPU.
- **Citations:** Bradshaw & Spangher, *Aria-MIDI: A Dataset of Piano MIDI Files for Symbolic Music Modeling*, ICLR 2025 ([arxiv.org/abs/2504.15071](https://arxiv.org/abs/2504.15071)); ([github.com/EleutherAI/aria-amt](https://github.com/EleutherAI/aria-amt)); [Hugging Face dataset](https://huggingface.co/datasets/loubb/aria-midi)
- **Maturity:** Production-ready. Open weights, official inference engine.
- **Open-source status:** **GitHub repo with weights** (Apache-2.0 inference, open dataset).
- **Expected gain:** Very high. Aria-AMT was specifically trained for "realistic" / non-studio audio — exactly Oh Sheet's pop-music input distribution. Replaces Basic Pitch with no architectural changes.
- **Integration cost:** Low. Drop-in replacement for the Transcribe stage. Need GPU at inference.

#### 2. hFT-Transformer (Sony) — current MAESTRO leader
- **Description:** Hierarchical frequency-time Transformer. Two-level: frequency-axis Transformer encoder/decoder, then time-axis Transformer. Achieves 96.72% onset F1 on MAESTRO and 91.86% pedal F1.
- **Citations:** Toyama et al., ISMIR 2023 ([arxiv.org/abs/2307.04305](https://arxiv.org/html/2307.04305)); [github.com/sony/hFT-Transformer](https://github.com/sony/hFT-Transformer)
- **Maturity:** Production-ready, open weights.
- **Open-source status:** **GitHub repo with pretrained checkpoints**.
- **Expected gain:** High for studio piano; MAESTRO-trained, so weaker on lo-fi pop. Best paired with Aria-AMT's robustness style of training.
- **Integration cost:** Low. PyTorch checkpoints, simple inference API.

#### 3. PiCoGen2 / AMT-APC / Etude — pop-audio→piano-cover models (the closest to what Oh Sheet does)
- **Description:** A family of 2024-2025 models that *directly* take a pop song waveform and output a playable piano arrangement (not a literal transcription, but a piano-cover — which is what most users actually want). Three variants:
  - **PiCoGen2** (2024) — two-stage: extract melody+chord lead-sheet, then symbolic piano-cover model. No paired data required; transfer-learning on weakly-aligned data.
  - **AMT-APC** (2024) — fine-tunes hFT-Transformer on YouTube piano-covers. "Reproduces original tracks more accurately than any existing models."
  - **Etude** (2025) — three-stage Extract→strucTUralize→DEcode with simplified REMI tokens; subjectively rated comparable to human composers.
- **Citations:** Tan et al., *PiCoGen2*, 2024 ([arxiv.org/abs/2408.01551](https://arxiv.org/html/2408.01551)); Komiya & Fukuhara, *AMT-APC*, 2024 ([arxiv.org/abs/2409.14086](https://arxiv.org/abs/2409.14086)); [github.com/misya11p/amt-apc](https://github.com/misya11p/amt-apc); *Etude*, 2025 ([arxiv.org/abs/2509.16522](https://arxiv.org/abs/2509.16522))
- **Maturity:** Demo / paper + GitHub. AMT-APC is the most production-ready (built on hFT-Transformer).
- **Open-source status:** **GitHub repos with weights** (PiCoGen, AMT-APC). Etude code unclear at writing.
- **Expected gain:** **Highest of any single technique for Oh Sheet's use case.** This is literally the Pop-Audio→Piano problem.
- **Integration cost:** Medium. These replace the Transcribe + Arrange + Humanize stages with a single model. Architecturally simpler but loses pipeline modularity.

#### 4. Diffusion-based AMT: DiffRoll & Noise-to-Notes
- **Description:** Reframe transcription as conditional generation — denoise from Gaussian noise to a piano-roll/event-list, conditioned on spectrogram. Two key wins: (a) flexible speed-accuracy trade-off (sample more steps for harder passages), (b) **inpainting** capability — can fill in masked regions, useful for low-confidence patches. DiffRoll also uniquely supports unsupervised pretraining on unpaired piano rolls (just MIDI files, no audio).
- **Citations:** Cheuk et al., *DiffRoll*, ICASSP 2023 ([arxiv.org/abs/2210.05148](https://arxiv.org/abs/2210.05148)); [github.com/sony/DiffRoll](https://github.com/sony/DiffRoll); Maman et al., TASLP 2025 ([audiolabs-erlangen.de/.../2025_Diffusion](https://www.audiolabs-erlangen.de/content/05_fau/professor/00_mueller/03_publications/2025_MamanZMB_DiffusionMusic_TASLP_ePrint.pdf)); *Noise-to-Notes* (drum), Sept 2025 ([arxiv.org/abs/2509.21739](https://arxiv.org/abs/2509.21739))
- **Maturity:** Paper + GitHub for DiffRoll; N2N is paper-only.
- **Open-source status:** **GitHub repo with weights** (DiffRoll). N2N: paper-only.
- **Expected gain:** Moderate. Outperforms discriminative counterpart by 19 ppt in DiffRoll's eval, but discriminative SOTA (hFT-Transformer/Aria-AMT) is now stronger overall. Best as a *refinement* pass.
- **Integration cost:** Medium-high. Slow inference (many denoising steps).

#### 5. Music Foundation Models as encoders: MERT, MusicFM, Music Flamingo
- **Description:** Self-supervised audio encoders pretrained on huge unlabeled music corpora. MERT uses RVQ-VAE + CQT teachers; MusicFM uses BEST-RQ. Both produce features that beat raw spectrograms on downstream MIR tasks including transcription. Strategy: keep encoder frozen, train a small AMT head. NVIDIA's *Music Flamingo* (Nov 2025) is a music-specialist Audio LLM built on Audio Flamingo 3 — strong on theory-aware QA and lyrics WER (12.9% Chinese, 19.6% English).
- **Citations:** Li et al., *MERT*, ICLR 2024 ([arxiv.org/abs/2306.00107](https://arxiv.org/abs/2306.00107)); Won et al., *MusicFM* ([github.com/minzwon/musicfm](https://github.com/minzwon/musicfm)); *Music Flamingo*, Nov 2025 ([arxiv.org/abs/2511.10289](https://arxiv.org/abs/2511.10289), [research.nvidia.com/labs/adlr/MF/](https://research.nvidia.com/labs/adlr/MF/)); *Do Foundational Audio Encoders Understand Music Structure?*, Dec 2025 ([arxiv.org/abs/2512.17209](https://arxiv.org/html/2512.17209v1))
- **Maturity:** Encoders are production-ready. Music Flamingo is research-grade.
- **Open-source status:** **GitHub repos with weights** for MERT and MusicFM. Music Flamingo: open per NVIDIA ADLR.
- **Expected gain:** High *if* used for non-piano sub-tasks (chord, beat, structure, melody) — those are where Oh Sheet's heuristic stack is weakest.
- **Integration cost:** Medium. Need to train MIR heads, but datasets exist.

#### 6. Score-Informed Refinement: Score-HPT and Score-Transformer
- **Description:** Take *any* AMT system's output and refine selected attributes (especially MIDI velocity, which all current AMT systems handle poorly) with a small Transformer or BiLSTM correction module attached to the base AMT velocity branch. Score-HPT achieves SOTA velocity with only 1M parameter overhead.
- **Citations:** *Score-Informed Transformer for Refining MIDI Velocity in AMT*, Aug 2025 ([arxiv.org/abs/2508.07757](https://arxiv.org/abs/2508.07757)); Score-Informed BiLSTM ([researchgate](https://www.researchgate.net/publication/394439634))
- **Maturity:** Paper, code expected.
- **Open-source status:** Paper-only at time of writing; modular enough to reimplement.
- **Expected gain:** Moderate but **focused** — fixes the dynamics/feel problem that hurts engraving readability and humanization quality.
- **Integration cost:** Low. ~1M extra params, modular post-processor.

#### 7. Cascade w/ specialist transformer modules: Beat This!, Mel-RoFormer, BTC
- **Description:** Replace the venerable `madmom`/`Btc-Btc` heuristic stack with 2024 transformer-class specialists:
  - **Beat This!** (ISMIR 2024) — transformer beat tracker, no DBN postprocessing, accurate cross-genre.
  - **Mel-RoFormer** (Sept 2024) — vocal separation + vocal melody transcription; SOTA on MIR-ST500 and POP909.
  - **BTC + GPT-4o chord chain-of-thought** (2025) — 5-stage CoT for chord recognition, +1-2.77% MIREX accuracy.
- **Citations:** Foscarin et al., *Beat This!*, ISMIR 2024 ([github.com/CPJKU/beat_this](https://github.com/CPJKU/beat_this)); Wang et al., *Mel-RoFormer*, 2024 ([arxiv.org/abs/2409.04702](https://arxiv.org/pdf/2409.04702)); *Enhancing ACR through LLM CoT*, 2025 ([arxiv.org/abs/2509.18700](https://arxiv.org/html/2509.18700v1))
- **Maturity:** Production-ready (Beat This!, Mel-RoFormer); experimental (LLM-CoT chord).
- **Open-source status:** **GitHub repos with weights**.
- **Expected gain:** Very high for engraving-friendly bar lines and melody/chord layer separation in pop. Direct upgrade path.
- **Integration cost:** Low. Each is a drop-in module replacement.

#### 8. Symbolic LLMs for Arrangement & Humanization
- **Description:** Pass a transcribed MIDI/ABC to a music-specialist LLM that produces a polished piano arrangement. Active models:
  - **ChatMusician** (LLaMA-2 + ABC, ACL 2024) — composes/arranges in ABC notation, beats GPT-4 baseline.
  - **NotaGen** (2025) — classical score generator emphasizing voice arrangement.
  - **MIDI-LLM** (NeurIPS AI4Music 2025) — extends LLaMA vocab with MIDI tokens, uses Anticipatory Music Transformer's arrival-time tokenization (no beat-sync required).
  - **Anticipatory Music Transformer** — controllable infilling; can be conditioned on a sparse melody+chord skeleton to fill in piano accompaniment.
- **Citations:** Yuan et al., *ChatMusician*, ACL 2024 ([arxiv.org/abs/2402.16153](https://arxiv.org/abs/2402.16153)); *NotaGen*, 2025 ([arxiv.org/abs/2502.18008](https://arxiv.org/html/2502.18008v1)); *MIDI-LLM*, 2025 ([arxiv.org/abs/2511.03942](https://arxiv.org/html/2511.03942v1)); Thickstun et al., *Anticipatory Music Transformer*, TMLR 2024 ([arxiv.org/abs/2306.08620](https://arxiv.org/abs/2306.08620), [github.com/jthickstun/anticipation](https://github.com/jthickstun/anticipation))
- **Maturity:** ChatMusician + AMT are production-ready; MIDI-LLM/NotaGen are research-grade.
- **Open-source status:** **GitHub repos with weights** (all four).
- **Expected gain:** High for *Humanize* and *Arrange* stages. Particularly strong for pop, where arrangement is more about feel than accuracy.
- **Integration cost:** Medium. LLM inference cost; prompt engineering.

#### 9. Score-Conditioned Diffusion: Music ControlNet, MusicLDM-arrangement, D3PIA
- **Description:** Latent-diffusion music generators with time-varying conditioning:
  - **Music ControlNet** (TASLP 2024) — pixel-wise time-varying control of melody/dynamics/rhythm.
  - **Multi-Track MusicLDM** (EAI ArtsIT 2024) — generate any subset of tracks (e.g., piano given bass+drums).
  - **D3PIA** (2025) — discrete denoising diffusion conditioned on lead sheet (melody+chord).
  - **Structured Multi-Track Accompaniment Arrangement** (NeurIPS 2024).
- **Citations:** *Music ControlNet*, TASLP 2024 ([dl.acm.org/doi/10.1109/TASLP.2024.3399026](https://dl.acm.org/doi/10.1109/TASLP.2024.3399026)); *Multi-Track MusicLDM*, 2024 ([arxiv.org/abs/2409.02845](https://arxiv.org/html/2409.02845v1)); *D3PIA*, 2025; Zhao et al., NeurIPS 2024 ([github.com/zhaojw1998/Structured-Arrangement-Code](https://github.com/zhaojw1998/Structured-Arrangement-Code))
- **Maturity:** Mostly research, some demos.
- **Open-source status:** Mixed — Multi-Track MusicLDM and Structured Arrangement have **GitHub repos with weights**.
- **Expected gain:** Speculative for sheet-music; most outputs are audio (waveform), not symbolic. *Not yet ready as a primary engine.*
- **Integration cost:** High; outputs need re-transcription.

#### 10. Voice/Staff GNN + Hierarchical Audio-to-Score
- **Description:** Two complementary symbolic post-stages:
  - **Cluster-and-Separate GNN** (2024) — uses graph neural network for assigning notes to voices and staves; replaces brittle heuristics in engraving step.
  - **End-to-End Polyphonic Piano Audio-to-Score with Hierarchical Decoding** (IJCAI 2024) — bypasses MIDI-to-score by going directly audio→hierarchical-score (bar-level + note-level).
- **Citations:** Karystinaios & Widmer, *Cluster and Separate*, 2024 ([arxiv.org/abs/2407.21030](https://arxiv.org/html/2407.21030v1)); *End-to-End Real-World Polyphonic Piano Audio-to-Score Transcription*, IJCAI 2024 ([ijcai.org/proceedings/2024/0862](https://www.ijcai.org/proceedings/2024/0862))
- **Maturity:** GNN is published, code likely available. Hierarchical audio-to-score is paper-stage.
- **Open-source status:** Mixed.
- **Expected gain:** High for engraving readability — these solve the "MIDI-to-readable-score" gap that abc2ly/MuseScore handles imperfectly.
- **Integration cost:** Medium for the GNN, integrates with current Engrave stage.

---

### Honest "Not Yet Ready" Calls

| Idea | Why Not Yet |
|---|---|
| **Multimodal LLMs (Gemini 2.5/3, GPT-4o, Qwen2.5-Omni) directly producing notation tokens** | These models are good at speech transcription, music *captioning*, and music-theory QA, but Gemini explicitly notes "non-speech audio: only English-language speech responses". GPT-4 in tests "performs marginally better than random in music reasoning". Even Music Flamingo, the strongest music-LALM, focuses on caption-style understanding and lyric WER, not symbolic-note transcription. **Wait 6-12 months.** ([CMI-Bench, 2025](https://arxiv.org/html/2506.12285v1) shows audio→symbolic transcription performance "drops significantly" for current Audio LLMs.) |
| **Mamba/state-space backbone for AMT** | Mamba is competitive with Transformers in speech, but no published AMT/piano-transcription paper using Mamba beats hFT-Transformer or Aria-AMT yet. Worth tracking; not worth integrating today. |
| **EnCodec/DAC/SoundStream tokens → LLM-transcriber** | Acoustic codec tokens are heavy (DAC: 900 tokens/sec/9 quantizers) and *acoustic*, not *symbolic*. Recent work (WavTokenizer, 2024) is improving compression but no transcription system yet outperforms the spectrogram → seq2seq paradigm. |
| **RLHF on piano-cover quality** | MusicRL (DeepMind, 2024) showed it works for text-to-music *audio*, but no published RLHF system for piano-cover *symbolic* output exists. Requires 100k+ pairwise preferences — costly. |
| **Active learning loops** | Sound idea but operationally expensive for a small team. Pseudo-labeling (ISMIR 2024, [github.com/groupmm/onsets_frames_semisup](https://github.com/groupmm/onsets_frames_semisup)) gives most of the benefit at lower cost. |
| **Pure test-time compute / self-consistency for AMT** | Has been studied for Audio LLM cognition tasks ([arxiv.org/abs/2503.23395](https://arxiv.org/html/2503.23395)) but not for note-level AMT. Diffusion AMT (#4) is a smarter way to spend compute at test time — it gives you principled ensemble-by-sampling. |

---

### Speculative-but-Promising Hybrid Stacks

#### Stack A: "Drop-In Pop-First" (lowest engineering risk)
```
YouTube/MP3
  └→ Demucs htdemucs (vocal/bass/drums/other separation)
      └→ AMT-APC (hFT-Transformer fine-tuned on piano covers)
              ↳ Beat This! transformer for barlines
              ↳ Mel-RoFormer for vocal melody track
              ↳ BTC + GPT-4o chord-CoT for chord progression
          └→ Cluster-and-Separate GNN for voice/staff
              └→ Score-HPT velocity refinement
                  └→ MusicXML / LilyPond engraving
```
**Why it works:** Every component has open weights and is already proven. Substantively replaces Basic Pitch with a model literally trained on YouTube piano covers (the closest data distribution to what Oh Sheet's users want). Modular — keeps Oh Sheet's pipeline architecture. Practical RAM/GPU envelope.

#### Stack B: "Foundation + Specialist Heads" (medium risk, future-proof)
```
Audio
  └→ MERT-large or MusicFM (frozen 30s-context music encoder)
      ├→ Onset/pitch head (fine-tuned, à la "Do FAEs Understand Music Structure?" 2025)
      ├→ Beat/downbeat head
      ├→ Chord head
      └→ Structure (verse/chorus) head
  → Symbolic merge → Anticipatory Music Transformer for piano voicing/humanization
  → MusicXML
```
**Why it's interesting:** Single shared encoder reduces compute 4x vs. running specialists separately. As MERT/MusicFM scale up, all heads improve at once. AMT (Anticipatory Music Transformer) is uniquely good at *infilling* — given melody+chord skeleton, it generates a tasteful piano arrangement with controllable density.

#### Stack C: "Generative AMT + LLM Polish" (highest risk, highest ceiling)
```
Audio
  └→ Aria-AMT (robust seq-to-seq transcription)
  └→ DiffRoll refinement pass (sample N times, reconcile via majority vote)
      → Raw piano-roll
  → ABC notation conversion
  → ChatMusician (LLaMA-2 ABC) for arrangement polish: "make this playable for an intermediate pianist, preserve the melody, simplify left hand"
  → NotaGen for final voice-aware engraving
```
**Why it's intriguing:** Uses 2025-era SOTA at every step. The DiffRoll refinement is the speculative bit — it gives you self-consistency for free, and lets you trade compute for accuracy on demand. The ChatMusician arrangement step is exactly the *kind* of skill that previously required hand-coded rules.
**Risk:** ChatMusician has known weaknesses on complex modern music; ABC tokenization is lossy.

---

### Quick-Reference Reading List (most important if you only read 5)

1. *Aria-MIDI* (ICLR 2025) — [arxiv.org/abs/2504.15071](https://arxiv.org/abs/2504.15071) — the dataset + transcriber that changes the game
2. *AMT-APC* (2024) — [arxiv.org/abs/2409.14086](https://arxiv.org/abs/2409.14086) — most directly applicable model
3. *PiCoGen2 / Etude* (2024-2025) — [arxiv.org/abs/2408.01551](https://arxiv.org/html/2408.01551), [arxiv.org/abs/2509.16522](https://arxiv.org/abs/2509.16522) — what "good" piano-cover output looks like
4. *Beat This!* (ISMIR 2024) — [github.com/CPJKU/beat_this](https://github.com/CPJKU/beat_this) — modernized barlines
5. *Music Flamingo / CMI-Bench* (2025) — [research.nvidia.com/labs/adlr/MF/](https://research.nvidia.com/labs/adlr/MF/), [arxiv.org/abs/2506.12285](https://arxiv.org/html/2506.12285v1) — sober view of where Audio LLMs *don't* yet help

---

# Document end

_End of deliverable. Source reports preserved at `/tmp/oh-sheet-research/` for reproducibility. 13 specialist Opus 4.7 agents across 2 phases (10 discovery, 3 cross-pollination)._
