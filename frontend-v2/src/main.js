/**
 * App bootstrap — wires api.js (pure HTTP) + state.js (pure reducer) +
 * views.js (pure DOM) together.
 *
 * Flow:
 *   user fills form → views emits onSubmit(formData)
 *     → if audio/midi: api.uploadAudio/uploadMidi, then api.createJob with the ref
 *     → if title/youtube: api.createJob with title
 *     → api.subscribeToJob opens WS, pipes events through state.reduceJobEvent
 *     → store notifies → views re-renders the new phase + mascot swaps
 */

import * as api from "./api.js";
import { mountLegalDisclaimer } from "./legal.js";
import { createStore, reduceJobEvent } from "./state.js";
import { renderPhase } from "./views.js";

// Mascot file per phase/stage — same seven assets we extracted from
// origin/gau-81-png-to-svg and embed in the cream mockup.
const MASCOT_FOR = {
  idle: "/mascots/mascot-home-happy.svg",
  "idle:audio": "/mascots/mascot-home-happy.svg",
  "idle:midi": "/mascots/mascot-progress-arrange.svg",
  "idle:youtube": "/mascots/mascot-progress-ingest.svg",
  submitting: "/mascots/mascot-progress-ingest.svg",
  "working:ingest": "/mascots/mascot-progress-ingest.svg",
  "working:transcribe": "/mascots/mascot-progress-transcribe.svg",
  "working:arrange": "/mascots/mascot-progress-arrange.svg",
  "working:engrave": "/mascots/mascot-progress-engrave.svg",
  complete: "/mascots/mascot-success.svg",
  error: "/mascots/mascot-error.svg",
};

function mascotFor(phase) {
  if (phase.name === "working") return MASCOT_FOR[`working:${phase.stage}`];
  if (phase.name === "idle" && phase.source) {
    return MASCOT_FOR[`idle:${phase.source}`] || MASCOT_FOR.idle;
  }
  return MASCOT_FOR[phase.name];
}

const card = document.getElementById("card");
const mascot = document.getElementById("mascot");
let lastMascot = null;
function swapMascot(phase) {
  const src = mascotFor(phase);
  if (!src || !mascot || lastMascot === src) return;
  lastMascot = src;
  mascot.classList.add("swap");
  setTimeout(() => {
    mascot.src = src;
    mascot.classList.remove("swap");
  }, 250);
}

const store = createStore();
let unsubscribeWs = null; // close the previous job's WS on each re-submit

// Every phase change: morph the card, swap the mascot, re-render body,
// and mark body[data-phase] so CSS can hide/show chrome based on phase
// (e.g. the "Turn any song..." tagline is irrelevant once the job's
// done — target `body[data-phase="complete"]` in style.css to hide it).
store.onChange((phase) => {
  card.classList.add("morph");
  setTimeout(() => card.classList.remove("morph"), 450);
  swapMascot(phase);
  document.body.setAttribute("data-phase", phase.name);
  renderPhase(card, phase, handlers);
});

// ── Handlers bridge views → api ─────────────────────────────────
const handlers = {
  onSourceChange(source) {
    // If the user tabs between sources while a job is in flight, the
    // old WS would otherwise keep delivering events after we've
    // already transitioned back to idle — and a late job_succeeded
    // would clobber the fresh idle state with a stale Complete phase.
    // Close the socket first so the subscriber is a no-op.
    if (unsubscribeWs) { unsubscribeWs(); unsubscribeWs = null; }
    store.setPhase({ name: "idle", source });
  },

  onRetry() {
    if (unsubscribeWs) { unsubscribeWs(); unsubscribeWs = null; }
    const phase = store.getPhase();
    const source = phase.source || "youtube";
    store.setPhase({ name: "idle", source });
  },

  async onSubmit(formData) {
    store.setPhase({ name: "submitting" });
    try {
      let jobPayload = {};
      // Upload step for audio/midi before creating the job
      if (formData.source === "audio") {
        const ref = await api.uploadAudio(formData.file);
        jobPayload = { audio: ref, title: formData.title, artist: formData.artist };
      } else if (formData.source === "midi") {
        const ref = await api.uploadMidi(formData.file);
        jobPayload = { midi: ref, title: formData.title, artist: formData.artist };
      } else if (formData.source === "youtube") {
        jobPayload = {
          title: formData.url,
          artist: formData.artist,
          prefer_clean_source: true,
        };
      }

      const job = await api.createJob(jobPayload);

      // Start listening for job events BEFORE we assume progress; the
      // first event may race with the HTTP response.
      unsubscribeWs = api.subscribeToJob(job.job_id, async (event) => {
        const next = reduceJobEvent(store.getPhase(), event);
        store.setPhase(next);

        // Terminal state: refetch job to get full result payload (the
        // job_succeeded event's data may not include the full artifacts
        // depending on backend impl).
        if (event.type === "job_succeeded") {
          try {
            const full = await api.getJob(job.job_id);
            store.setPhase({ name: "complete", job: full });
          } catch { /* keep current complete phase */ }
          if (unsubscribeWs) { unsubscribeWs(); unsubscribeWs = null; }
        }
        if (event.type === "job_failed") {
          if (unsubscribeWs) { unsubscribeWs(); unsubscribeWs = null; }
        }
      });

      // If WS is slow to fire, still transition out of "submitting" into
      // a minimal "working:ingest" within 2s so the user doesn't stare
      // at a spinner.
      setTimeout(() => {
        const p = store.getPhase();
        if (p.name === "submitting") {
          store.setPhase({ name: "working", stage: "ingest", progress: 0 });
        }
      }, 2000);
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      store.setPhase({ name: "error", message: msg, retryable: true });
    }
  },
};

// Initial render
store.setPhase({ name: "idle", source: "youtube" });
mountLegalDisclaimer();
