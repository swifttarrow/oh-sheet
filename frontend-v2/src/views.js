/**
 * views.js — pure renderer for the morphing card body.
 *
 * Public API:
 *   renderPhase(container, phase, handlers)
 *
 * No innerHTML / no template strings for HTML — the repo's security hook
 * rejects innerHTML for user-facing content. All DOM construction goes
 * through document.createElement / createElementNS + appendChild.
 * Atomic swaps use container.replaceChildren().
 *
 * Does NOT import api.js or state.js — takes a Phase + handlers and
 * populates the given container. Mascot rendering is the app shell's
 * responsibility; views.js only fills the card body.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * Matches a YouTube *video* URL. Intentionally permissive on prefix
 * (http/https optional, www optional), strict on the video path —
 * must be either `watch?v=<id>` or `youtu.be/<id>` where `<id>` is
 * the 11-character YouTube ID charset.
 *
 * Rejects (returns false for these, so we can warn the user BEFORE
 * kicking off a 2-3 min backend roundtrip that would fail):
 *   - search pages:     /results?search_query=...
 *   - channel pages:    /channel/... or /@handle
 *   - playlists:        /playlist?list=...
 *   - other sites:      spotify, soundcloud, etc.
 *   - empty / garbage
 */
const YOUTUBE_VIDEO_RE =
  /^(?:https?:\/\/)?(?:www\.|m\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)[\w-]+/i;

function isYoutubeVideoUrl(s) {
  return typeof s === "string" && YOUTUBE_VIDEO_RE.test(s.trim());
}

const SOURCES = [
  { key: "audio", label: "Audio" },
  { key: "midi", label: "MIDI" },
  { key: "youtube", label: "YouTube" },
];

const STAGES = [
  { key: "ingest", name: "Fetch", icon: "cloud_download" },
  { key: "transcribe", name: "Transcribe", icon: "graphic_eq" },
  { key: "arrange", name: "Arrange", icon: "queue_music" },
  { key: "engrave", name: "Engrave", icon: "menu_book" },
];

// ── tiny DOM helpers ─────────────────────────────────────────────────

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  if (props) {
    for (const key of Object.keys(props)) {
      const value = props[key];
      if (value == null) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "style" && typeof value === "object") {
        Object.assign(node.style, value);
      } else node.setAttribute(key, value);
    }
  }
  for (const child of children) appendChild(node, child);
  return node;
}

function appendChild(parent, child) {
  if (child == null || child === false) return;
  if (Array.isArray(child)) {
    for (const c of child) appendChild(parent, c);
    return;
  }
  if (typeof child === "string") {
    parent.appendChild(document.createTextNode(child));
    return;
  }
  parent.appendChild(child);
}

function icon(name) {
  return el("span", { class: "material-symbols-rounded", text: name });
}

// ── sub-components ───────────────────────────────────────────────────

function segmented(activeSource, onSourceChange) {
  const wrap = el("div", { class: "segmented" });
  for (const src of SOURCES) {
    const isActive = src.key === activeSource;
    const btn = el(
      "button",
      { class: "seg" + (isActive ? " on" : ""), type: "button" },
      src.label,
    );
    btn.addEventListener("click", () => {
      if (!isActive) onSourceChange(src.key);
    });
    wrap.appendChild(btn);
  }
  return wrap;
}

function field(placeholder, iconName) {
  const wrap = el("div", { class: "field" + (iconName ? " with-icon" : "") });
  if (iconName) {
    const ic = icon(iconName);
    ic.classList.add("field-icon");
    wrap.appendChild(ic);
  }
  const input = el("input", { type: "text", placeholder });
  wrap.appendChild(input);
  return { wrap, input };
}

function filePickerButton(accept, labelText, onPick) {
  const input = el("input", {
    type: "file",
    accept,
    style: { display: "none" },
  });
  const label = el("span", { class: "file-label", text: labelText });
  const btn = el(
    "button",
    { class: "btn-outlined", type: "button" },
    icon(accept.includes(".mid") ? "piano" : "attach_file"),
    label,
    input,
  );
  btn.addEventListener("click", (e) => {
    if (e.target !== input) input.click();
  });
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) {
      label.textContent = input.files[0].name;
      onPick(input.files[0]);
    }
  });
  return { btn, input };
}

function spacer(height = 12) {
  return el("div", { style: { height: `${height}px` } });
}

// ── CTA pulse helper ─────────────────────────────────────────────────

function triggerCtaPulse(btn) {
  if (!btn) return;
  btn.classList.remove("pulse");
  void btn.offsetWidth; // reflow to restart animation
  btn.classList.add("pulse");
}

const YT_RE = /^https?:\/\/(www\.|music\.|m\.)?youtu(\.be\/|be\.com\/watch\?v=)([\w-]{11})/;

function tryPasteYoutube(input) {
  if (!input || typeof navigator === "undefined" ||
      !navigator.clipboard || !navigator.clipboard.readText) return;
  navigator.clipboard.readText().then((text) => {
    if (text && YT_RE.test(text.trim()) && !input.value) {
      input.value = text.trim();
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }).catch(() => {
    // Clipboard API unavailable or denied — silently ignore.
  });
}

// ── phase bodies ─────────────────────────────────────────────────────

function idleBody(source, handlers) {
  const wrap = el("div", { class: "body" });
  wrap.appendChild(segmented(source, handlers.onSourceChange));

  let fileInput = null;
  let selectedFile = null;
  let urlInput = null;
  let urlError = null;
  let titleInput = null;
  let artistInput = null;
  let submitBtn = null; // forward ref for pulse trigger

  if (source === "audio" || source === "midi") {
    const isMidi = source === "midi";
    const accept = isMidi
      ? ".mid,.midi"
      : ".mp3,.wav,.flac,.m4a,audio/*";
    const labelText = isMidi
      ? "Tap or drag MIDI file here"
      : "Tap or drag audio file here";
    const picker = filePickerButton(accept, labelText, (file) => {
      selectedFile = file;
      triggerCtaPulse(submitBtn);
    });
    fileInput = picker.input;
    wrap.appendChild(picker.btn);
    wrap.appendChild(spacer(4));
    wrap.appendChild(
      el("div", {
        class: "drop-hint",
        style: {
          textAlign: "center",
          fontSize: "12px",
          color: "var(--on-surface-variant)",
          fontWeight: "500",
        },
        text: isMidi ? ".mid \u00b7 .midi" : "mp3 \u00b7 wav \u00b7 flac \u00b7 m4a",
      }),
    );
    wrap.appendChild(spacer(12));
    const t = field("Title (optional)");
    const a = field("Artist (optional)");
    titleInput = t.input;
    artistInput = a.input;
    wrap.appendChild(t.wrap);
    wrap.appendChild(a.wrap);
  } else {
    // youtube (default — also the fallback for any unknown source
    // value, since the segmented picker only exposes audio/midi/
    // youtube after PR #83's UI simplification)
    const u = field("YouTube URL", "play_circle");
    const a = field("Artist (optional)");
    urlInput = u.input;
    artistInput = a.input;
    wrap.appendChild(u.wrap);
    // Inline error slot — stays empty until the user submits an invalid
    // URL; then shows a coral hint below the URL input. Clearing on
    // input keystroke gives users immediate feedback that the error
    // has been acknowledged.
    urlError = el("div", {
      class: "field-error",
      style: {
        fontSize: "12px",
        color: "var(--error)",
        margin: "-8px 0 12px 20px",
        display: "none",
        fontWeight: "500",
      },
    });
    wrap.appendChild(urlError);
    urlInput.addEventListener("input", () => {
      if (urlError) {
        urlError.style.display = "none";
        urlError.textContent = "";
      }
    });
    wrap.appendChild(a.wrap);
    // Auto-paste from clipboard if it contains a YouTube URL.
    if (!urlInput.value) tryPasteYoutube(urlInput);
  }

  wrap.appendChild(spacer(8));
  const submit = el(
    "button",
    { class: "btn-filled", type: "button" },
    icon("play_arrow"),
    "Let's go!",
  );
  submitBtn = submit;
  submit.addEventListener("click", () => {
    const payload = { source };
    if (source === "audio" || source === "midi") {
      // Prefer the captured File; fall back to reading the live input.
      const file =
        selectedFile ||
        (fileInput && fileInput.files && fileInput.files[0]) ||
        null;
      if (!file) return; // silently refuse — no file means no submit
      payload.file = file;
      if (titleInput && titleInput.value) payload.title = titleInput.value;
      if (artistInput && artistInput.value) payload.artist = artistInput.value;
    } else {
      // youtube (and fallback for any unknown source)
      const url = urlInput && urlInput.value.trim();
      if (!url) return;
      // Client-side guard: non-video YouTube URLs (search pages,
      // channels, playlists) and non-YouTube URLs would all fail
      // minutes later at the yt-dlp stage. Catch them now.
      if (!isYoutubeVideoUrl(url)) {
        if (urlError) {
          urlError.textContent =
            "Paste a video link, not a search or channel page " +
            "(e.g. https://youtu.be/abc123XYZ_).";
          urlError.style.display = "block";
        }
        if (urlInput) urlInput.focus();
        return;
      }
      payload.url = url;
      if (artistInput && artistInput.value) payload.artist = artistInput.value;
    }
    handlers.onSubmit(payload);
  });
  wrap.appendChild(submit);
  return wrap;
}

function submittingBody() {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "m3-spinner");
  svg.setAttribute("viewBox", "0 0 48 48");
  const circle = document.createElementNS(SVG_NS, "circle");
  circle.setAttribute("cx", "24");
  circle.setAttribute("cy", "24");
  circle.setAttribute("r", "20");
  svg.appendChild(circle);

  const note = el("p", {
    class: "note",
    style: { margin: "0" },
    text: "Uploading and queuing your job\u2026",
  });
  return el("div", { class: "body" }, svg, note);
}

function stageIndex(stage) {
  const idx = STAGES.findIndex((s) => s.key === stage);
  return idx === -1 ? 0 : idx;
}

function workingBody(stage, progress) {
  const active = stageIndex(stage);
  const stepper = el("div", { class: "stepper" });
  STAGES.forEach((s, i) => {
    const cls =
      i < active ? "step done" : i === active ? "step active" : "step";
    const pillIcon = i < active ? "check_circle" : s.icon;
    stepper.appendChild(
      el("div", { class: cls }, icon(pillIcon), s.name),
    );
  });

  const pct = Math.round((progress || 0) * 100);
  const bar = el(
    "div",
    { class: "bar" },
    el("span", { style: { width: `${pct}%` } }),
  );
  const meta = el(
    "div",
    { class: "meta" },
    el("span", { text: `${STAGES[active].name}\u2026` }),
    el("span", { text: `${pct}%` }),
  );
  return el("div", { class: "body" }, stepper, bar, meta);
}

function downloadChips(jobId) {
  const dl = el("div", { class: "downloads" });
  const kinds = [
    { kind: "pdf", label: "PDF", ic: "picture_as_pdf" },
    { kind: "musicxml", label: "MusicXML", ic: "description" },
    { kind: "midi", label: "MIDI", ic: "piano" },
  ];
  for (const k of kinds) {
    dl.appendChild(
      el(
        "a",
        {
          class: "assist-chip",
          href: `/v1/artifacts/${jobId}/${k.kind}`,
          download: "",
        },
        icon(k.ic),
        k.label,
      ),
    );
  }
  return dl;
}

/**
 * Derive the TuneChat origin for iframe embedding.
 *
 * Preferred source: the backend returns `tunechat_preview_image_url`
 * as an absolute URL (e.g. `http://localhost:3000/pipeline/…/preview.png`)
 * — parsing its origin gives us the right base whether we're in dev
 * (`http://localhost:3000`) or prod (`https://tunechat.raqdrobinson.com`).
 * Fallback if that URL is null/malformed: window.__TUNECHAT_ORIGIN__
 * (a global the app shell can set), then finally a sane dev default.
 */
function tunechatOriginFor(job) {
  const previewUrl = job && job.result && job.result.tunechat_preview_image_url;
  if (previewUrl) {
    try {
      return new URL(previewUrl).origin;
    } catch { /* fall through */ }
  }
  if (typeof window !== "undefined" && window.__TUNECHAT_ORIGIN__) {
    return window.__TUNECHAT_ORIGIN__;
  }
  return "http://localhost:3000"; // dev default — safe because iframe 404s loudly
}

function completeBody(job) {
  const wrap = el("div", { class: "body" });
  const tunechatId = job && job.result && job.result.tunechat_job_id;
  if (tunechatId) {
    // Wrap in .iframe-stub so existing CSS sizes the frame.
    const origin = tunechatOriginFor(job);
    // Pass title + artist to TuneChat's embed so its internal title bar
    // can show both. Prefer the refined metadata from the engrave result
    // (cleaned by cover_search extraction) over the raw user input.
    const meta = (job.result && job.result.metadata) || {};
    const title = meta.title || job.title || "";
    const artist = meta.composer || job.artist || "";
    const params = new URLSearchParams();
    params.set("job", tunechatId);
    if (title) params.set("title", title);
    if (artist) params.set("artist", artist);
    const frame = el("iframe", {
      src: `${origin}/embed?${params.toString()}`,
      title: "TuneChat interactive sheet music",
      // allow= hints for AudioContext autoplay on first gesture inside
      // the iframe (Chrome/Firefox respect; Safari uses user-gesture heuristic)
      allow: "autoplay; clipboard-write",
      // allowfullscreen enables the Fullscreen API path below; without
      // it, iframe.requestFullscreen() rejects on Safari.
      allowfullscreen: "true",
    });
    // Fullscreen toggle. Uses the browser Fullscreen API — Escape key
    // handling, iOS Safari chrome hiding, and screen-reader mode
    // announcements all come for free. The button sits as an overlay
    // in the top-right corner of the iframe wrap (matching YouTube /
    // Vimeo / Figma convention) so the semantic is "expand this
    // frame", not "another action on this result".
    const fullscreenBtn = el("button", {
      class: "fullscreen-btn",
      type: "button",
      "aria-label": "Toggle fullscreen",
      title: "Fullscreen",
    }, icon("fullscreen"));
    fullscreenBtn.addEventListener("click", () => {
      // Exit if already fullscreen, otherwise enter. Swallow the
      // returned promise — failures are surfaced by the browser as
      // permission errors, and we don't want to pop an app-level
      // error toast for what's essentially a UI affordance.
      try {
        if (document.fullscreenElement) {
          document.exitFullscreen();
        } else if (typeof frame.requestFullscreen === "function") {
          frame.requestFullscreen();
        }
      } catch {
        // Older browsers / sandboxed iframes — silent no-op.
      }
    });
    // .iframe-stub.tunechat triggers the fullscreen-ish mobile CSS
    // where the iframe fills the viewport and the card sheds its padding.
    wrap.appendChild(el("div", { class: "iframe-stub tunechat" }, frame, fullscreenBtn));
  } else {
    wrap.appendChild(
      el(
        "div",
        { class: "iframe-stub inline-render" },
        icon("piano"),
        el("span", { text: "Inline sheet music (OSMD)" }),
      ),
    );
  }
  wrap.appendChild(downloadChips(job.job_id));
  return wrap;
}

function errorBody(message, retryable, handlers) {
  const wrap = el("div", { class: "body" });
  const box = el(
    "div",
    { class: "err-box" },
    icon("error"),
    el("div", null, el("p", { text: message })),
  );
  wrap.appendChild(box);

  if (retryable !== false) {
    const actions = el("div", { class: "actions" });
    const retry = el(
      "button",
      { class: "btn-filled", type: "button" },
      icon("refresh"),
      "Try again",
    );
    retry.addEventListener("click", () => handlers.onRetry());
    actions.appendChild(retry);
    wrap.appendChild(actions);
  }
  return wrap;
}

// ── public entry point ───────────────────────────────────────────────

export function renderPhase(container, phase, handlers) {
  let body;
  switch (phase.name) {
    case "idle":
      body = idleBody(phase.source, handlers);
      break;
    case "submitting":
      body = submittingBody();
      break;
    case "working":
      body = workingBody(phase.stage, phase.progress);
      break;
    case "complete":
      body = completeBody(phase.job);
      break;
    case "error":
      body = errorBody(phase.message, phase.retryable, handlers);
      break;
    default:
      body = el("div", null);
  }
  container.replaceChildren(body);
}
