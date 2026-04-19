// Pure HTTP/WS layer for the oh-sheet backend. No DOM, no state.
// All endpoints are same-origin (`/v1/*`), proxied in dev via vite.config.js.

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024; // 50 MB

/**
 * Parse an error out of a non-2xx fetch Response. Prefers the server's
 * JSON `error` field, falls back to status text so the user never sees
 * a bare "500" with no context.
 */
async function extractError(response) {
  try {
    const body = await response.json();
    if (body && typeof body.error === "string" && body.error.length > 0) {
      return new Error(body.error);
    }
    if (body && typeof body.message === "string" && body.message.length > 0) {
      return new Error(body.message);
    }
  } catch {
    // fall through to status text
  }
  return new Error(response.statusText || `HTTP ${response.status}`);
}

function assertUnderSizeLimit(file) {
  if (file && typeof file.size === "number" && file.size > MAX_UPLOAD_BYTES) {
    throw new Error("File exceeds 50 MB upload limit");
  }
}

async function postMultipart(url, file, responseKey) {
  assertUnderSizeLimit(file);
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(url, { method: "POST", body: form });
  if (!response.ok) throw await extractError(response);
  const body = await response.json();
  return body[responseKey];
}

export async function uploadAudio(file) {
  return postMultipart("/v1/uploads/audio", file, "audio");
}

export async function uploadMidi(file) {
  return postMultipart("/v1/uploads/midi", file, "midi");
}

export async function createJob(payload) {
  const response = await fetch("/v1/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw await extractError(response);
  return response.json();
}

export async function getJob(jobId) {
  const response = await fetch(`/v1/jobs/${jobId}`);
  if (!response.ok) throw await extractError(response);
  return response.json();
}

/**
 * Build a WS URL from the current page origin so the socket hits the same
 * host/port as the API (and the vite proxy in dev).
 */
function buildWsUrl(path) {
  if (typeof window !== "undefined" && window.location) {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}${path}`;
  }
  return path;
}

/**
 * Subscribe to a job's event stream over WebSocket.
 *
 * The caller's `onEvent` receives each parsed event frame as-is. When
 * the socket drops before a terminal event (job_succeeded / job_failed)
 * has been seen, we synthesize a local `job_failed`-shaped event with
 * a retryable error message so the UI can land in a recoverable state
 * instead of spinning forever on the last "working:..." phase.
 *
 * Terminal events seen during normal flow (job_succeeded/failed) flip
 * the `sawTerminal` flag so the same close/error handlers become a
 * no-op — we only synthesize when the socket actually dropped early.
 *
 * @param {string} jobId
 * @param {(event: object) => void} onEvent
 * @returns {() => void} unsubscribe function (closes the socket; safe
 *   to call multiple times; once called, subsequent close events are
 *   treated as intentional and no synthesized error fires)
 */
export function subscribeToJob(jobId, onEvent) {
  const ws = new WebSocket(buildWsUrl(`/v1/jobs/${jobId}/ws`));
  let sawTerminal = false;
  let unsubscribed = false;

  ws.onmessage = (ev) => {
    // Late-frame guard: between unsubscribe() calling ws.close() and the
    // browser actually tearing down the socket, a queued frame can still
    // fire and overwrite a fresh idle phase via the reducer. Chromium
    // cancels pending tasks synchronously in practice, but the WebSocket
    // spec doesn't require it (RFC 6455 §7.1.1) — cheap to harden.
    if (unsubscribed) return;
    try {
      const parsed = JSON.parse(ev.data);
      if (parsed && (parsed.type === "job_succeeded" || parsed.type === "job_failed")) {
        sawTerminal = true;
      }
      onEvent(parsed);
    } catch {
      // Ignore malformed frames — server should never send non-JSON.
    }
  };

  // Fires once per socket; both onclose and onerror can lead here
  // (Chrome fires onerror THEN onclose on network drops; Firefox often
  // just onclose with code 1006). De-duplicated via the `closed` flag.
  let closed = false;
  const handleEarlyClose = (message) => {
    if (closed) return;
    closed = true;
    if (unsubscribed || sawTerminal) return;
    try {
      // stage/progress are intentionally null — this event is synthesized
      // client-side, not emitted by the pipeline. The reducer (state.js)
      // only reads `message`, so null fields are harmless today; the
      // `data.synthesized` flag is the signal for any future consumer
      // that wants to distinguish a real failure from a WS drop.
      onEvent({
        job_id: jobId,
        type: "job_failed",
        stage: null,
        message,
        progress: null,
        data: { synthesized: true, reason: "ws_early_close" },
      });
    } catch {
      // swallow — subscriber failure shouldn't crash the WS cleanup
    }
  };

  ws.onclose = (ev) => {
    // A graceful close (1000) after a terminal event is normal cleanup.
    // Anything else (1006 abnormal, server-initiated close mid-job,
    // proxy timeout) should surface as a retryable error to the user.
    handleEarlyClose(
      ev && ev.code === 1000
        ? "Connection closed"
        : "Connection lost — try again",
    );
  };
  ws.onerror = () => {
    handleEarlyClose("Connection error — try again");
  };

  return () => {
    unsubscribed = true;
    try {
      ws.close();
    } catch {
      // no-op
    }
  };
}

export function artifactUrl(jobId, kind) {
  return `/v1/artifacts/${jobId}/${kind}`;
}
