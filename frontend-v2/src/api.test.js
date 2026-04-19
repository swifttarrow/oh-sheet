import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  uploadAudio,
  uploadMidi,
  createJob,
  getJob,
  subscribeToJob,
  artifactUrl,
} from "./api.js";

// Helper to build a fetch response.
function jsonResponse(body, { ok = true, status = 200, statusText = "OK" } = {}) {
  return {
    ok,
    status,
    statusText,
    json: async () => body,
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
  };
}

function textResponse(text, { ok = false, status = 500, statusText = "Server Error" } = {}) {
  return {
    ok,
    status,
    statusText,
    json: async () => {
      throw new Error("not json");
    },
    text: async () => text,
  };
}

describe("artifactUrl", () => {
  it("builds the artifact URL from jobId and kind", () => {
    expect(artifactUrl("abc123", "pdf")).toBe("/v1/artifacts/abc123/pdf");
    expect(artifactUrl("j-1", "midi")).toBe("/v1/artifacts/j-1/midi");
    expect(artifactUrl("j-2", "musicxml")).toBe("/v1/artifacts/j-2/musicxml");
  });
});

describe("uploadAudio", () => {
  let fetchMock;
  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs multipart to /v1/uploads/audio and returns the audio field", async () => {
    const remote = { id: "a1", original_filename: "song.mp3" };
    fetchMock.mockResolvedValueOnce(jsonResponse({ audio: remote }));

    const file = new File(["hello"], "song.mp3", { type: "audio/mpeg" });
    const result = await uploadAudio(file);

    expect(result).toEqual(remote);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe("/v1/uploads/audio");
    expect(opts.method).toBe("POST");
    expect(opts.body).toBeInstanceOf(FormData);
    expect(opts.body.get("file")).toBe(file);
  });

  it("rejects files over 50 MB before hitting the network", async () => {
    const big = new File([new Uint8Array(1)], "big.mp3", { type: "audio/mpeg" });
    Object.defineProperty(big, "size", { value: 50 * 1024 * 1024 + 1 });

    await expect(uploadAudio(big)).rejects.toThrow(/50 ?MB/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("throws the server error message when response is non-2xx JSON", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ error: "bad audio format" }, { ok: false, status: 400, statusText: "Bad Request" }),
    );
    const file = new File(["x"], "x.mp3");
    await expect(uploadAudio(file)).rejects.toThrow("bad audio format");
  });

  it("throws status text when non-2xx response is not JSON", async () => {
    fetchMock.mockResolvedValueOnce(textResponse("<html>oops</html>", { ok: false, status: 502, statusText: "Bad Gateway" }));
    const file = new File(["x"], "x.mp3");
    await expect(uploadAudio(file)).rejects.toThrow("Bad Gateway");
  });
});

describe("uploadMidi", () => {
  let fetchMock;
  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs multipart to /v1/uploads/midi and returns the midi field", async () => {
    const remote = { id: "m1", original_filename: "song.mid" };
    fetchMock.mockResolvedValueOnce(jsonResponse({ midi: remote }));

    const file = new File(["MThd"], "song.mid", { type: "audio/midi" });
    const result = await uploadMidi(file);

    expect(result).toEqual(remote);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe("/v1/uploads/midi");
    expect(opts.method).toBe("POST");
    expect(opts.body).toBeInstanceOf(FormData);
    expect(opts.body.get("file")).toBe(file);
  });

  it("rejects files over 50 MB before hitting the network", async () => {
    const big = new File([new Uint8Array(1)], "big.mid");
    Object.defineProperty(big, "size", { value: 50 * 1024 * 1024 + 1 });

    await expect(uploadMidi(big)).rejects.toThrow(/50 ?MB/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("createJob", () => {
  let fetchMock;
  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs JSON to /v1/jobs and returns the job", async () => {
    const job = { job_id: "j1", status: "queued", variant: "audio_upload" };
    fetchMock.mockResolvedValueOnce(jsonResponse(job));

    const payload = { audio: { id: "a1" }, title: "Hey" };
    const result = await createJob(payload);

    expect(result).toEqual(job);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe("/v1/jobs");
    expect(opts.method).toBe("POST");
    expect(opts.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(opts.body)).toEqual(payload);
  });

  it("throws server error message on non-2xx", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ error: "missing source" }, { ok: false, status: 422, statusText: "Unprocessable" }),
    );
    await expect(createJob({})).rejects.toThrow("missing source");
  });
});

describe("getJob", () => {
  let fetchMock;
  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("GETs /v1/jobs/:id and returns job", async () => {
    const job = { job_id: "j1", status: "running" };
    fetchMock.mockResolvedValueOnce(jsonResponse(job));

    const result = await getJob("j1");
    expect(result).toEqual(job);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe("/v1/jobs/j1");
    expect(opts == null || opts.method === undefined || opts.method === "GET").toBe(true);
  });

  it("throws on non-2xx", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ error: "not found" }, { ok: false, status: 404, statusText: "Not Found" }),
    );
    await expect(getJob("missing")).rejects.toThrow("not found");
  });
});

describe("subscribeToJob", () => {
  let instances;
  class MockWebSocket {
    constructor(url) {
      this.url = url;
      this.onopen = null;
      this.onmessage = null;
      this.onclose = null;
      this.onerror = null;
      this.closed = false;
      instances.push(this);
    }
    close() {
      this.closed = true;
      if (this.onclose) this.onclose({});
    }
    // Test helper
    _fireMessage(data) {
      if (this.onmessage) this.onmessage({ data: JSON.stringify(data) });
    }
  }

  beforeEach(() => {
    instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("opens a WS at /v1/jobs/:id/ws and pipes parsed messages to onEvent", () => {
    const onEvent = vi.fn();
    subscribeToJob("j1", onEvent);

    expect(instances).toHaveLength(1);
    const ws = instances[0];
    expect(ws.url).toMatch(/\/v1\/jobs\/j1\/ws$/);

    const event = { job_id: "j1", type: "stage_started", stage: "ingest", progress: 0 };
    ws._fireMessage(event);
    expect(onEvent).toHaveBeenCalledWith(event);
  });

  it("returns an unsubscribe function that closes the socket", () => {
    const onEvent = vi.fn();
    const unsub = subscribeToJob("j1", onEvent);
    const ws = instances[0];
    expect(ws.closed).toBe(false);
    unsub();
    expect(ws.closed).toBe(true);
  });

  it("synthesizes a retryable job_failed when the socket drops before terminal", () => {
    // If the WS closes with an abnormal code (1006 network drop, or
    // anything non-1000) before we've seen job_succeeded/job_failed,
    // the UI would otherwise spin forever on the last "working:..."
    // phase. The subscriber must surface a job_failed so the state
    // machine can transition to an error/retryable phase.
    const onEvent = vi.fn();
    subscribeToJob("j1", onEvent);
    const ws = instances[0];
    ws.onclose({ code: 1006 });
    expect(onEvent).toHaveBeenCalledTimes(1);
    const evt = onEvent.mock.calls[0][0];
    expect(evt.type).toBe("job_failed");
    expect(evt.job_id).toBe("j1");
    expect(evt.message).toMatch(/connection lost|try again/i);
    expect(evt.data.synthesized).toBe(true);
  });

  it("does NOT synthesize job_failed after a real job_succeeded event", () => {
    // Terminal event received → subsequent close is normal cleanup,
    // not a failure. Synthesizing here would double-emit and confuse
    // the reducer (the real job_succeeded already transitioned the UI
    // to Complete).
    const onEvent = vi.fn();
    subscribeToJob("j1", onEvent);
    const ws = instances[0];
    ws._fireMessage({ job_id: "j1", type: "job_succeeded", data: {} });
    onEvent.mockClear();
    ws.onclose({ code: 1000 });
    expect(onEvent).not.toHaveBeenCalled();
  });

  it("does NOT synthesize job_failed after unsubscribe (user-initiated close)", () => {
    // The unsubscribe fn calls ws.close(), which in the mock fires
    // onclose. That must not synthesize an error — the user asked to
    // stop listening, they don't want a spurious "Connection lost".
    const onEvent = vi.fn();
    const unsub = subscribeToJob("j1", onEvent);
    unsub();
    expect(onEvent).not.toHaveBeenCalled();
  });

  it("drops late frames that arrive after unsubscribe", () => {
    // Between unsubscribe() calling ws.close() and the browser tearing
    // down the socket, a queued frame can still fire onmessage. That
    // frame must not reach the subscriber — otherwise a stale late
    // job event could clobber the fresh phase the caller just set
    // (e.g. on source-tab switch, where main.js unsubscribes and then
    // resets to idle).
    const onEvent = vi.fn();
    const unsub = subscribeToJob("j1", onEvent);
    const ws = instances[0];
    unsub();
    onEvent.mockClear(); // ignore any close-path side effects
    ws._fireMessage({ job_id: "j1", type: "stage_progress", stage: "ingest", progress: 0.5 });
    expect(onEvent).not.toHaveBeenCalled();
  });

  it("synthesizes job_failed on WS error event too (not just close)", () => {
    // Firefox often fires only onclose with code 1006; Chrome fires
    // onerror THEN onclose. Both paths must land in the same error
    // state, and the dedupe (closed flag) must prevent double-emit.
    const onEvent = vi.fn();
    subscribeToJob("j1", onEvent);
    const ws = instances[0];
    ws.onerror(new Event("error"));
    expect(onEvent).toHaveBeenCalledTimes(1);
    // Follow-up onclose after onerror must NOT emit a second event.
    ws.onclose({ code: 1006 });
    expect(onEvent).toHaveBeenCalledTimes(1);
  });
});
