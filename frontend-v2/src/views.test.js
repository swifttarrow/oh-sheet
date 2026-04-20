import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderPhase } from "./views.js";

/**
 * DOM-level tests for renderPhase(container, phase, handlers).
 *
 * Tests assert rendered structure (segmented picker, inputs, spinner svg,
 * stepper pills, progress bar, iframe, download chips, error box) and
 * handler wiring (onSubmit / onRetry / onSourceChange).
 *
 * No innerHTML — assertions use querySelector / textContent / attributes.
 */

let container;
let handlers;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  handlers = {
    onSubmit: vi.fn(),
    onRetry: vi.fn(),
    onSourceChange: vi.fn(),
  };
});

function findButtonByText(root, text) {
  return [...root.querySelectorAll("button")].find(
    (b) => b.textContent.trim() === text || b.textContent.includes(text),
  );
}

describe("renderPhase — idle / youtube", () => {
  beforeEach(() => {
    renderPhase(container, { name: "idle", source: "youtube" }, handlers);
  });

  // Demo scope (Apr 20): audio + midi sources are hidden, so SOURCES
  // has length 1 and the segmented picker auto-hides. Old assertions
  // about 3 buttons / "marks YouTube tab active" don't apply until
  // audio/midi come back. Keep this test to lock in the auto-hide
  // behavior — if SOURCES ever grows, the picker should reappear.
  it("does NOT render a segmented picker when only one source is defined", () => {
    expect(container.querySelector(".segmented")).toBeNull();
  });

  it("renders a URL input (no file picker)", () => {
    const inputs = container.querySelectorAll("input[type='text']");
    expect(inputs.length).toBeGreaterThanOrEqual(1);
    expect(container.querySelector("input[type='file']")).toBeNull();
    const urlInput = [...inputs].find(
      (i) => (i.placeholder || "").toLowerCase().includes("youtube") ||
             (i.placeholder || "").toLowerCase().includes("url"),
    );
    expect(urlInput).toBeTruthy();
  });

  it("renders a 'Let's go!' submit button", () => {
    expect(findButtonByText(container, "Let's go!")).toBeTruthy();
  });

  it("fires onSubmit({source:'youtube', url}) when submit clicked", () => {
    const urlInput = [...container.querySelectorAll("input[type='text']")].find(
      (i) => (i.placeholder || "").toLowerCase().match(/youtube|url/),
    );
    urlInput.value = "https://youtu.be/abc";
    urlInput.dispatchEvent(new Event("input", { bubbles: true }));
    findButtonByText(container, "Let's go!").click();
    expect(handlers.onSubmit).toHaveBeenCalledTimes(1);
    const arg = handlers.onSubmit.mock.calls[0][0];
    expect(arg.source).toBe("youtube");
    expect(arg.url).toBe("https://youtu.be/abc");
  });

  // Demo scope: no Audio tab exists anymore, so the tab-click test is
  // skipped until audio/midi are restored to SOURCES. The onSourceChange
  // handler wiring itself is unchanged — still covered by the
  // segmented() unit when SOURCES.length > 1.

  it("does NOT fire onSubmit when URL is empty", () => {
    findButtonByText(container, "Let's go!").click();
    expect(handlers.onSubmit).not.toHaveBeenCalled();
  });
});

// Demo scope (Apr 20): audio + midi sources are hidden from the UI
// for demo. The views.js branches still render correctly if the
// source IS set to "audio" / "midi" (the segmented picker is just
// gated separately), so these tests are skipped rather than deleted
// — restoring the feature post-demo means flipping .skip → .describe
// in one edit plus restoring SOURCES in views.js.
describe.skip("renderPhase — idle / audio", () => {
  beforeEach(() => {
    renderPhase(container, { name: "idle", source: "audio" }, handlers);
  });

  it("marks Audio tab active", () => {
    const active = container.querySelector(".segmented .seg.on");
    expect(active.textContent).toContain("Audio");
  });

  it("renders a file picker input (accept audio types)", () => {
    const fileInput = container.querySelector("input[type='file']");
    expect(fileInput).toBeTruthy();
    const accept = fileInput.getAttribute("accept") || "";
    expect(accept).toMatch(/\.mp3|audio/);
  });

  it("renders optional title + artist inputs", () => {
    const placeholders = [...container.querySelectorAll("input[type='text']")]
      .map((i) => (i.placeholder || "").toLowerCase());
    expect(placeholders.some((p) => p.includes("title"))).toBe(true);
    expect(placeholders.some((p) => p.includes("artist"))).toBe(true);
  });

  it("does NOT fire onSubmit when no file selected", () => {
    findButtonByText(container, "Let's go!").click();
    expect(handlers.onSubmit).not.toHaveBeenCalled();
  });

  it("fires onSubmit with file when a file is present", () => {
    const fileInput = container.querySelector("input[type='file']");
    const file = new File(["hi"], "song.mp3", { type: "audio/mpeg" });
    Object.defineProperty(fileInput, "files", { value: [file], configurable: true });
    fileInput.dispatchEvent(new Event("change", { bubbles: true }));
    findButtonByText(container, "Let's go!").click();
    expect(handlers.onSubmit).toHaveBeenCalledTimes(1);
    const arg = handlers.onSubmit.mock.calls[0][0];
    expect(arg.source).toBe("audio");
    expect(arg.file).toBe(file);
  });
});

describe.skip("renderPhase — idle / midi", () => {
  beforeEach(() => {
    renderPhase(container, { name: "idle", source: "midi" }, handlers);
  });

  it("file input accepts .mid,.midi", () => {
    const fileInput = container.querySelector("input[type='file']");
    expect(fileInput).toBeTruthy();
    const accept = fileInput.getAttribute("accept") || "";
    expect(accept).toMatch(/\.mid/);
  });
});

describe("renderPhase — idle / title (legacy source, treated as youtube fallback)", () => {
  it("still renders a form body even for unknown sources", () => {
    renderPhase(container, { name: "idle", source: "title" }, handlers);
    // Sentinel switched from .segmented (auto-hidden post-demo-cut)
    // to the body wrapper that idleBody() always emits.
    expect(container.querySelector(".body")).toBeTruthy();
  });
});

describe("renderPhase — submitting", () => {
  beforeEach(() => {
    renderPhase(container, { name: "submitting" }, handlers);
  });

  it("renders an SVG spinner", () => {
    const svg = container.querySelector("svg.m3-spinner");
    expect(svg).toBeTruthy();
    expect(svg.namespaceURI).toBe("http://www.w3.org/2000/svg");
    expect(svg.querySelector("circle")).toBeTruthy();
  });

  it("renders a status message", () => {
    expect(container.textContent.toLowerCase()).toMatch(/upload|queue|submit/);
  });

  it("does NOT render form controls", () => {
    expect(container.querySelector(".segmented")).toBeNull();
    expect(container.querySelector("input")).toBeNull();
    expect(findButtonByText(container, "Let's go!")).toBeFalsy();
  });
});

describe("renderPhase — working (transcribe, 0.35)", () => {
  beforeEach(() => {
    renderPhase(
      container,
      { name: "working", stage: "transcribe", progress: 0.35 },
      handlers,
    );
  });

  it("renders 4 stage pills", () => {
    expect(container.querySelectorAll(".stepper .step").length).toBe(4);
  });

  it("marks the first pill done and the second pill active", () => {
    const pills = container.querySelectorAll(".stepper .step");
    expect(pills[0].classList.contains("done")).toBe(true);
    expect(pills[1].classList.contains("active")).toBe(true);
    expect(pills[2].classList.contains("done")).toBe(false);
    expect(pills[2].classList.contains("active")).toBe(false);
  });

  it("renders a progress bar with width 35%", () => {
    const bar = container.querySelector(".bar span");
    expect(bar).toBeTruthy();
    expect(bar.style.width).toBe("35%");
  });

  it("renders the percent in the meta row", () => {
    expect(container.querySelector(".meta").textContent).toContain("35%");
  });
});

describe("renderPhase — complete (with tunechat_job_id)", () => {
  const phase = {
    name: "complete",
    job: {
      job_id: "job-123",
      result: {
        schema_version: "1",
        metadata: { title: "Sonata", composer: "Mozart" },
        pdf_uri: "/v1/artifacts/job-123/pdf",
        musicxml_uri: "/v1/artifacts/job-123/musicxml",
        humanized_midi_uri: "/v1/artifacts/job-123/midi",
        tunechat_job_id: "tc-xyz",
        tunechat_preview_image_url: null,
      },
    },
  };

  beforeEach(() => {
    renderPhase(container, phase, handlers);
  });

  it("renders an iframe with src pattern /embed?job=<id>", () => {
    const iframe = container.querySelector("iframe");
    expect(iframe).toBeTruthy();
    expect(iframe.getAttribute("src")).toContain("/embed?job=tc-xyz");
  });

  it("renders 3 download chips (PDF / MusicXML / MIDI)", () => {
    const chips = container.querySelectorAll(".downloads .assist-chip");
    expect(chips.length).toBe(3);
    const text = [...chips].map((c) => c.textContent).join(" ");
    expect(text).toContain("PDF");
    expect(text).toContain("MusicXML");
    expect(text).toContain("MIDI");
  });

  it("renders a fullscreen button inside the iframe wrap", () => {
    // Overlay lives in the iframe container, not in the downloads row,
    // so the semantic is "expand this frame" (like YouTube) not
    // "another download action". Convention > discoverability-in-context.
    const wrap = container.querySelector(".iframe-stub.tunechat");
    expect(wrap).toBeTruthy();
    const btn = wrap.querySelector(".fullscreen-btn");
    expect(btn).toBeTruthy();
    expect(btn.getAttribute("aria-label")).toMatch(/fullscreen/i);
  });

  // Fullscreen tests capture+restore document.fullscreenElement so the
  // mock doesn't leak into later suites. Also mocks exitFullscreen
  // since happy-dom doesn't ship it by default.
  let fsDescriptor;
  beforeEach(() => {
    fsDescriptor = Object.getOwnPropertyDescriptor(document, "fullscreenElement");
  });
  afterEach(() => {
    if (fsDescriptor) {
      Object.defineProperty(document, "fullscreenElement", fsDescriptor);
    } else {
      delete document.fullscreenElement;
    }
    delete document.exitFullscreen;
  });

  it("fullscreen button calls requestFullscreen on the WRAPPER, not the iframe", () => {
    // Target must be the .iframe-stub wrapper, not the iframe itself —
    // requesting on the iframe would hide the button (the button is
    // the iframe's sibling, not its child) making click-to-exit
    // unreachable. See PR #89 review.
    const wrap = container.querySelector(".iframe-stub.tunechat");
    const iframe = container.querySelector("iframe");
    const wrapSpy = vi.fn().mockResolvedValue(undefined);
    const iframeSpy = vi.fn().mockResolvedValue(undefined);
    wrap.requestFullscreen = wrapSpy;
    iframe.requestFullscreen = iframeSpy;
    Object.defineProperty(document, "fullscreenElement", {
      configurable: true,
      get: () => null,
    });
    container.querySelector(".fullscreen-btn").click();
    expect(wrapSpy).toHaveBeenCalledTimes(1);
    expect(iframeSpy).not.toHaveBeenCalled();
  });

  it("fullscreen button calls exitFullscreen when already fullscreen", () => {
    const wrap = container.querySelector(".iframe-stub.tunechat");
    const exitSpy = vi.fn().mockResolvedValue(undefined);
    document.exitFullscreen = exitSpy;
    Object.defineProperty(document, "fullscreenElement", {
      configurable: true,
      get: () => wrap,
    });
    container.querySelector(".fullscreen-btn").click();
    expect(exitSpy).toHaveBeenCalledTimes(1);
  });
});

describe("renderPhase — complete (without tunechat_job_id)", () => {
  const phase = {
    name: "complete",
    job: {
      job_id: "job-456",
      result: {
        schema_version: "1",
        metadata: { title: "Prelude", composer: "Bach" },
        pdf_uri: "/v1/artifacts/job-456/pdf",
        musicxml_uri: "/v1/artifacts/job-456/musicxml",
        humanized_midi_uri: "/v1/artifacts/job-456/midi",
        tunechat_job_id: null,
        tunechat_preview_image_url: null,
      },
    },
  };

  beforeEach(() => {
    renderPhase(container, phase, handlers);
  });

  it("does NOT render an iframe", () => {
    expect(container.querySelector("iframe")).toBeNull();
  });

  it("renders an inline-render placeholder", () => {
    expect(container.querySelector(".iframe-stub, .inline-render")).toBeTruthy();
  });

  it("still renders 3 download chips", () => {
    const chips = container.querySelectorAll(".downloads .assist-chip");
    expect(chips.length).toBe(3);
  });
});

describe("renderPhase — error", () => {
  beforeEach(() => {
    renderPhase(
      container,
      { name: "error", message: "YouTube blocked the fetch", retryable: true },
      handlers,
    );
  });

  it("renders an err-box with the message text", () => {
    const box = container.querySelector(".err-box");
    expect(box).toBeTruthy();
    expect(box.textContent).toContain("YouTube blocked the fetch");
  });

  it("renders a 'Try again' button that fires onRetry()", () => {
    const btn = findButtonByText(container, "Try again");
    expect(btn).toBeTruthy();
    btn.click();
    expect(handlers.onRetry).toHaveBeenCalledTimes(1);
  });
});

describe("renderPhase — atomic swap", () => {
  it("clears previous content when called again", () => {
    renderPhase(container, { name: "idle", source: "youtube" }, handlers);
    // Post-demo-cut: the segmented picker is hidden (SOURCES has 1
    // entry). Use the YouTube URL input placeholder as the idle-phase
    // sentinel instead — it's stable across the source-picker cut.
    expect(
      [...container.querySelectorAll("input")].some(
        (i) => (i.placeholder || "").toLowerCase().includes("youtube"),
      ),
    ).toBe(true);
    renderPhase(container, { name: "submitting" }, handlers);
    expect(
      [...container.querySelectorAll("input")].some(
        (i) => (i.placeholder || "").toLowerCase().includes("youtube"),
      ),
    ).toBe(false);
    expect(container.querySelector("svg.m3-spinner")).toBeTruthy();
  });
});
