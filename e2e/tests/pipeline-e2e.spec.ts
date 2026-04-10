/**
 * End-to-end pipeline integration test (GAU-59).
 *
 * Wires the entire service stack together and verifies the pipeline
 * produces a sheet music PDF from a MIDI upload, for three different
 * MIDI files. Enforces a 2-minute per-file latency budget and asserts
 * on error propagation when the pipeline is fed an invalid input.
 *
 * Run locally against the docker compose stack:
 *     docker compose up -d
 *     BASE_URL=http://localhost:8000 npm test -- pipeline-e2e.spec.ts
 */
import { test, expect } from "@playwright/test";
import path from "path";
import fs from "fs";

const FIXTURES_ROOT = path.resolve(__dirname, "../..");

// Three representative MIDI files spanning different styles/complexities.
// These exist in-repo so the test is fully reproducible.
const MIDI_FIXTURES = [
  {
    name: "Rat Dance (original test fixture)",
    path: "test_files/Rat Dance.mid",
    title: "Rat Dance",
  },
  {
    name: "ABBA - Knowing Me, Knowing You",
    path: "eval/fixtures/clean_midi/ABBA/Knowing Me, Knowing You.5.mid",
    title: "Knowing Me, Knowing You",
    artist: "ABBA",
  },
  {
    name: "Elvis Presley - Hound Dog",
    path: "eval/fixtures/clean_midi/Elvis Presley/Hound Dog.4.mid",
    title: "Hound Dog",
    artist: "Elvis Presley",
  },
] as const;

// Per-file budget: spec says "reasonable time (under 2 minutes)".
const PIPELINE_TIMEOUT_MS = 120_000;

/**
 * Drive a single MIDI file through the pipeline and return the
 * elapsed wall-clock time plus the resulting job id.
 */
async function runPipeline(
  request: import("@playwright/test").APIRequestContext,
  page: import("@playwright/test").Page,
  midiPath: string,
  title: string,
  artist?: string,
): Promise<{ jobId: string; elapsedMs: number }> {
  const startedAt = Date.now();

  // 1. Upload MIDI
  const fileBuffer = fs.readFileSync(path.resolve(FIXTURES_ROOT, midiPath));
  const uploadRes = await request.post("/v1/uploads/midi", {
    multipart: {
      file: {
        name: path.basename(midiPath),
        mimeType: "audio/midi",
        buffer: fileBuffer,
      },
    },
  });
  expect(uploadRes.ok(), `upload failed for ${midiPath}`).toBeTruthy();
  const uploadBody = await uploadRes.json();
  expect(uploadBody.uri).toBeTruthy();
  expect(uploadBody.ticks_per_beat).toBeGreaterThan(0);

  // 2. Create pipeline job
  const jobRes = await request.post("/v1/jobs", {
    data: {
      midi: {
        uri: uploadBody.uri,
        ticks_per_beat: uploadBody.ticks_per_beat,
        content_hash: uploadBody.content_hash,
      },
      title,
      ...(artist ? { artist } : {}),
    },
  });
  expect(jobRes.ok(), `job create failed for ${midiPath}`).toBeTruthy();
  const jobBody = await jobRes.json();
  const jobId: string = jobBody.job_id;
  expect(jobId).toBeTruthy();

  // 3. Wait for completion via WebSocket, with a hard 2-minute ceiling.
  const httpBase =
    (test.info().project.use as { baseURL?: string }).baseURL ??
    "http://localhost:8000";
  const wsBase = httpBase.replace(/^http/, "ws");
  const wsUrl = `${wsBase}/v1/jobs/${jobId}/ws`;

  await new Promise<void>((resolve, reject) => {
    const hardDeadline = setTimeout(
      () =>
        reject(
          new Error(
            `Pipeline exceeded ${PIPELINE_TIMEOUT_MS / 1000}s for ${midiPath}`,
          ),
        ),
      PIPELINE_TIMEOUT_MS,
    );

    page
      .evaluate(
        (url: string) =>
          new Promise<void>((res, rej) => {
            const ws = new WebSocket(url);
            ws.onmessage = (evt) => {
              const data = JSON.parse(evt.data);
              if (data.type === "job_succeeded") {
                ws.close();
                res();
              }
              if (
                data.type === "job_failed" ||
                data.type === "stage_failed"
              ) {
                ws.close();
                rej(
                  new Error(
                    data.message ?? `Pipeline failed: ${data.type}`,
                  ),
                );
              }
            };
            ws.onerror = () => rej(new Error("WebSocket error"));
          }),
        wsUrl,
      )
      .then(() => {
        clearTimeout(hardDeadline);
        resolve();
      })
      .catch((err) => {
        clearTimeout(hardDeadline);
        reject(err);
      });
  });

  return { jobId, elapsedMs: Date.now() - startedAt };
}

test.describe("Pipeline E2E — MIDI → sheet music PDF", () => {
  // Configure per-test timeout so a hung pipeline doesn't hold the runner forever.
  test.setTimeout(PIPELINE_TIMEOUT_MS + 30_000);

  for (const fixture of MIDI_FIXTURES) {
    test(`${fixture.name} → PDF within 2 minutes`, async ({
      request,
      page,
    }) => {
      const { jobId, elapsedMs } = await runPipeline(
        request,
        page,
        fixture.path,
        fixture.title,
        fixture.artist,
      );

      // Acceptance: pipeline completes under 2 minutes
      expect(
        elapsedMs,
        `pipeline took ${elapsedMs}ms (budget ${PIPELINE_TIMEOUT_MS}ms)`,
      ).toBeLessThan(PIPELINE_TIMEOUT_MS);

      // Acceptance: a sheet music PDF artifact is produced
      const pdfRes = await request.get(`/v1/artifacts/${jobId}/pdf`);
      expect(pdfRes.ok(), "PDF artifact should be downloadable").toBeTruthy();
      expect(pdfRes.headers()["content-type"]).toContain("application/pdf");

      const pdfBytes = await pdfRes.body();
      expect(pdfBytes.length, "PDF should not be empty").toBeGreaterThan(0);
      // PDF files begin with %PDF- signature
      expect(pdfBytes.slice(0, 5).toString()).toBe("%PDF-");

      // Acceptance: the humanized MIDI artifact is also produced and valid
      const midiRes = await request.get(`/v1/artifacts/${jobId}/midi`);
      expect(midiRes.ok(), "MIDI artifact should be downloadable").toBeTruthy();
      const midiBytes = await midiRes.body();
      // MIDI files begin with MThd header
      expect(midiBytes.slice(0, 4).toString()).toBe("MThd");

      // Latency telemetry for the report
      console.log(
        `[pipeline-e2e] ${fixture.name}: ${(elapsedMs / 1000).toFixed(1)}s`,
      );
    });
  }
});

test.describe("Pipeline E2E — error propagation", () => {
  test("invalid MIDI upload surfaces a clear error", async ({ request }) => {
    // Posting a non-MIDI payload to the MIDI upload endpoint should be
    // rejected at the upload stage (400/422/415), not silently accepted.
    const uploadRes = await request.post("/v1/uploads/midi", {
      multipart: {
        file: {
          name: "garbage.mid",
          mimeType: "audio/midi",
          buffer: Buffer.from("this is clearly not a MIDI file"),
        },
      },
    });

    // We expect a 4xx (bad input) — the exact code is less important than
    // "not a 5xx and not silently accepted".
    expect(
      uploadRes.status(),
      `expected 4xx for invalid MIDI, got ${uploadRes.status()}`,
    ).toBeGreaterThanOrEqual(400);
    expect(uploadRes.status()).toBeLessThan(500);

    // Body should include an error message so a client can surface it.
    const body = await uploadRes.text();
    expect(body.length, "error response should have a body").toBeGreaterThan(0);
  });
});
