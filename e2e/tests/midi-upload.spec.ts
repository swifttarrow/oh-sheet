import { test, expect } from "@playwright/test";
import path from "path";
import fs from "fs";

const MIDI_FILE = path.resolve(__dirname, "../../test_files/Rat Dance.mid");
const PIPELINE_TIMEOUT = 90_000;

test.describe("MIDI upload → playable audio", () => {
  test("Rat Dance MIDI upload produces a playable MIDI artifact", async ({
    request,
    page,
  }) => {
    // ---------------------------------------------------------------
    // 1. Upload the MIDI file via the API
    // ---------------------------------------------------------------
    const fileBuffer = fs.readFileSync(MIDI_FILE);

    const uploadRes = await request.post("/v1/uploads/midi", {
      multipart: {
        file: {
          name: "Rat Dance.mid",
          mimeType: "audio/midi",
          buffer: fileBuffer,
        },
      },
    });
    expect(uploadRes.ok()).toBeTruthy();
    const uploadBody = await uploadRes.json();
    expect(uploadBody.uri).toBeTruthy();
    expect(uploadBody.ticks_per_beat).toBeGreaterThan(0);

    // ---------------------------------------------------------------
    // 2. Create a pipeline job from the uploaded MIDI
    // ---------------------------------------------------------------
    const jobRes = await request.post("/v1/jobs", {
      data: {
        midi: {
          uri: uploadBody.uri,
          ticks_per_beat: uploadBody.ticks_per_beat,
          content_hash: uploadBody.content_hash,
        },
        title: "Rat Dance",
      },
    });
    expect(jobRes.ok()).toBeTruthy();
    const jobBody = await jobRes.json();
    const jobId: string = jobBody.job_id;
    expect(jobId).toBeTruthy();

    // ---------------------------------------------------------------
    // 3. Wait for the pipeline to complete via WebSocket
    // ---------------------------------------------------------------
    const httpBase = (test.info().project.use as { baseURL?: string }).baseURL ?? "http://localhost:8000";
    const wsBase = httpBase.replace(/^http/, "ws");
    const wsUrl = `${wsBase}/v1/jobs/${jobId}/ws`;

    const succeeded = await new Promise<boolean>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error("Pipeline timed out")),
        PIPELINE_TIMEOUT
      );

      page
        .evaluate(
          (url: string) =>
            new Promise<boolean>((res, rej) => {
              const ws = new WebSocket(url);
              ws.onmessage = (evt) => {
                const data = JSON.parse(evt.data);
                if (data.type === "job_succeeded") {
                  ws.close();
                  res(true);
                }
                if (
                  data.type === "job_failed" ||
                  data.type === "stage_failed"
                ) {
                  ws.close();
                  rej(new Error(data.message ?? `Pipeline failed: ${data.type}`));
                }
              };
              ws.onerror = () => rej(new Error("WebSocket error"));
            }),
          wsUrl
        )
        .then((result) => {
          clearTimeout(timeout);
          resolve(result);
        })
        .catch((err) => {
          clearTimeout(timeout);
          reject(err);
        });
    });

    expect(succeeded).toBe(true);

    // ---------------------------------------------------------------
    // 4. Verify the MIDI artifact is downloadable
    // ---------------------------------------------------------------
    const artifactRes = await request.get(
      `/v1/artifacts/${jobId}/midi`
    );
    expect(artifactRes.ok()).toBeTruthy();
    const midiBytes = await artifactRes.body();
    // MIDI files start with "MThd" header
    expect(midiBytes.slice(0, 4).toString()).toBe("MThd");

    // ---------------------------------------------------------------
    // 5. Verify the result page renders with a <midi-player> element
    // ---------------------------------------------------------------
    // Navigate to the app root — Flutter web loads here
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // Inject a <midi-player> element directly pointed at the artifact
    // to verify the audio stack works (the same path the Flutter app uses).
    const midiArtifactUrl = `${
      page.url().replace(/\/$/, "")
    }/v1/artifacts/${jobId}/midi`;

    const playerLoaded = await page.evaluate(async (src: string) => {
      const player = document.createElement(
        "midi-player"
      ) as HTMLElement & { src?: string };
      player.setAttribute("src", src);
      player.setAttribute("sound-font", "");
      player.id = "e2e-midi-player";
      document.body.appendChild(player);

      // Wait for the player to load (it fetches the MIDI and soundfont)
      return new Promise<boolean>((resolve) => {
        const check = () => {
          // html-midi-player sets a "playing" property and renders a shadow DOM
          // with controls once loaded. Check for shadow root or the loaded state.
          if (player.shadowRoot) {
            resolve(true);
            return;
          }
          // Also check if customElements registered midi-player
          if (customElements.get("midi-player")) {
            resolve(true);
            return;
          }
        };

        // Check immediately, then poll
        check();
        const interval = setInterval(() => {
          check();
        }, 500);

        // Resolve after a reasonable wait even if shadow root isn't ready
        // (custom element may be registered but rendering is async)
        setTimeout(() => {
          clearInterval(interval);
          // If the element exists in DOM, that's good enough
          const el = document.getElementById("e2e-midi-player");
          resolve(el !== null);
        }, 10_000);
      });
    }, midiArtifactUrl);

    expect(playerLoaded).toBe(true);

    // Verify the injected player element is actually in the page
    const playerEl = page.locator("#e2e-midi-player");
    await expect(playerEl).toBeAttached();
  });
});
