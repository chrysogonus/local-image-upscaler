import { afterEach, describe, expect, it, vi } from "vitest";
import { cancelOrDeleteJob, createJob, getCapabilities, observeJob } from "./api";
import type { JobSettings, JobSnapshot } from "./types";

const settings: JobSettings = {
  target_edge: 3840,
  processing_mode: "upscale",
  sharpen: 15,
  tile_size: 0,
  tta: false,
  restore_large: false,
  max_neural_passes: 3,
  workflow: null,
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("API requests", () => {
  it("serializes job settings without turning null into a string", async () => {
    const snapshot = { id: "job-1", state: "queued" } as JobSnapshot;
    const fetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const body = init?.body as FormData;
      expect(body.get("file")).toBeInstanceOf(File);
      expect(body.get("target_edge")).toBe("3840");
      expect(body.get("tta")).toBe("false");
      expect(body.has("workflow")).toBe(false);
      return jsonResponse(snapshot, 202);
    });
    vi.stubGlobal("fetch", fetch);

    await expect(createJob(new File(["pixels"], "source.png"), settings)).resolves.toEqual(
      snapshot,
    );
    expect(fetch).toHaveBeenCalledWith("/api/v1/jobs", expect.objectContaining({ method: "POST" }));
  });

  it("surfaces structured validation details", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse(
          {
            detail: [{ msg: "target is unsafe" }, { msg: "tile is too large" }],
          },
          422,
        ),
      ),
    );

    await expect(getCapabilities()).rejects.toThrow("target is unsafe; tile is too large");
  });

  it("treats an already removed job as a successful cleanup", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ detail: "gone" }, 404)),
    );

    await expect(cancelOrDeleteJob("gone")).resolves.toBeUndefined();
  });
});

describe("job event stream", () => {
  it("delivers snapshots and closes itself at a terminal state", () => {
    class FakeEventSource {
      static CLOSED = 2;
      static instance: FakeEventSource;
      readyState = 1;
      onerror: (() => void) | null = null;
      listeners = new Map<string, (event: Event) => void>();

      constructor(public url: string) {
        FakeEventSource.instance = this;
      }

      addEventListener(name: string, listener: EventListener) {
        this.listeners.set(name, listener);
      }

      close() {
        this.readyState = FakeEventSource.CLOSED;
      }
    }
    vi.stubGlobal("EventSource", FakeEventSource);
    const onJob = vi.fn();
    const onError = vi.fn();

    observeJob("job-1", onJob, onError);
    const source = FakeEventSource.instance;
    const snapshot = { id: "job-1", state: "completed" } as JobSnapshot;
    source.listeners.get("job")?.(new MessageEvent("job", { data: JSON.stringify(snapshot) }));

    expect(source.url).toBe("/api/v1/jobs/job-1/events");
    expect(onJob).toHaveBeenCalledWith(snapshot);
    expect(source.readyState).toBe(FakeEventSource.CLOSED);
    expect(onError).not.toHaveBeenCalled();
  });
});
