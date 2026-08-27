import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { App } from "./App";
import type { Capabilities, JobSnapshot } from "./types";

const api = vi.hoisted(() => ({
  cancelOrDeleteJob: vi.fn(),
  createJob: vi.fn(),
  fetchResult: vi.fn(),
  getCapabilities: vi.fn(),
  observeJob: vi.fn(),
}));

vi.mock("./api", () => api);

const settings = {
  target_edge: 3840,
  processing_mode: "upscale" as const,
  sharpen: 15,
  tile_size: 0,
  tta: false,
  restore_large: false,
  max_neural_passes: 3,
  workflow: null,
};

const capabilities: Capabilities = {
  version: "test",
  modes: [
    {
      mode: "upscale",
      name: "Upscale",
      description: "Faithful enlargement",
      available: true,
      generative: false,
      engine: "Test engine",
      device: "CPU",
      unavailable_reason: null,
      fallback_reason: null,
      max_passes: 3,
      native_scales: [2, 3, 4],
      resource_requirement: { ram_mib: 512, vram_mib: 0, unified_mib: 512 },
      safe_tile_sizes: [0, 128],
      safe_targets: [3840, 7680],
    },
  ],
  targets: [3840, 7680],
  max_upload_bytes: 100_000_000,
  max_input_pixels: 120_000_000,
  platform: {},
  hardware: [
    {
      scope: "backend",
      ram_physical_mib: 16_384,
      ram_effective_mib: 16_384,
      ram_available_mib: 12_288,
      gpu_name: null,
      vram_total_mib: null,
      vram_available_mib: null,
      memory_kind: "dedicated",
      source: "test",
      warnings: [],
    },
  ],
  hardware_policy: {
    mode: "safe",
    version: 1,
    ram_reserve_mib: 4096,
    vram_reserve_mib: 1536,
    visibility_basis: "stable total capacity",
    admission_basis: "currently available memory",
  },
  excluded_features: [],
};

function snapshot(state: JobSnapshot["state"]): JobSnapshot {
  return {
    id: "job-1",
    state,
    phase: state,
    message: state === "completed" ? "Full-resolution result is ready" : "Queued",
    progress: state === "completed" ? 1 : 0,
    settings,
    source:
      state === "completed"
        ? {
            filename: "source.png",
            width: 64,
            height: 32,
            mode: "RGBA",
            format: "PNG",
            animated: false,
            frames: 1,
            has_alpha: true,
            has_icc: false,
            bit_depth: 8,
            warnings: [],
          }
        : null,
    result:
      state === "completed"
        ? {
            width: 3840,
            height: 1920,
            bytes: 1024,
            engine: "test:x4",
            processing_mode: "upscale",
            filename: "source-upscaled.png",
            neural_passes: [4, 4, 4],
            resolved_tile_size: 128,
            generative: false,
            warnings: [],
          }
        : null,
    error: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    revision: state === "completed" ? 1 : 0,
  };
}

beforeEach(() => {
  class PreviewImage {
    naturalWidth = 64;
    naturalHeight = 32;
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;

    set src(_value: string) {
      queueMicrotask(() => this.onload?.());
    }
  }
  vi.stubGlobal("Image", PreviewImage);
  vi.stubGlobal("URL", {
    createObjectURL: vi.fn(() => "blob:test"),
    revokeObjectURL: vi.fn(),
  });
  api.getCapabilities.mockResolvedValue(capabilities);
  api.createJob.mockResolvedValue(snapshot("queued"));
  api.fetchResult.mockResolvedValue(new Blob(["result"], { type: "image/png" }));
  api.observeJob.mockImplementation((_id, onJob) => {
    queueMicrotask(() => onJob(snapshot("completed")));
    return vi.fn();
  });
});

afterEach(() => {
  vi.clearAllMocks();
});

it("runs the select, submit, progress, compare, and download-ready flow", async () => {
  const user = userEvent.setup();
  const { container } = render(<App />);
  await screen.findByText("Faithful enlargement");
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;

  await user.upload(input, new File(["pixels"], "source.png", { type: "image/png" }));
  await screen.findByText("source.png");
  await user.click(screen.getByRole("button", { name: "Upscale image" }));

  await waitFor(() => expect(api.createJob).toHaveBeenCalledTimes(1));
  expect(api.observeJob).toHaveBeenCalledWith("job-1", expect.any(Function), expect.any(Function));
  await screen.findByAltText("Upscaled result");
  expect(screen.getByText("100%")).not.toBeNull();
  expect((screen.getByRole("button", { name: /Download PNG/ }) as HTMLButtonElement).disabled).toBe(
    false,
  );
});
