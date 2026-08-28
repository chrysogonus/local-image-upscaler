import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const png = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAFElEQVR4nGP4z8Dwn4GBgYGJAQoAHgQCAf1h4ioAAAAASUVORK5CYII=",
  "base64",
);

const settings = {
  target_edge: 3840,
  processing_mode: "upscale",
  sharpen: 35,
  tile_size: 0,
  tta: false,
  restore_large: false,
  max_neural_passes: 3,
  workflow: null,
};

const queued = {
  id: "browser-job",
  state: "queued",
  phase: "queued",
  message: "Waiting for the local processor",
  progress: 0,
  settings,
  source: null,
  result: null,
  error: null,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  revision: 0,
};

const completed = {
  ...queued,
  state: "completed",
  phase: "completed",
  message: "Full-resolution result is ready",
  progress: 1,
  revision: 1,
  source: {
    filename: "source.png",
    width: 2,
    height: 1,
    mode: "RGBA",
    format: "PNG",
    animated: false,
    frames: 1,
    has_alpha: true,
    has_icc: false,
    bit_depth: 8,
    warnings: [],
  },
  result: {
    width: 3840,
    height: 1920,
    bytes: png.length,
    engine: "browser-smoke:x4",
    processing_mode: "upscale",
    filename: "source-upscaled.png",
    neural_passes: [4, 4, 4],
    resolved_tile_size: 128,
    generative: false,
    warnings: [],
  },
};

const capabilities = {
  version: "browser-smoke",
  modes: [
    {
      mode: "upscale",
      name: "Upscale",
      description: "Faithful enlargement for photographs.",
      available: true,
      generative: false,
      engine: "Browser smoke engine",
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
      source: "browser-smoke",
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

test("selects, submits, reports progress, compares, downloads, and remains accessible", async ({
  page,
}) => {
  let deleteRequests = 0;
  await page.addInitScript((terminalSnapshot) => {
    class BrowserEventSource extends EventTarget {
      static CLOSED = 2;
      readyState = 1;
      onerror: (() => void) | null = null;

      constructor() {
        super();
        window.setTimeout(() => {
          this.dispatchEvent(new MessageEvent("job", { data: JSON.stringify(terminalSnapshot) }));
        }, 50);
      }

      close() {
        this.readyState = BrowserEventSource.CLOSED;
      }
    }
    Object.defineProperty(window, "EventSource", { value: BrowserEventSource });
  }, completed);

  await page.route("**/api/v1/capabilities", (route) => route.fulfill({ json: capabilities }));
  await page.route("**/api/v1/jobs", async (route) => {
    if (route.request().method() === "POST") await route.fulfill({ status: 202, json: queued });
    else await route.continue();
  });
  await page.route("**/api/v1/jobs/browser-job/result", (route) =>
    route.fulfill({
      body: png,
      contentType: "image/png",
    }),
  );
  await page.route("**/api/v1/jobs/browser-job", (route) => {
    deleteRequests += 1;
    return route.fulfill({ status: 202, json: completed });
  });

  await page.goto("/");
  await expect(page.getByText("Faithful enlargement for photographs.")).toBeVisible();
  await page.locator('input[type="file"]').first().setInputFiles({
    name: "source.png",
    mimeType: "image/png",
    buffer: png,
  });
  await expect(page.getByAltText("Original source")).toBeVisible();
  await page.getByRole("button", { name: "Upscale image" }).click();

  await expect(page.getByAltText("Upscaled result")).toBeVisible();
  await expect(page.getByRole("status").filter({ hasText: "100%" })).toBeVisible();
  await page.getByRole("button", { name: "1:1 output pixels" }).click();
  await expect(page.getByRole("button", { name: "Fit view" })).toBeVisible();

  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);

  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: /Download PNG/ }).click();
  expect((await download).suggestedFilename()).toBe("source-upscaled.png");
  await expect.poll(() => deleteRequests).toBe(1);
  await expect(page.getByRole("button", { name: "Downloaded · job cleared" })).toBeVisible();
});
