import { describe, expect, it } from "vitest";
import {
  MODES,
  MODE_SHARPEN,
  SHARPEN_PRESETS,
  actionableModes,
  isNeuralMode,
  isUpscalingMode,
  offersPassChoice,
  offersRestoreLarge,
  offersTta,
  plannedDimensions,
  processingAction,
  processingResultLabel,
  resolveAvailableMode,
  resolveSafeTile,
  safeTargetsForImage,
  sharpenPresets,
} from "./processing";
import type { HardwareReport, ModeCapability } from "./types";

describe("processing modes", () => {
  it("preserves dimensions for sharpen-only processing", () => {
    expect(plannedDimensions("sharpen_only", 1234, 777, 3840)).toEqual({
      width: 1234,
      height: 777,
      scale: 1,
    });
    expect(isUpscalingMode("sharpen_only")).toBe(false);
  });

  it("uses the selected long edge for both resizing modes", () => {
    expect(plannedDimensions("upscale", 1920, 1080, 3840)).toEqual({
      width: 3840,
      height: 2160,
      scale: 2,
    });
    expect(plannedDimensions("illustration", 1920, 1080, 3840)).toEqual({
      width: 3840,
      height: 2160,
      scale: 2,
    });
    expect(plannedDimensions("upscale", 1080, 1920, 7680)).toEqual({
      width: 4320,
      height: 7680,
      scale: 4,
    });
  });

  it("provides operation-specific action and result labels", () => {
    expect(processingAction("upscale")).toBe("Upscale image");
    expect(processingAction("illustration")).toBe("Upscale illustration");
    expect(processingAction("sharpen_only")).toBe("Sharpen image");
    expect(processingResultLabel("sharpen_only")).toBe("Sharpened");
    expect(processingResultLabel("illustration")).toBe("Illustration upscaled");
    expect(processingResultLabel("upscale", 0.5)).toBe("Resized");
    expect(processingResultLabel("upscale", 4)).toBe("Upscaled");
  });

  it("opens every mode on a usable sharpen strength", () => {
    expect(MODE_SHARPEN.upscale).toBeGreaterThan(0);
    expect(MODE_SHARPEN.illustration).toBeGreaterThan(0);
    expect(MODE_SHARPEN.sharpen_only).toBeGreaterThan(0);
  });

  it("opens every mode on a preset the control can show as chosen", () => {
    const values = SHARPEN_PRESETS.map((preset) => preset.value);
    for (const mode of MODES) {
      expect(values).toContain(MODE_SHARPEN[mode]);
    }
  });

  it("offers no off switch in the mode whose only job is sharpening", () => {
    // The backend rejects a zero strength there outright.
    expect(sharpenPresets("sharpen_only").map((preset) => preset.value)).not.toContain(0);
    expect(sharpenPresets("upscale")).toEqual(SHARPEN_PRESETS);
    expect(MODE_SHARPEN.sharpen_only).toBeGreaterThan(0);
  });
});

describe("safe hardware choices", () => {
  const capability = (mode: ModeCapability["mode"], available: boolean): ModeCapability => ({
    mode,
    name: mode,
    description: `${mode} description`,
    available,
    generative: false,
    engine: "test",
    device: "CPU",
    unavailable_reason: available ? null : "too large",
    fallback_reason: null,
    max_passes: 1,
    native_scales: [],
    resource_requirement: { ram_mib: 2048, vram_mib: 0, unified_mib: 2048 },
    safe_tile_sizes: [],
    safe_targets: [3840, 7680],
  });

  it("hides unavailable modes and recovers to the first actionable one", () => {
    const modes = [
      capability("upscale", true),
      capability("illustration", false),
      capability("sharpen_only", true),
    ];
    expect(actionableModes(modes).map((entry) => entry.mode)).toEqual(["upscale", "sharpen_only"]);
    expect(resolveAvailableMode(modes, "illustration")).toBe("upscale");
  });

  it("resets an unsafe tile to the first safe choice", () => {
    expect(resolveSafeTile([0, 128, 256], 768)).toBe(0);
    expect(resolveSafeTile([0, 128, 256], 128)).toBe(128);
    expect(resolveSafeTile([], 512)).toBe(0);
  });

  it("offers an advanced control only where the engine can act on it", () => {
    const chaining: ModeCapability = {
      ...capability("upscale", true),
      max_passes: 3,
      native_scales: [2, 3, 4],
      supports_tta: true,
    };
    // A single-pass engine that discards augmentation - the ComfyUI modes.
    const singlePass: ModeCapability = {
      ...capability("illustration", true),
      max_passes: 1,
      native_scales: [1],
      supports_tta: false,
    };
    expect(offersPassChoice(chaining, "upscale")).toBe(true);
    expect(offersPassChoice(singlePass, "illustration")).toBe(false);
    expect(offersTta(chaining, "upscale")).toBe(true);
    expect(offersTta(singlePass, "illustration")).toBe(false);
    // The resampler infers nothing, and sharpen-only changes no dimensions.
    expect(offersPassChoice(capability("upscale", true), "upscale")).toBe(false);
    expect(offersTta(chaining, "sharpen_only")).toBe(false);
    expect(isNeuralMode(null, "upscale")).toBe(false);
  });

  it("offers restore-before-reducing only against a source that is already large", () => {
    const engine: ModeCapability = {
      ...capability("illustration", true),
      native_scales: [1],
    };
    expect(offersRestoreLarge(engine, "illustration", true)).toBe(true);
    // Every smaller source runs the engine anyway, so the flag changes nothing.
    expect(offersRestoreLarge(engine, "illustration", false)).toBe(false);
    expect(offersRestoreLarge(capability("upscale", true), "upscale", true)).toBe(false);
  });

  it("filters targets against stable RAM but policy off preserves both", () => {
    const hardware: HardwareReport = {
      scope: "backend",
      ram_physical_mib: 5400,
      ram_effective_mib: 5400,
      ram_available_mib: 5000,
      gpu_name: null,
      vram_total_mib: null,
      vram_available_mib: null,
      memory_kind: "dedicated",
      source: "test",
      warnings: [],
    };
    const base = {
      targets: [3840, 7680],
      mode: "upscale" as const,
      capability: capability("upscale", true),
      hardware,
      width: 10000,
      height: 10000,
      tileSize: 0,
      maxPasses: 1,
      restoreLarge: false,
    };
    expect(
      safeTargetsForImage({
        ...base,
        policy: {
          mode: "safe",
          version: 1,
          ram_reserve_mib: 4096,
          vram_reserve_mib: 1536,
          visibility_basis: "stable",
          admission_basis: "live",
        },
      }),
    ).toEqual([3840]);
    expect(
      safeTargetsForImage({
        ...base,
        policy: {
          mode: "off",
          version: 1,
          ram_reserve_mib: 4096,
          vram_reserve_mib: 1536,
          visibility_basis: "stable",
          admission_basis: "live",
        },
      }),
    ).toEqual([3840, 7680]);
  });
});
