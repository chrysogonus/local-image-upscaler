import { estimateWorkingBytes, planNativeScales, targetDimensions } from "./geometry";
import type { HardwarePolicyInfo, HardwareReport, ModeCapability, ProcessingMode } from "./types";

export const MODES: ProcessingMode[] = ["upscale", "illustration", "sharpen_only"];

// The finishing sharpen each mode uses. Every default is a preset value, so the
// control always opens showing what it will do.
//
// The modes do not need different strengths to compensate for different
// engines: the sharpener is told how far the last stage stretched the image and
// sizes itself to that, so an illustration engine that lands on the target is
// already treated more gently than a plain 4x resample at the same setting.
export const MODE_SHARPEN: Record<ProcessingMode, number> = {
  upscale: 35,
  illustration: 35,
  sharpen_only: 35,
};

/**
 * Plain-language anchors for the strength, so the common choice is one click and
 * nobody has to guess what a percentage of sharpening means. The slider under
 * Advanced processing stays the fine control over the same value.
 */
export const SHARPEN_PRESETS: { label: string; value: number }[] = [
  { label: "Off", value: 0 },
  { label: "Natural", value: 35 },
  { label: "Crisp", value: 60 },
  { label: "Strong", value: 85 },
];

/** The presets a mode may offer. Sharpen-only has nothing to do at zero. */
export function sharpenPresets(mode: ProcessingMode): { label: string; value: number }[] {
  if (mode === "sharpen_only") return SHARPEN_PRESETS.filter((preset) => preset.value > 0);
  return SHARPEN_PRESETS;
}

export function isUpscalingMode(mode: ProcessingMode): boolean {
  return mode !== "sharpen_only";
}

/**
 * Whether the mode resolved to an engine that actually infers.
 *
 * An engine that declares no native scale is the plain resampler, so there is
 * no tiled inference to plan, budget memory for, or expose controls over.
 */
export function isNeuralMode(capability: ModeCapability | null, mode: ProcessingMode): boolean {
  return Boolean(capability?.native_scales.length) && isUpscalingMode(mode);
}

/*
 * Which advanced controls this mode can still act on.
 *
 * A setting the resolved engine discards is not a tradeoff the user can make,
 * so the panel drops it rather than showing it disabled: greying implies the
 * control would work once something else changed, which is untrue of an engine
 * that cannot honour it at all. The tile selector has always worked this way,
 * hiding itself when the backend reports no safe size.
 */

/** Chaining only exists on engines that may run more than once. */
export function offersPassChoice(capability: ModeCapability | null, mode: ProcessingMode): boolean {
  return isNeuralMode(capability, mode) && (capability?.max_passes ?? 1) > 1;
}

/** Augmentation costs eight inferences, so it is offered only where it lands. */
export function offersTta(capability: ModeCapability | null, mode: ProcessingMode): boolean {
  return isNeuralMode(capability, mode) && Boolean(capability?.supports_tta);
}

/**
 * Restoring before reducing changes nothing unless the source already meets the
 * target: every smaller source runs the engine anyway. Offering it against an
 * image it cannot affect reads as a quality dial, which it is not.
 */
export function offersRestoreLarge(
  capability: ModeCapability | null,
  mode: ProcessingMode,
  sourceAlreadyLarge: boolean,
): boolean {
  return isNeuralMode(capability, mode) && sourceAlreadyLarge;
}

export function plannedDimensions(
  mode: ProcessingMode,
  width: number,
  height: number,
  targetEdge: number,
): { width: number; height: number; scale: number } {
  if (mode === "sharpen_only") return { width, height, scale: 1 };
  return targetDimensions(width, height, targetEdge);
}

export function processingAction(mode: ProcessingMode): string {
  if (mode === "illustration") return "Upscale illustration";
  if (mode === "sharpen_only") return "Sharpen image";
  return "Upscale image";
}

export function processingResultLabel(mode: ProcessingMode, scale = 2): string {
  if (mode === "illustration") return "Illustration upscaled";
  if (mode === "sharpen_only") return "Sharpened";
  if (scale < 1) return "Resized";
  if (scale === 1) return "Same-size result";
  return "Upscaled";
}

export function resolveAvailableMode(
  modes: ModeCapability[],
  current: ProcessingMode,
): ProcessingMode | null {
  if (modes.some((entry) => entry.mode === current && entry.available)) return current;
  return modes.find((entry) => entry.available)?.mode ?? null;
}

export function actionableModes(modes: ModeCapability[]): ModeCapability[] {
  return MODES.flatMap((id) => {
    const capability = modes.find((entry) => entry.mode === id);
    return capability?.available ? [capability] : [];
  });
}

export function resolveSafeTile(choices: number[], current: number): number {
  if (!choices.length) return 0;
  return choices.includes(current) ? current : choices[0];
}

interface SafeTargetInput {
  targets: number[];
  mode: ProcessingMode;
  capability: ModeCapability;
  hardware: HardwareReport | undefined;
  policy: HardwarePolicyInfo;
  width: number;
  height: number;
  tileSize: number;
  maxPasses: number;
  restoreLarge: boolean;
}

/** Stable target visibility. Free memory is deliberately checked only by submission. */
export function safeTargetsForImage(input: SafeTargetInput): number[] {
  const allowed = new Set(input.capability.safe_targets ?? input.targets);
  const candidates = input.targets.filter((target) => allowed.has(target));
  const hardware = input.hardware;
  if (input.policy.mode === "off" || !hardware) return candidates;
  const requirement = input.capability.resource_requirement;
  if (!requirement) return candidates;
  return candidates.filter((target) => {
    if (input.mode === "sharpen_only") return true;
    const dimensions = targetDimensions(input.width, input.height, target);
    const neural = input.capability.native_scales.length > 0;
    const scale = input.restoreLarge ? Math.max(dimensions.scale, 2) : dimensions.scale;
    const passes = neural
      ? planNativeScales(
          scale,
          Math.min(input.maxPasses, input.capability.max_passes),
          input.capability.native_scales,
        )
      : [];
    const workingMib = Math.ceil(
      estimateWorkingBytes(
        input.width,
        input.height,
        dimensions.width,
        dimensions.height,
        neural,
        input.tileSize,
        passes,
      ) /
        (1024 * 1024),
    );
    if (hardware.memory_kind === "unified") {
      const total = hardware.ram_effective_mib ?? hardware.vram_total_mib;
      return (
        total !== null &&
        total >= Math.max(requirement.unified_mib, workingMib + input.policy.ram_reserve_mib)
      );
    }
    const total = hardware.ram_effective_mib;
    return (
      total !== null &&
      total >= Math.max(requirement.ram_mib, workingMib + input.policy.ram_reserve_mib)
    );
  });
}
