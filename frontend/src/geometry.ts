export function targetDimensions(
  width: number,
  height: number,
  targetEdge: number,
): { width: number; height: number; scale: number } {
  if (width < 1 || height < 1 || targetEdge < 1) {
    throw new Error("Dimensions must be positive");
  }
  const scale = targetEdge / Math.max(width, height);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
    scale,
  };
}

export function formatBytes(value: number): string {
  if (value < 1024 * 1024) return `${Math.max(1, Math.round(value / 1024))} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

// Mirrors upscaler.geometry. One model pass caps at 4x, so a larger factor is
// covered by chaining passes rather than handing the remainder to Lanczos.
const NATIVE_SCALES = [2, 3, 4];
const RESIDUAL_TOLERANCE = 1.15;

function smallestNativeScaleAtLeast(factor: number, nativeScales: number[]): number {
  return nativeScales.find((scale) => factor <= scale) ?? nativeScales[nativeScales.length - 1];
}

export function planNativeScales(
  requestedScale: number,
  maxPasses = 3,
  nativeScales: number[] = NATIVE_SCALES,
): number[] {
  if (maxPasses < 1) throw new Error("At least one neural pass must be allowed");
  if (!nativeScales.length) throw new Error("An engine must support at least one native scale");
  if (requestedScale <= 1) return [];
  const ordered = [...nativeScales].sort((a, b) => a - b);
  const plan: number[] = [];
  let remaining = requestedScale;
  while (remaining > RESIDUAL_TOLERANCE && plan.length < maxPasses) {
    const step = smallestNativeScaleAtLeast(remaining, ordered);
    plan.push(step);
    remaining /= step;
  }
  if (plan.length === 0) plan.push(smallestNativeScaleAtLeast(requestedScale, ordered));
  return plan;
}

export function estimateWorkingBytes(
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
  neural: boolean,
  tileSize: number,
  passes: number[] = [],
): number {
  const rgba = (width: number, height: number) => width * height * 4;
  const sourceAndOutput = rgba(sourceWidth, sourceHeight) + rgba(targetWidth, targetHeight);
  if (!neural) return Math.ceil(sourceAndOutput * 2.5);
  const nativeScale = passes.length ? Math.max(...passes) : 4;
  const tile = tileSize || Math.min(512, Math.max(sourceWidth, sourceHeight));
  const tileWorking = tile * tile * (4 + 32 + 32 * nativeScale * nativeScale);
  // A chained plan overshoots the target on purpose, so the largest buffer the
  // run ever holds is the last pass's output rather than the encoded result.
  const total = passes.reduce((product, scale) => product * scale, 1);
  const chained = passes.length ? rgba(sourceWidth * total, sourceHeight * total) : 0;
  return Math.ceil(sourceAndOutput * 1.5 + tileWorking + chained);
}
