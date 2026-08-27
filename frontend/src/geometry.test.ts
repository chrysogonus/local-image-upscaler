import { describe, expect, it } from "vitest";
import { estimateWorkingBytes, planNativeScales, targetDimensions } from "./geometry";

describe("targetDimensions", () => {
  it("preserves a landscape aspect ratio", () => {
    expect(targetDimensions(1920, 1080, 3840)).toEqual({ width: 3840, height: 2160, scale: 2 });
  });

  it("uses the portrait long edge", () => {
    expect(targetDimensions(1080, 1920, 7680)).toEqual({ width: 4320, height: 7680, scale: 4 });
  });
});

describe("planNativeScales", () => {
  // These expectations mirror backend/tests/test_geometry.py. The panel would
  // report passes and memory the backend never planned if the two drift.
  it.each([
    [8, [4, 2]],
    [16, [4, 4]],
    [19.2, [4, 4, 2]],
    [4, [4]],
    [2.5, [3]],
  ])("covers %sx with native passes", (requested, expected) => {
    expect(planNativeScales(requested)).toEqual(expected);
  });

  it("stops once the remainder is negligible", () => {
    expect(planNativeScales(4.1)).toEqual([4]);
  });

  it("respects the pass budget", () => {
    expect(planNativeScales(60, 1)).toEqual([4]);
    expect(planNativeScales(60, 2)).toEqual([4, 4]);
  });

  it("plans nothing when no enlargement is needed", () => {
    expect(planNativeScales(1)).toEqual([]);
    expect(planNativeScales(0.4)).toEqual([]);
  });

  it("never asks a 4x-only engine for another scale", () => {
    // The NCNN runtime returns a wrongly cropped image for -s 2 and -s 3.
    expect(planNativeScales(2, 3, [4])).toEqual([4]);
    expect(planNativeScales(8, 3, [4])).toEqual([4, 4]);
    expect(planNativeScales(19.2, 3, [4])).toEqual([4, 4, 4]);
  });
});

describe("estimateWorkingBytes", () => {
  it("includes a neural tile working set", () => {
    expect(estimateWorkingBytes(100, 100, 400, 400, true, 128)).toBeGreaterThan(
      estimateWorkingBytes(100, 100, 400, 400, false, 0),
    );
  });

  it("accounts for the chained intermediate buffer", () => {
    expect(estimateWorkingBytes(200, 200, 3840, 3840, true, 128, [4, 4, 2])).toBeGreaterThan(
      estimateWorkingBytes(200, 200, 3840, 3840, true, 128, [4]),
    );
  });
});
