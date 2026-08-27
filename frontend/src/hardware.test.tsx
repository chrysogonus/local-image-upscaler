import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { HardwarePanel } from "./App";
import type { Capabilities } from "./types";

describe("hardware summary", () => {
  it("shows totals, current availability, policy, and omitted reasons", () => {
    const capabilities: Capabilities = {
      version: "test",
      modes: [],
      targets: [3840, 7680],
      max_upload_bytes: 1,
      max_input_pixels: 1,
      platform: {},
      hardware_policy: {
        mode: "safe",
        version: 1,
        ram_reserve_mib: 4096,
        vram_reserve_mib: 1536,
        visibility_basis: "stable total capacity",
        admission_basis: "currently available memory",
      },
      hardware: [
        {
          scope: "backend",
          ram_physical_mib: 65536,
          ram_effective_mib: 65536,
          ram_available_mib: 49152,
          gpu_name: "RTX Test",
          vram_total_mib: 16384,
          vram_available_mib: 12288,
          memory_kind: "dedicated",
          source: "test",
          warnings: [],
        },
      ],
      excluded_features: [
        {
          id: "illustration",
          name: "Illustration",
          reason: "Requires 8 GiB RAM.",
        },
      ],
    };

    const html = renderToStaticMarkup(<HardwarePanel capabilities={capabilities} />);
    expect(html).toContain("Hardware policy");
    expect(html).toContain("RTX Test");
    expect(html).toContain("64 GiB total");
    expect(html).toContain("48 GiB free");
    expect(html).toContain("Illustration");
    expect(html).toContain("Requires 8 GiB RAM");
  });
});
