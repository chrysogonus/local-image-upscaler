import react from "@vitejs/plugin-react";
import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    target: "es2022",
    sourcemap: true,
  },
  test: {
    environment: "jsdom",
    exclude: [...configDefaults.exclude, "e2e/**"],
    setupFiles: ["./src/test/setup.ts"],
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/**/*.test.{ts,tsx}", "src/test/**", "src/main.tsx", "src/types.ts"],
      // Calibrated against Vitest 4's AST-aware remapping, which is the default
      // for the v8 provider and counts what the old mapping over-credited. The
      // same suite reported 85.30 statements and 85.30 lines under Vitest 3 --
      // identical figures, because that provider reported line counts for both
      // -- and reports 73.88 and 77.71 here, while functions rose from 59.15 to
      // 69.84. Coverage did not fall; it is measured properly. Raising these to
      // the old numbers means writing tests, not restoring a threshold.
      thresholds: {
        branches: 68,
        functions: 65,
        lines: 75,
        statements: 70,
      },
    },
  },
});
