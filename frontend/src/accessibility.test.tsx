import { render } from "@testing-library/react";
import axe from "axe-core";
import { expect, it } from "vitest";
import { CompareStage } from "./components/CompareStage";
import { ProgressStatus } from "./components/ProgressStatus";

it("has no detectable accessibility violations in the comparison and status controls", async () => {
  const { container } = render(
    <main>
      <CompareStage
        originalUrl="blob:original"
        resultUrl="blob:result"
        width={3840}
        height={2160}
        mode="split"
        onModeChange={() => undefined}
        split={50}
        onSplitChange={() => undefined}
        pixelView={false}
        onPixelViewChange={() => undefined}
        resultLabel="Upscaled"
      />
      <ProgressStatus job={null} hasSource />
    </main>,
  );

  // JSDOM has no canvas implementation, so colour contrast is covered by the
  // real-browser smoke test instead of emitting a false infrastructure error here.
  const result = await axe.run(container, { rules: { "color-contrast": { enabled: false } } });
  expect(result.violations).toEqual([]);
});
