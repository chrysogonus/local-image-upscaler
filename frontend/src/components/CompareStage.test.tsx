import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { CompareStage } from "./CompareStage";

function renderStage(resultUrl: string | null = "blob:result") {
  const onModeChange = vi.fn();
  const onSplitChange = vi.fn();
  const onPixelViewChange = vi.fn();
  render(
    <CompareStage
      originalUrl="blob:original"
      resultUrl={resultUrl}
      width={3840}
      height={2160}
      mode="split"
      onModeChange={onModeChange}
      split={50}
      onSplitChange={onSplitChange}
      pixelView={false}
      onPixelViewChange={onPixelViewChange}
      resultLabel="Upscaled"
    />,
  );
  return { onModeChange, onSplitChange, onPixelViewChange };
}

describe("image comparison", () => {
  it("keeps result controls disabled until a result exists", () => {
    renderStage(null);

    expect((screen.getByRole("button", { name: "Upscaled" }) as HTMLButtonElement).disabled).toBe(
      true,
    );
    expect((screen.getByRole("button", { name: "Split" }) as HTMLButtonElement).disabled).toBe(
      true,
    );
    expect(screen.queryByRole("slider")).toBeNull();
  });

  it("exposes keyboard-operable view, split, and pixel controls", async () => {
    const user = userEvent.setup();
    const handlers = renderStage();

    await user.click(screen.getByRole("button", { name: "Upscaled" }));
    await user.click(screen.getByRole("button", { name: "1:1 output pixels" }));
    fireEvent.change(screen.getByRole("slider"), { target: { value: "64" } });

    expect(handlers.onModeChange).toHaveBeenCalledWith("result");
    expect(handlers.onPixelViewChange).toHaveBeenCalledWith(true);
    expect(handlers.onSplitChange).toHaveBeenLastCalledWith(64);
  });
});
