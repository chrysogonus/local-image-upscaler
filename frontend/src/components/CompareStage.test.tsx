import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { CompareStage } from "./CompareStage";

function renderStage(resultUrl: string | null = "blob:result") {
  const onModeChange = vi.fn();
  const onSplitChange = vi.fn();
  const onPixelViewChange = vi.fn();
  const { container } = render(
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
      filename="source.png"
      onReplaceFile={() => undefined}
    />,
  );
  return { container, onModeChange, onSplitChange, onPixelViewChange };
}

// jsdom has no PointerEvent, so pointer input arrives as a MouseEvent
// carrying the pointerId React and the capture calls read.
class TestPointerEvent extends MouseEvent {
  readonly pointerId: number;

  constructor(type: string, init: MouseEventInit & { pointerId: number }) {
    super(type, { bubbles: true, ...init });
    this.pointerId = init.pointerId;
  }
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

  it("moves the split by pressing and dragging inside the image", () => {
    const handlers = renderStage();
    const stack = handlers.container.querySelector(".media-stack") as HTMLDivElement;
    stack.getBoundingClientRect = () => ({ left: 100, width: 400 }) as DOMRect;
    const captured = new Set<number>();
    stack.setPointerCapture = (pointerId: number) => {
      captured.add(pointerId);
    };
    stack.hasPointerCapture = (pointerId: number) => captured.has(pointerId);

    fireEvent(stack, new TestPointerEvent("pointermove", { pointerId: 1, clientX: 400 }));
    expect(handlers.onSplitChange).not.toHaveBeenCalled();

    fireEvent(stack, new TestPointerEvent("pointerdown", { pointerId: 1, clientX: 200 }));
    expect(handlers.onSplitChange).toHaveBeenLastCalledWith(25);

    fireEvent(stack, new TestPointerEvent("pointermove", { pointerId: 1, clientX: 400 }));
    expect(handlers.onSplitChange).toHaveBeenLastCalledWith(75);

    fireEvent(stack, new TestPointerEvent("pointermove", { pointerId: 1, clientX: 9000 }));
    expect(handlers.onSplitChange).toHaveBeenLastCalledWith(100);
  });
});
