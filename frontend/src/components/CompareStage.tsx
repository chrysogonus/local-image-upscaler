import type { CSSProperties, PointerEvent as ReactPointerEvent } from "react";

type ViewMode = "original" | "result" | "split";

interface CompareStageProps {
  originalUrl: string;
  resultUrl: string | null;
  width: number;
  height: number;
  mode: ViewMode;
  onModeChange: (mode: ViewMode) => void;
  split: number;
  onSplitChange: (value: number) => void;
  pixelView: boolean;
  onPixelViewChange: (value: boolean) => void;
  resultLabel: string;
  filename: string;
  onReplaceFile: (file: File) => void;
}

export function CompareStage({
  originalUrl,
  resultUrl,
  width,
  height,
  mode,
  onModeChange,
  split,
  onSplitChange,
  pixelView,
  onPixelViewChange,
  resultLabel,
  filename,
  onReplaceFile,
}: CompareStageProps) {
  const effectiveMode = resultUrl ? mode : "original";
  const stackStyle = {
    "--media-ratio": `${width} / ${height}`,
    ...(pixelView ? { width: `${width}px`, height: `${height}px` } : {}),
  } as CSSProperties;
  const resultStyle = {
    clipPath: effectiveMode === "split" ? `inset(0 ${100 - split}% 0 0)` : undefined,
  };
  const splitting = Boolean(resultUrl) && effectiveMode === "split";

  // Dragging anywhere on the image moves the divider; the range input below
  // stays the keyboard- and screen-reader-operable control.
  const splitFromPointer = (event: ReactPointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (bounds.width <= 0) return;
    const ratio = (event.clientX - bounds.left) / bounds.width;
    onSplitChange(Math.round(Math.min(1, Math.max(0, ratio)) * 100));
  };
  const dragHandlers = splitting
    ? {
        onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => {
          event.preventDefault();
          event.currentTarget.setPointerCapture?.(event.pointerId);
          splitFromPointer(event);
        },
        onPointerMove: (event: ReactPointerEvent<HTMLDivElement>) => {
          if (!event.currentTarget.hasPointerCapture?.(event.pointerId)) return;
          splitFromPointer(event);
        },
      }
    : {};

  return (
    <section className="preview-panel" aria-label="Image comparison">
      <div className="preview-toolbar">
        <div className="segmented" role="group" aria-label="Preview image">
          <button
            aria-pressed={effectiveMode === "original"}
            onClick={() => onModeChange("original")}
          >
            Original
          </button>
          <button
            disabled={!resultUrl}
            aria-pressed={effectiveMode === "result"}
            onClick={() => onModeChange("result")}
          >
            {resultLabel}
          </button>
          <button
            disabled={!resultUrl}
            aria-pressed={effectiveMode === "split"}
            onClick={() => onModeChange("split")}
          >
            Split
          </button>
        </div>
        <div className="toolbar-source">
          <span className="source-name" title={filename}>
            {filename}
          </span>
          <label className="replace-button">
            Replace
            <input
              type="file"
              accept="image/*"
              onChange={(event) => {
                const chosen = event.target.files?.[0];
                if (chosen) onReplaceFile(chosen);
              }}
            />
          </label>
          <button
            className="pixel-toggle"
            aria-pressed={pixelView}
            onClick={() => onPixelViewChange(!pixelView)}
          >
            {pixelView ? "Fit view" : "1:1 output pixels"}
          </button>
        </div>
      </div>
      <div
        className={`stage-viewport ${pixelView ? "pixel-view" : "fit-view"}`}
        tabIndex={pixelView ? 0 : undefined}
        aria-label={pixelView ? "Scrollable full-size image viewport" : undefined}
      >
        <div
          className={`media-stack${splitting ? " draggable-split" : ""}`}
          style={stackStyle}
          {...dragHandlers}
        >
          <img
            className="comparison-image original-image"
            src={originalUrl}
            alt="Original source"
          />
          {resultUrl && effectiveMode !== "original" ? (
            <img
              className="comparison-image result-image"
              style={resultStyle}
              src={resultUrl}
              alt={`${resultLabel} result`}
            />
          ) : null}
          {resultUrl && effectiveMode === "split" ? (
            <div className="split-rule" style={{ left: `${split}%` }} aria-hidden="true" />
          ) : null}
        </div>
      </div>
      {resultUrl && effectiveMode === "split" ? (
        <label className="split-control">
          <span>Original</span>
          <input
            type="range"
            min="0"
            max="100"
            value={split}
            onChange={(event) => onSplitChange(Number(event.target.value))}
            aria-label="Before and after split position"
          />
          <span>{resultLabel}</span>
        </label>
      ) : null}
    </section>
  );
}
