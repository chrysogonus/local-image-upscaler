import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cancelOrDeleteJob, createJob, fetchResult, getCapabilities, observeJob } from "./api";
import { CompareStage } from "./components/CompareStage";
import { ProgressStatus } from "./components/ProgressStatus";
import { estimateWorkingBytes, formatBytes, planNativeScales } from "./geometry";
import {
  MODE_SHARPEN,
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
import type {
  Capabilities,
  JobSettings,
  JobSnapshot,
  LocalImageInfo,
  ProcessingMode,
} from "./types";

const ACTIVE_STATES = new Set([
  "queued",
  "analyzing",
  "loading_model",
  "enhancing",
  "finishing",
  "encoding",
  "cancelling",
]);

// Nothing in this application sets the flag that shows this notice, and the
// tests keep it that way. It stays because the label has to already exist on
// the day an engine that does invent detail is added: see ACCEPTABLE_USE.md.
const GENERATIVE_NOTICE =
  "This mode synthesises detail that was never in the source. Treat the result as an " +
  "interpretation of the image, not a recovery of it, and never as evidence about a person.";

function inspectImage(file: File, url: string): Promise<LocalImageInfo> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight, url });
    image.onerror = () => reject(new Error("The browser could not preview that image."));
    image.src = url;
  });
}

function AppHeader() {
  return (
    <header className="app-header">
      <div className="eyebrow">Image / local refinement</div>
      <div className="header-copy">
        <h1>More detail, kept honest.</h1>
        <p>
          Private 4K and 8K upscaling on your computer. Choose a mode, choose a size, and the app
          picks the right engine, model, and settings for it.
        </p>
        <div className="local-badge">
          <span />
          127.0.0.1 only
        </div>
      </div>
    </header>
  );
}

function formatMib(value: number | null): string {
  if (value === null) return "Unknown";
  return value >= 1024 ? `${(value / 1024).toFixed(value % 1024 ? 1 : 0)} GiB` : `${value} MiB`;
}

export function HardwarePanel({ capabilities }: { capabilities: Capabilities }) {
  return (
    <section className="hardware-panel" aria-label="Detected hardware">
      <div className="hardware-heading">
        <strong>Hardware policy</strong>
        <span>
          {capabilities.hardware_policy.mode === "safe"
            ? `Safe v${capabilities.hardware_policy.version}`
            : "Off"}
        </span>
      </div>
      {capabilities.hardware.map((report) => (
        <div className="hardware-report" key={report.scope}>
          <b>{report.scope === "comfyui" ? "ComfyUI" : "This app"}</b>
          <span>{report.gpu_name ?? "CPU only / GPU unknown"}</span>
          <span>
            RAM {formatMib(report.ram_effective_mib)} total · {formatMib(report.ram_available_mib)}{" "}
            free
          </span>
          {report.gpu_name || report.vram_total_mib !== null ? (
            <span>
              {report.memory_kind === "unified" ? "Unified" : "VRAM"}{" "}
              {formatMib(report.vram_total_mib)} total · {formatMib(report.vram_available_mib)} free
            </span>
          ) : null}
          {report.warnings.map((warning) => (
            <span className="hardware-warning" key={warning}>
              {warning}
            </span>
          ))}
        </div>
      ))}
      <p>
        Choices use stable total capacity. Free memory is checked again when a job is submitted.
      </p>
      {capabilities.excluded_features.length ? (
        <details>
          <summary>
            {capabilities.excluded_features.length} omitted feature
            {capabilities.excluded_features.length === 1 ? "" : "s"}
          </summary>
          <ul>
            {capabilities.excluded_features.map((item) => (
              <li key={item.id}>
                <b>{item.name}:</b> {item.reason}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}

export function App() {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [capabilityError, setCapabilityError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [localImage, setLocalImage] = useState<LocalImageInfo | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [uiError, setUiError] = useState<string | null>(null);
  const [downloaded, setDownloaded] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [targetEdge, setTargetEdge] = useState(3840);
  const [mode, setMode] = useState<ProcessingMode>("upscale");
  const [sharpen, setSharpen] = useState(MODE_SHARPEN.upscale);
  const [tileSize, setTileSize] = useState(0);
  const [tta, setTta] = useState(false);
  const [restoreLarge, setRestoreLarge] = useState(false);
  const [maxNeuralPasses, setMaxNeuralPasses] = useState(3);
  const [viewMode, setViewMode] = useState<"original" | "result" | "split">("original");
  const [split, setSplit] = useState(50);
  const [pixelView, setPixelView] = useState(false);
  const stopObserving = useRef<(() => void) | null>(null);
  const loadedResultFor = useRef<string | null>(null);
  const fileUrl = useRef<string | null>(null);
  const resultObjectUrl = useRef<string | null>(null);

  const refreshCapabilities = useCallback(async () => {
    try {
      setCapabilities(await getCapabilities());
      setCapabilityError(null);
    } catch (error) {
      setCapabilityError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    void refreshCapabilities();
    const onFocus = () => void refreshCapabilities();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [refreshCapabilities]);

  const clearResult = useCallback(() => {
    stopObserving.current?.();
    stopObserving.current = null;
    loadedResultFor.current = null;
    if (resultObjectUrl.current) URL.revokeObjectURL(resultObjectUrl.current);
    resultObjectUrl.current = null;
    setResultUrl(null);
    setViewMode("original");
    setPixelView(false);
    setDownloaded(false);
  }, []);

  const chooseFile = useCallback(
    async (nextFile: File) => {
      setUiError(null);
      clearResult();
      if (job) void cancelOrDeleteJob(job.id).catch(() => undefined);
      if (fileUrl.current) URL.revokeObjectURL(fileUrl.current);
      const url = URL.createObjectURL(nextFile);
      fileUrl.current = url;
      try {
        const info = await inspectImage(nextFile, url);
        setFile(nextFile);
        setLocalImage(info);
        setJob(null);
      } catch (error) {
        URL.revokeObjectURL(url);
        fileUrl.current = null;
        setFile(null);
        setLocalImage(null);
        setUiError(error instanceof Error ? error.message : String(error));
      }
    },
    [clearResult, job],
  );

  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const pasted = Array.from(event.clipboardData?.files ?? []).find((item) =>
        item.type.startsWith("image/"),
      );
      if (pasted) void chooseFile(pasted);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [chooseFile]);

  useEffect(
    () => () => {
      stopObserving.current?.();
      if (fileUrl.current) URL.revokeObjectURL(fileUrl.current);
      if (resultObjectUrl.current) URL.revokeObjectURL(resultObjectUrl.current);
    },
    [],
  );

  const modes = useMemo(() => actionableModes(capabilities?.modes ?? []), [capabilities]);
  const selected = useMemo(
    () => capabilities?.modes.find((entry) => entry.mode === mode) ?? null,
    [capabilities, mode],
  );
  const neural = isNeuralMode(selected, mode);
  const dimensions = useMemo(
    () =>
      localImage ? plannedDimensions(mode, localImage.width, localImage.height, targetEdge) : null,
    [localImage, mode, targetEdge],
  );
  const passPlan = useMemo(() => {
    if (!dimensions || !neural || !selected) return [];
    const budget = Math.min(maxNeuralPasses, selected.max_passes);
    const scale = restoreLarge ? Math.max(dimensions.scale, 2) : dimensions.scale;
    return planNativeScales(scale, budget, selected.native_scales);
  }, [dimensions, maxNeuralPasses, neural, restoreLarge, selected]);
  const memoryEstimate = useMemo(() => {
    if (!localImage || !dimensions) return null;
    return estimateWorkingBytes(
      localImage.width,
      localImage.height,
      dimensions.width,
      dimensions.height,
      neural,
      tileSize,
      passPlan,
    );
  }, [dimensions, localImage, neural, passPlan, tileSize]);
  const busy = Boolean(job && ACTIVE_STATES.has(job.state));
  const sourceAlreadyLarge = Boolean(isUpscalingMode(mode) && dimensions && dimensions.scale <= 1);
  // Advanced settings this mode's engine can still act on. The panel is shared
  // across modes, so without this it offers Upscale's controls everywhere.
  const passChoice = offersPassChoice(selected, mode);
  const ttaChoice = offersTta(selected, mode);
  const restoreLargeChoice = offersRestoreLarge(selected, mode, sourceAlreadyLarge);
  // Upscale falls back to the resampler rather than becoming unusable, but that
  // is a materially weaker result and has to be said before the job runs.
  const fallbackReason = selected?.fallback_reason ?? null;
  const tileChoices = useMemo(() => selected?.safe_tile_sizes ?? [], [selected]);
  const selectedHardware = capabilities?.hardware.find((report) =>
    selected?.device.startsWith("ComfyUI")
      ? report.scope === "comfyui"
      : report.scope === "backend",
  );
  const safeTargets = useMemo(() => {
    const targets = capabilities?.targets ?? [3840, 7680];
    if (!capabilities || !selected || !localImage) {
      const allowed = new Set(selected?.safe_targets ?? targets);
      return targets.filter((target) => allowed.has(target));
    }
    return safeTargetsForImage({
      targets,
      mode,
      capability: selected,
      hardware: selectedHardware,
      policy: capabilities.hardware_policy,
      width: localImage.width,
      height: localImage.height,
      tileSize,
      maxPasses: maxNeuralPasses,
      restoreLarge,
    });
  }, [
    capabilities,
    localImage,
    maxNeuralPasses,
    mode,
    restoreLarge,
    selected,
    selectedHardware,
    tileSize,
  ]);

  const selectMode = useCallback((next: ProcessingMode) => {
    setMode(next);
    setSharpen(MODE_SHARPEN[next]);
    if (next === "sharpen_only") {
      setTileSize(0);
      setTta(false);
      setRestoreLarge(false);
    }
  }, []);

  useEffect(() => {
    if (!capabilities || busy) return;
    const resolved = resolveAvailableMode(capabilities.modes, mode);
    if (resolved && resolved !== mode) selectMode(resolved);
  }, [busy, capabilities, mode, selectMode]);

  useEffect(() => {
    const resolved = resolveSafeTile(tileChoices, tileSize);
    if (resolved !== tileSize) setTileSize(resolved);
  }, [tileChoices, tileSize]);

  useEffect(() => {
    if (safeTargets.length && !safeTargets.includes(targetEdge)) setTargetEdge(safeTargets[0]);
  }, [safeTargets, targetEdge]);

  const loadCompletedResult = useCallback(async (snapshot: JobSnapshot) => {
    if (loadedResultFor.current === snapshot.id) return;
    loadedResultFor.current = snapshot.id;
    try {
      const blob = await fetchResult(snapshot.id);
      if (resultObjectUrl.current) URL.revokeObjectURL(resultObjectUrl.current);
      const url = URL.createObjectURL(blob);
      resultObjectUrl.current = url;
      setResultUrl(url);
      setViewMode("split");
    } catch (error) {
      loadedResultFor.current = null;
      setUiError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const run = async () => {
    if (!file) return;
    setUiError(null);
    clearResult();
    if (job) await cancelOrDeleteJob(job.id).catch(() => undefined);
    const sharpenOnly = mode === "sharpen_only";
    const settings: JobSettings = {
      target_edge: targetEdge,
      processing_mode: mode,
      sharpen,
      tile_size: sharpenOnly ? 0 : tileSize,
      // A control the panel did not offer must not reach the job record: the
      // settings are what a result is reproduced from.
      tta: ttaChoice && tta,
      restore_large: restoreLargeChoice && restoreLarge,
      max_neural_passes: maxNeuralPasses,
    };
    try {
      const created = await createJob(file, settings);
      setJob(created);
      stopObserving.current = observeJob(
        created.id,
        (snapshot) => {
          setJob(snapshot);
          if (snapshot.state === "completed") void loadCompletedResult(snapshot);
        },
        setUiError,
      );
    } catch (error) {
      setUiError(error instanceof Error ? error.message : String(error));
    }
  };

  const cancel = async () => {
    if (!job) return;
    try {
      await cancelOrDeleteJob(job.id);
    } catch (error) {
      setUiError(error instanceof Error ? error.message : String(error));
    }
  };

  const download = async () => {
    if (!job?.result || !resultUrl) return;
    try {
      const anchor = document.createElement("a");
      anchor.href = resultUrl;
      anchor.download = job.result.filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      await cancelOrDeleteJob(job.id);
      setDownloaded(true);
    } catch (error) {
      setUiError(error instanceof Error ? error.message : String(error));
    }
  };

  const resultDimensions = job?.result ?? dimensions;
  const resultMode = job?.result?.processing_mode ?? mode;
  const resultScale =
    localImage && resultDimensions
      ? Math.max(resultDimensions.width, resultDimensions.height) /
        Math.max(localImage.width, localImage.height)
      : null;
  const resultLabel = processingResultLabel(resultMode, resultScale ?? undefined);
  const plannedPasses = job?.result?.neural_passes?.length ? job.result.neural_passes : passPlan;
  // A lone 1x is not a chain. It is how an engine with no fixed enlargement
  // factor reports itself, and showing it as a pass plan invites the reader to
  // look for a multiplier that does not exist.
  const showPassPlan =
    plannedPasses.length > 0 && !(plannedPasses.length === 1 && plannedPasses[0] === 1);
  const resultIsGenerative = job?.result ? job.result.generative : Boolean(selected?.generative);

  return (
    <main className="app-shell">
      <AppHeader />
      {capabilityError ? (
        <div className="global-error">Local backend unavailable: {capabilityError}</div>
      ) : null}
      <div className="workspace-grid">
        <div className="source-column">
          {!file || !localImage ? (
            <label
              className={`drop-zone ${dragging ? "dragging" : ""}`}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                const dropped = event.dataTransfer.files[0];
                if (dropped) void chooseFile(dropped);
              }}
            >
              <input
                type="file"
                accept="image/*"
                onChange={(event) => {
                  const chosen = event.target.files?.[0];
                  if (chosen) void chooseFile(chosen);
                }}
              />
              <span className="drop-index">01</span>
              <strong>Choose an image</strong>
              <span>
                Drop, paste, or browse. PNG, JPEG, WebP, GIF, and other browser-decodable still
                formats.
              </span>
              <span className="browse-button">Browse files</span>
            </label>
          ) : (
            <>
              <div className="source-heading">
                <div>
                  <span>Source</span>
                  <strong>{file.name}</strong>
                </div>
                <label className="replace-button">
                  Replace
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(event) => {
                      const chosen = event.target.files?.[0];
                      if (chosen) void chooseFile(chosen);
                    }}
                  />
                </label>
              </div>
              <CompareStage
                originalUrl={localImage.url}
                resultUrl={resultUrl}
                width={resultDimensions?.width ?? localImage.width}
                height={resultDimensions?.height ?? localImage.height}
                mode={viewMode}
                onModeChange={setViewMode}
                split={split}
                onSplitChange={setSplit}
                pixelView={pixelView}
                onPixelViewChange={setPixelView}
                resultLabel={resultLabel}
              />
              <div className="dimension-strip">
                <span>
                  {localImage.width.toLocaleString()} × {localImage.height.toLocaleString()}
                </span>
                <span className="arrow">→</span>
                <strong>
                  {resultDimensions?.width.toLocaleString()} ×{" "}
                  {resultDimensions?.height.toLocaleString()}
                </strong>
                <span>{resultScale?.toFixed(2)}×</span>
                <span>{formatBytes(file.size)}</span>
              </div>
            </>
          )}
        </div>

        <aside className="settings-panel">
          <div className="panel-title">
            <span>Settings &amp; output</span>
            <span>{capabilities ? `v${capabilities.version}` : "connecting"}</span>
          </div>
          <div className="run-panel">
            {localImage && resultDimensions ? (
              <div className="resource-card">
                <div>
                  <span>Operation</span>
                  <b>{job?.result ? resultLabel : (selected?.name ?? "—")}</b>
                </div>
                <div>
                  <span>Output</span>
                  <b>
                    {resultDimensions.width.toLocaleString()} ×{" "}
                    {resultDimensions.height.toLocaleString()} PNG
                  </b>
                </div>
                <div>
                  <span>Working-memory estimate</span>
                  <b>{memoryEstimate ? formatBytes(memoryEstimate) : "—"}</b>
                </div>
                <div>
                  <span>Engine</span>
                  <b>{job?.result?.engine ?? selected?.engine ?? "Detecting"}</b>
                </div>
                <div>
                  <span>Processor</span>
                  <b>{selected?.device ?? "Detecting"}</b>
                </div>
                {showPassPlan ? (
                  <div>
                    <span>Neural passes</span>
                    <b>{plannedPasses.map((scale) => `${scale}×`).join(" → ")}</b>
                  </div>
                ) : null}
                {job?.result && job.result.resolved_tile_size ? (
                  <div>
                    <span>Resolved tile</span>
                    <b>{job.result.resolved_tile_size} px</b>
                  </div>
                ) : null}
              </div>
            ) : null}
            {resultIsGenerative ? <p className="inline-warning">{GENERATIVE_NOTICE}</p> : null}
            {sourceAlreadyLarge && !restoreLarge ? (
              <p className="inline-warning">
                The source already meets this target. Neural enlargement will be skipped and one
                faithful reduction will be used.
              </p>
            ) : null}
            {fallbackReason ? (
              <p className="inline-warning">
                {fallbackReason} This mode currently uses deterministic Lanczos resampling, which
                cannot add detail.{" "}
                <button type="button" onClick={() => void refreshCapabilities()}>
                  Detect again
                </button>
              </p>
            ) : null}
            {uiError ? <p className="inline-error">{uiError}</p> : null}
            <ProgressStatus job={job} hasSource={Boolean(file)} />

            <div className="actions">
              {busy ? (
                <button className="primary-button cancel" onClick={() => void cancel()}>
                  Cancel processing
                </button>
              ) : (
                <button
                  className="primary-button"
                  disabled={
                    !file || !selected?.available || (isUpscalingMode(mode) && !safeTargets.length)
                  }
                  onClick={() => void run()}
                >
                  {processingAction(mode)}
                </button>
              )}
              <button
                className="secondary-button"
                disabled={!job?.result || !resultUrl || downloaded}
                onClick={() => void download()}
              >
                {downloaded
                  ? "Downloaded · local job cleaned"
                  : job?.result
                    ? `Download PNG · ${formatBytes(job.result.bytes)}`
                    : "Download result"}
              </button>
            </div>
            {job?.result?.warnings.length ? (
              <ul className="warnings-list">
                {job.result.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            ) : null}
          </div>
          <div className="settings-body">
            {capabilities ? <HardwarePanel capabilities={capabilities} /> : null}
            <fieldset>
              <legend>What should happen</legend>
              <div className="goal-grid mode-grid">
                {modes.map((entry) => {
                  const id = entry.mode;
                  return (
                    <button
                      key={id}
                      className="goal-card mode-card"
                      aria-pressed={mode === id}
                      aria-describedby={mode === id ? "selected-mode-description" : undefined}
                      onClick={() => selectMode(id)}
                      disabled={busy}
                    >
                      <span className="goal-heading">
                        <strong>{entry.name}</strong>
                        {entry.generative ? <em>Generative</em> : null}
                      </span>
                    </button>
                  );
                })}
              </div>
              <p className="mode-description" id="selected-mode-description">
                {selected ? selected.description : "Detecting the local engine."}
              </p>
            </fieldset>

            {isUpscalingMode(mode) ? (
              <fieldset disabled={busy}>
                <legend>Target resolution</legend>
                <div className="target-grid">
                  {safeTargets.map((edge) => (
                    <button
                      key={edge}
                      className="target-card"
                      aria-pressed={targetEdge === edge}
                      onClick={() => setTargetEdge(edge)}
                    >
                      <strong>{edge === 3840 ? "4K" : edge === 7680 ? "8K" : `${edge}px`}</strong>
                      <span>{edge.toLocaleString()} px long edge</span>
                    </button>
                  ))}
                </div>
                {!safeTargets.length ? (
                  <p className="field-note">
                    No target size fits the current image and hardware policy.
                  </p>
                ) : null}
              </fieldset>
            ) : null}

            <fieldset disabled={busy}>
              <legend>Finishing</legend>
              <div className="preset-grid">
                {sharpenPresets(mode).map((preset) => (
                  <button
                    key={preset.label}
                    className="target-card"
                    aria-pressed={sharpen === preset.value}
                    onClick={() => setSharpen(preset.value)}
                  >
                    <strong>{preset.label}</strong>
                    <span>{preset.value}%</span>
                  </button>
                ))}
              </div>
              <p className="field-note">
                Luminance-only sharpening at three scales, with halos clamped. It recovers clarity
                the resize cost; it cannot add detail the pixels never implied. Fine-tune it under
                Advanced processing.
              </p>
            </fieldset>

            <details className="advanced">
              <summary>Advanced processing</summary>
              <label className="range-field">
                <span>
                  <b>Sharpening strength</b>
                  <output>{sharpen}%</output>
                </span>
                <input
                  type="range"
                  min={mode === "sharpen_only" ? 1 : 0}
                  max="100"
                  value={sharpen}
                  onChange={(event) => setSharpen(Number(event.target.value))}
                  disabled={busy}
                />
              </label>
              {sharpen >= 60 ? (
                <p className="inline-warning">
                  High sharpening can create halos and brittle-looking texture. Inspect the result
                  at 1:1.
                </p>
              ) : null}
              {tileChoices.length ? (
                <label className="select-field">
                  <span>Tile size</span>
                  <select
                    value={tileSize}
                    onChange={(event) => setTileSize(Number(event.target.value))}
                    disabled={busy || !neural}
                  >
                    {tileChoices.map((size) => (
                      <option key={size} value={size}>
                        {size ? `${size} px` : "Automatic"}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
              {ttaChoice ? (
                <label className="check-field">
                  <input
                    type="checkbox"
                    checked={tta}
                    onChange={(event) => setTta(event.target.checked)}
                    disabled={busy}
                  />
                  <span>
                    <b>Test-time augmentation</b>
                    <small>
                      Can improve difficult edges. Costs eight inferences per pass, so it compounds
                      with a chained plan.
                    </small>
                  </span>
                </label>
              ) : null}
              {restoreLargeChoice ? (
                <label className="check-field">
                  <input
                    type="checkbox"
                    checked={restoreLarge}
                    onChange={(event) => setRestoreLarge(event.target.checked)}
                    disabled={busy}
                  />
                  <span>
                    <b>Restore before reducing</b>
                    <small>
                      Run neural restoration even when the source already exceeds the target.
                    </small>
                  </span>
                </label>
              ) : null}
              {passChoice ? (
                <>
                  <label className="select-field">
                    <span>Maximum neural passes</span>
                    <select
                      value={maxNeuralPasses}
                      onChange={(event) => setMaxNeuralPasses(Number(event.target.value))}
                      disabled={busy}
                    >
                      {[1, 2, 3, 4].map((count) => (
                        <option key={count} value={count}>
                          {count === 1 ? "1 (single pass)" : `${count} passes`}
                        </option>
                      ))}
                    </select>
                  </label>
                  <p className="field-note">
                    One pass enlarges up to 4x. Beyond that the remainder falls back to a plain
                    resample, so a small source needs several chained passes to stay sharp.
                  </p>
                </>
              ) : null}
            </details>
          </div>
        </aside>
      </div>
      <footer>
        <span>All processing stays on this machine.</span>
        <span>
          Neural restoration may generate plausible detail; inspect at 1:1 before relying on it.
        </span>
      </footer>
    </main>
  );
}
