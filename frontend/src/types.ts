export type ProcessingMode = "upscale" | "illustration" | "sharpen_only";
export type JobState =
  | "queued"
  | "analyzing"
  | "loading_model"
  | "enhancing"
  | "finishing"
  | "encoding"
  | "cancelling"
  | "completed"
  | "failed"
  | "cancelled";

export interface ModeCapability {
  mode: ProcessingMode;
  name: string;
  description: string;
  available: boolean;
  /** Whether this mode invents detail. Always false here; see ACCEPTABLE_USE.md. */
  generative: boolean;
  engine: string;
  device: string;
  unavailable_reason: string | null;
  fallback_reason: string | null;
  max_passes: number;
  native_scales: number[];
  /** Whether test-time augmentation reaches this mode's engine at all. */
  supports_tta?: boolean;
  resource_requirement?: ResourceRequirement | null;
  safe_tile_sizes?: number[];
  safe_targets?: number[];
}

export interface ResourceRequirement {
  ram_mib: number;
  vram_mib: number;
  unified_mib: number;
}

export interface HardwareReport {
  scope: "backend" | "comfyui" | string;
  ram_physical_mib: number | null;
  ram_effective_mib: number | null;
  ram_available_mib: number | null;
  gpu_name: string | null;
  vram_total_mib: number | null;
  vram_available_mib: number | null;
  memory_kind: "dedicated" | "unified" | string;
  source: string;
  warnings: string[];
}

export interface HardwarePolicyInfo {
  mode: "safe" | "off" | string;
  version: number;
  ram_reserve_mib: number;
  vram_reserve_mib: number;
  visibility_basis: string;
  admission_basis: string;
}

export interface FeatureExclusion {
  id: string;
  name: string;
  reason: string;
}

export interface Capabilities {
  version: string;
  modes: ModeCapability[];
  targets: number[];
  max_upload_bytes: number;
  max_input_pixels: number;
  platform: Record<string, string | number | boolean | null>;
  hardware: HardwareReport[];
  hardware_policy: HardwarePolicyInfo;
  excluded_features: FeatureExclusion[];
}

export interface JobSettings {
  target_edge: number;
  processing_mode: ProcessingMode;
  sharpen: number;
  tile_size: number;
  tta: boolean;
  restore_large: boolean;
  max_neural_passes: number;
  workflow?: string | null;
}

export interface SourceInfo {
  filename: string;
  width: number;
  height: number;
  mode: string;
  format: string | null;
  animated: boolean;
  frames: number;
  has_alpha: boolean;
  has_icc: boolean;
  bit_depth: number;
  warnings: string[];
}

export interface ResultInfo {
  width: number;
  height: number;
  bytes: number;
  engine: string;
  processing_mode: ProcessingMode;
  filename: string;
  neural_passes: number[];
  resolved_tile_size: number;
  generative: boolean;
  warnings: string[];
}

export interface JobSnapshot {
  id: string;
  state: JobState;
  phase: string;
  message: string;
  progress: number | null;
  settings: JobSettings;
  source: SourceInfo | null;
  result: ResultInfo | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  revision: number;
}

export interface LocalImageInfo {
  width: number;
  height: number;
  url: string;
}
