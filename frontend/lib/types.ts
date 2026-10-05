export type ExecutionMode = "live" | "cached_result";

export interface TraceRecord {
  model_name: string;
  model_version: string;
  params: {
    execution_mode: ExecutionMode;
    results_artifact?: string;
    scene_id?: string;
    scene_id_2?: string;
    capability?: string;
    sensor?: string | null;
  };
  input_summary: { image_paths: string[]; question: string; n_images: number };
  timestamp_iso: string;
  record_hash: string;
  prev_hash: string;
}

/** One structured evidence item (grounding box, change statistics, optical-SAR summary, ...). */
export type EvidenceItem = Record<string, unknown>;

export interface AnalysisResponse {
  answer: string;
  evidence?: EvidenceItem[];
  execution_mode: ExecutionMode;
  results_artifact: string | null;
  model: { name: string; version: string };
  trace: TraceRecord;
  notice: string;
}

export interface RungMetrics {
  n: number;
  accuracy: number;
  binary_n: number;
  binary_accuracy: number;
  open_n: number;
  open_accuracy: number;
  pred_yes_rate_on_binary: number;
  warning: string | null;
}

export interface ResolutionReport {
  model: string;
  n_samples: number;
  timestamp: string;
  provenance?: string;
  per_rung: Record<string, RungMetrics>;
  degenerate_rungs: string[];
}

export interface SarReport {
  scene: string;
  title: string;
  human_validation: boolean;
  render_available: boolean;
  summaries: Record<"water" | "built_up" | "vegetation" | "terrain", string>;
  annotation: string;
}

export type SceneModality = "optical" | "multispectral" | "sar" | "unknown";

/** Curated scene-pack entry from GET /api/scenes. */
export interface CuratedScene {
  id: string;
  title: string;
  capability: string;
  result_state: string;
  catalog_visible: boolean;
  available: boolean;
  unavailable_reason: string | null;
  question: string;
  evaluated_expression?: unknown;
  source: {
    dataset?: string | null;
    dataset_split?: string | null;
    source_id?: string | null;
    sensor?: string | null;
    gsd?: number | null;
    location?: string | null;
    acquisition_date?: string | null;
  };
}

/** A scene previously uploaded through POST /api/scenes. */
export interface UploadedScene {
  scene_id: string;
  filename: string;
  format: string;
  width: number;
  height: number;
  modality: string;
  sensor: string | null;
  acquisition_time: string | null;
  has_native_raster: boolean;
  georeferenced: boolean;
  uploaded_at: string;
}

export interface SceneCatalog {
  version: string;
  scenes: CuratedScene[];
  uploads: UploadedScene[];
}

/** 201 body of POST /api/scenes. */
export interface SceneUploadResponse {
  scene_id: string;
  filename: string;
  format: "PNG" | "JPEG" | "TIFF";
  width: number;
  height: number;
  sensor: string | null;
  gsd: string | null;
  location: string | null;
  acquisition_date: string | null;
}

export interface SceneUploadMetadata {
  modality?: SceneModality;
  sensor?: string;
  /** ISO 8601 with an explicit timezone. */
  acquisition_timestamp?: string;
  polarization?: string;
}
