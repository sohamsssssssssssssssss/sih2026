export type ExecutionMode = "live" | "cached_result";

export type OperationState = "IDLE" | "LOADING" | "SUCCESS" | "UNAVAILABLE" | "INVALID INPUT" | "EXECUTION ERROR";

export interface AnalysisRequest {
  scene_id: string;
  question: string;
  sensor?: string | null;
  capability?: string | null;
  /** Second scene for pairwise capabilities (change_vqa, optical_sar).
   *  Omitted for every single-scene request, which is unchanged. */
  scene_id_2?: string | null;
}

export interface SceneUploadResponse {
  scene_id: string;
  filename: string;
  format: "PNG" | "JPEG";
  width: number;
  height: number;
  sensor: string | null;
  gsd: string | null;
  location: string | null;
  acquisition_date: string | null;
}

export interface CatalogScene {
  id: string;
  title: string;
  capability: string;
  result_state: "real_live" | "cached_real" | "prototype";
  catalog_visible: boolean;
  available: boolean;
  unavailable_reason: string | null;
  question: string;
  evaluated_expression: string;
  source: {
    dataset: string;
    dataset_split: string | null;
    source_id: string;
    sensor: string | null;
    gsd: number | null;
    location: string | null;
    acquisition_date: string | null;
  };
}

export interface SceneCatalog { version: string; scenes: CatalogScene[] }

export interface CapabilityStatus {
  name: string;
  available: boolean;
  provider: string | null;
}

export interface PlanResponse {
  planner_version: string;
  rule_id: string;
  requested_capability: string | null;
  selected_capability: string;
  executable: boolean;
  reason: string;
  required_inputs: string[];
  missing_inputs: string[];
  provider_available: boolean;
  provider: string | null;
  unavailable_reason: string | null;
  execution_plan_version: string;
  steps: {
    step_id: string;
    capability: string;
    depends_on: string[];
    required_inputs: string[];
    provider_available: boolean;
    provider: string | null;
  }[];
  unavailable_capabilities: string[];
}

export interface TraceVerification { verified: boolean; message: string }
export interface TraceHistory { records: TraceRecord[]; count: number }

export interface TraceRecord {
  model_name: string;
  model_version: string;
  params: {
    execution_mode: ExecutionMode;
    results_artifact?: string;
    scene_id?: string;
    sensor?: string | null;
    capability?: string;
    requested_capability?: string | null;
    planner_version?: string;
    planner_rule?: string;
    result_state?: string;
    source_scene_id?: string;
    evaluated_expression?: string;
    source_run_id?: string | null;
    source_git_sha?: string | null;
    source_working_tree_sha256?: string | null;
  };
  input_summary: { image_paths: string[]; question: string; n_images: number };
  timestamp_iso: string;
  record_hash: string;
  prev_hash: string;
}

/** One raw evidence item exactly as the backend emitted it.
 *
 * `AnalyzeResponse.evidence` is `list[dict[str, Any]] | None`, so the wire
 * shape is deliberately open: grounding providers emit `bounding_box`
 * records, VQA providers emit an empty list, and future capabilities may
 * emit shapes this build does not know. Fields stay optional and unnarrowed
 * here; `lib/evidence.ts` validates them before anything is drawn.
 */
export interface EvidenceRecord {
  type?: string;
  label?: string;
  coordinates?: unknown;
  coordinate_space?: string;
  confidence?: number | null;
  source_scene_id?: string | null;
  [key: string]: unknown;
}

export interface AnalysisResponse {
  answer: string;
  /** Absent on cached results and on any provider that reports no evidence. */
  evidence?: EvidenceRecord[] | null;
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
  assets: {
    gsd: number;
    available: boolean;
    unavailable_reason: string | null;
    width: number | null;
    height: number | null;
  }[];
}

export interface SarReport {
  scene: string;
  title: string;
  human_validation: boolean;
  render_available: boolean;
  summaries: Record<"water" | "built_up" | "vegetation" | "terrain", string>;
  annotation: string;
}
