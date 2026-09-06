export type ExecutionMode = "live" | "cached_result";

export interface TraceRecord {
  model_name: string;
  model_version: string;
  params: {
    execution_mode: ExecutionMode;
    results_artifact?: string;
    scene_id?: string;
    sensor?: string;
  };
  input_summary: { image_paths: string[]; question: string; n_images: number };
  timestamp_iso: string;
  record_hash: string;
  prev_hash: string;
}

export interface AnalysisResponse {
  answer: string;
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
