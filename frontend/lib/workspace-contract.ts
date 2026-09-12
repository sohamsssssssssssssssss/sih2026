import type { AnalysisResponse, ExecutionMode } from "./types";

// Verified against demo_gui/golden_assets.py and the committed ladder artifact.
// Phase D should expose this metadata through the existing API.
export const GOLDEN_SCENE = {
  id: "loveda_LoveDA_images_png_0_gsd0.3",
  question: "Is there a building in this image?",
  source: "LoveDA",
  gsd: 0.3,
} as const;

export function isExecutionMode(value: unknown): value is ExecutionMode {
  return value === "live" || value === "cached_result";
}

export function executionLabel(value: unknown) {
  return value === "live" ? "LIVE INFERENCE" : value === "cached_result" ? "VERIFIED CACHED RESULT" : "UNKNOWN EXECUTION MODE";
}

const object = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;

export function validateAnalysis(value: unknown, question: string): AnalysisResponse {
  if (!object(value) || !isExecutionMode(value.execution_mode)) throw new Error("Unknown execution mode. No validated answer was displayed.");
  const { trace, model } = value;
  if (!text(value.answer) || !object(model) || !text(model.name) || !text(model.version)
    || !object(trace) || !object(trace.params) || !object(trace.input_summary)) {
    throw new Error("Analysis response is incomplete. No validated answer was displayed.");
  }
  if (!isExecutionMode(trace.params.execution_mode) || trace.params.execution_mode !== value.execution_mode) {
    throw new Error("Unknown or inconsistent execution mode. No validated answer was displayed.");
  }
  if (trace.params.scene_id !== GOLDEN_SCENE.id || trace.input_summary.question !== question
    || trace.model_name !== model.name || trace.model_version !== model.version) {
    throw new Error("Execution provenance does not match this request. No validated answer was displayed.");
  }
  if (!text(trace.timestamp_iso) || !Number.isFinite(Date.parse(trace.timestamp_iso))
    || !text(trace.record_hash) || !/^[a-f0-9]{64}$/.test(trace.record_hash)
    || typeof trace.prev_hash !== "string" || (trace.prev_hash !== "" && !/^[a-f0-9]{64}$/.test(trace.prev_hash))) {
    throw new Error("Execution provenance is incomplete. No validated answer was displayed.");
  }
  if ((value.results_artifact != null && !text(value.results_artifact))
    || (trace.params.results_artifact != null && !text(trace.params.results_artifact))
    || !Array.isArray(trace.input_summary.image_paths) || !trace.input_summary.image_paths.every(path => typeof path === "string")
    || typeof trace.input_summary.n_images !== "number" || !Number.isInteger(trace.input_summary.n_images) || trace.input_summary.n_images < 0) {
    throw new Error("Execution provenance is malformed. No validated answer was displayed.");
  }
  if (value.execution_mode === "cached_result" && (!text(value.results_artifact) || value.results_artifact !== trace.params.results_artifact)) {
    throw new Error("Cached result is missing artifact provenance. No validated answer was displayed.");
  }
  return value as unknown as AnalysisResponse;
}

export function analysisFailure(status: number) {
  if (status === 422) return "Analysis unavailable: live inference could not complete or no cached result matched. No answer was generated.";
  if (status === 503) return "Required analysis artifacts or service are unavailable. No answer was generated.";
  return "Analysis service unavailable. Please try again. No answer was displayed.";
}

export function integrityState(verified: unknown): "verified" | "failed" | "unknown" {
  return verified === true ? "verified" : verified === false ? "failed" : "unknown";
}
