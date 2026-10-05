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

/** Mirrors backend `normalize_scene_id`: the trace records the normalized identifier. */
export function normalizeSceneId(sceneId: string) {
  const stripped = /\.(png|jpe?g|tiff?)$/i.test(sceneId) ? sceneId.replace(/\.[^.]+$/, "") : sceneId;
  return stripped.replace("loveda_Train_Rural_images_png_", "loveda_LoveDA_images_png_");
}

/**
 * The only request the backend can replay from a committed artifact. Cached replay is
 * offered for this exact scene + question and nothing else; it is never automatic.
 */
export function isGoldenRequest(sceneId: string, question: string, sceneId2?: string | null) {
  return sceneId === GOLDEN_SCENE.id && question.trim() === GOLDEN_SCENE.question && !sceneId2;
}

export interface ExpectedScenes { sceneId: string; sceneId2?: string | null }

export function validateAnalysis(value: unknown, question: string, expected: ExpectedScenes = { sceneId: GOLDEN_SCENE.id }): AnalysisResponse {
  if (!object(value) || !isExecutionMode(value.execution_mode)) throw new Error("Unknown execution mode. No validated answer was displayed.");
  const { trace, model } = value;
  if (!text(value.answer) || !object(model) || !text(model.name) || !text(model.version)
    || !object(trace) || !object(trace.params) || !object(trace.input_summary)) {
    throw new Error("Analysis response is incomplete. No validated answer was displayed.");
  }
  if (!isExecutionMode(trace.params.execution_mode) || trace.params.execution_mode !== value.execution_mode) {
    throw new Error("Unknown or inconsistent execution mode. No validated answer was displayed.");
  }
  if (trace.params.scene_id !== normalizeSceneId(expected.sceneId)
    || (expected.sceneId2 ? trace.params.scene_id_2 !== normalizeSceneId(expected.sceneId2) : trace.params.scene_id_2 != null)
    || trace.input_summary.question !== question
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
  if (value.evidence != null && (!Array.isArray(value.evidence) || !value.evidence.every(object))) {
    throw new Error("Analysis evidence is malformed. No validated answer was displayed.");
  }
  return value as unknown as AnalysisResponse;
}

export interface FailedCheck { code: string; message: string }

/** A readable, structured view of a non-2xx /api/analyze response. */
export type FailureDetail =
  | { kind: "message"; message: string }
  | { kind: "provider"; capability: string | null; provider: string | null; reasonCode: string; detail: string | null }
  | { kind: "pair"; workflow: string | null; failedChecks: FailedCheck[]; warnings: FailedCheck[]; unknownMetadata: string[] };

// Statuses whose string `detail` is authored by the API for users. Anything else
// (500, unexpected proxies) may carry internals, so it is replaced by a fixed message.
const USER_FACING_STATUSES = new Set([404, 413, 422, 503]);

const checks = (value: unknown): FailedCheck[] => Array.isArray(value)
  ? value.filter(object).map(item => ({ code: String(item.code ?? "unknown"), message: text(item.message) ? item.message : String(item.code ?? "Unspecified check") }))
  : [];

export function parseFailureDetail(status: number, body: unknown): FailureDetail {
  const detail = object(body) ? body.detail : undefined;
  if (object(detail) && text(detail.reason_code)) {
    return {
      kind: "provider",
      capability: text(detail.capability) ? detail.capability : null,
      provider: text(detail.provider) ? detail.provider : null,
      reasonCode: detail.reason_code,
      detail: text(detail.detail) ? detail.detail : null,
    };
  }
  if (object(detail) && Array.isArray(detail.failed_checks)) {
    return {
      kind: "pair",
      workflow: text(detail.requested_workflow) ? detail.requested_workflow : null,
      failedChecks: checks(detail.failed_checks),
      warnings: checks(detail.warnings),
      unknownMetadata: Array.isArray(detail.unknown_metadata) ? detail.unknown_metadata.filter(text) : [],
    };
  }
  if (text(detail) && USER_FACING_STATUSES.has(status)) return { kind: "message", message: detail.trim().slice(0, 500) };
  return { kind: "message", message: analysisFailure(status) };
}

export function describeFailure(detail: FailureDetail) {
  if (detail.kind === "message") return detail.message;
  if (detail.kind === "provider") {
    const who = [detail.capability, detail.provider].filter(Boolean).join(" via ");
    return `${who || "The selected capability"} is not ready (${detail.reasonCode})${detail.detail ? `: ${detail.detail}` : "."}`;
  }
  const failed = detail.failedChecks.map(check => check.message).join(" ");
  return `Scene pair is not eligible${detail.workflow ? ` for ${detail.workflow}` : ""}.${failed ? ` ${failed}` : ""}`;
}

export function analysisFailure(status: number) {
  if (status === 422) return "Analysis unavailable: live inference could not complete or no cached result matched. No answer was generated.";
  if (status === 503) return "Required analysis artifacts or service are unavailable. No answer was generated.";
  return "Analysis service unavailable. Please try again. No answer was displayed.";
}

export function integrityState(verified: unknown): "verified" | "failed" | "unknown" {
  return verified === true ? "verified" : verified === false ? "failed" : "unknown";
}
