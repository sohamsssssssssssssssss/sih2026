import type {
  AnalysisResponse, ExecutionMode, ResolutionReport, SarReport, SceneCatalog, SceneUploadMetadata,
  SceneUploadResponse, TraceRecord, UploadedScene,
} from "./types";
import {
  GOLDEN_SCENE, analysisFailure, describeFailure, parseFailureDetail, validateAnalysis, type FailureDetail,
} from "./workspace-contract";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function getSceneImageUrl(sceneId: string) {
  return `${API_URL}/api/scenes/${encodeURIComponent(sceneId)}/image`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export class AnalysisError extends Error {
  readonly status: number;
  readonly detail: FailureDetail;
  constructor(status: number, detail: FailureDetail) {
    super(describeFailure(detail));
    this.name = "AnalysisError";
    this.status = status;
    this.detail = detail;
  }
}

export interface AnalyzeRequest {
  sceneId: string;
  question: string;
  sceneId2?: string | null;
  /** Always sent explicitly. Cached replay is only ever a deliberate user choice. */
  executionMode: ExecutionMode;
  sensor?: string | null;
  capability?: string | null;
  signal?: AbortSignal;
}

export async function analyze({ sceneId, question, sceneId2, executionMode, sensor, capability, signal }: AnalyzeRequest): Promise<AnalysisResponse> {
  const trimmed = question.trim();
  const body: Record<string, string> = { scene_id: sceneId, question: trimmed, execution_mode: executionMode };
  if (sceneId2) body.scene_id_2 = sceneId2;
  if (sensor) body.sensor = sensor;
  if (capability) body.capability = capability;
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      signal,
      body: JSON.stringify(body),
    });
  } catch {
    throw new AnalysisError(0, { kind: "message", message: signal?.aborted ? "Analysis request stopped or timed out. No answer was displayed." : analysisFailure(0) });
  }
  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null);
    throw new AnalysisError(response.status, parseFailureDetail(response.status, payload));
  }
  const payload: unknown = await response.json().catch(() => null);
  return validateAnalysis(payload, trimmed, { sceneId, sceneId2 });
}

/** The pinned golden scene. Live by default; cached replay must be requested explicitly. */
export function analyzeGolden(question: string, signal?: AbortSignal, executionMode: ExecutionMode = "live"): Promise<AnalysisResponse> {
  return analyze({ sceneId: GOLDEN_SCENE.id, question, executionMode, sensor: "Unknown", signal });
}

export async function getScenes(): Promise<SceneCatalog> {
  const body = await request<Partial<SceneCatalog>>("/api/scenes");
  return {
    version: typeof body.version === "string" ? body.version : "unknown",
    scenes: Array.isArray(body.scenes) ? body.scenes : [],
    // Older backends predate the uploads listing; treat it as empty rather than failing.
    uploads: Array.isArray(body.uploads) ? body.uploads : [],
  };
}

export function getUploadedScene(sceneId: string) {
  return request<UploadedScene>(`/api/scenes/uploads/${encodeURIComponent(sceneId)}`);
}

export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

export class UploadError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "UploadError";
    this.status = status;
  }
}

export async function uploadScene(file: File, metadata: SceneUploadMetadata = {}, signal?: AbortSignal): Promise<SceneUploadResponse> {
  const form = new FormData();
  form.append("file", file, file.name);
  for (const [key, value] of Object.entries(metadata)) {
    if (typeof value === "string" && value.trim()) form.append(key, value.trim());
  }
  let response: Response;
  try {
    // No Content-Type header: the browser must set the multipart boundary itself.
    response = await fetch(`${API_URL}/api/scenes`, { method: "POST", body: form, cache: "no-store", signal });
  } catch {
    throw new UploadError(0, signal?.aborted ? "Upload cancelled." : "Could not reach the SatQuery API. The scene was not uploaded.");
  }
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: unknown } | null;
    const detail = typeof payload?.detail === "string" && payload.detail.trim() ? payload.detail.trim() : null;
    if (response.status === 413) throw new UploadError(413, detail ?? "The uploaded image exceeds the 20 MiB limit.");
    if (detail && (response.status === 422 || response.status === 500)) throw new UploadError(response.status, detail);
    throw new UploadError(response.status, `Upload failed (${response.status}). The scene was not stored.`);
  }
  return response.json() as Promise<SceneUploadResponse>;
}

export function getResolution() {
  return request<ResolutionReport>("/api/resolution");
}

export function getSar(scene = "mumbai-coastal") {
  return request<SarReport>(`/api/sar/${scene}`);
}

export function getTraces() {
  return request<{ records: TraceRecord[]; count: number }>("/api/traces");
}

export async function verifyTraces() {
  const response = await request<{ verified: unknown }>("/api/traces/verify", { method: "POST", signal: AbortSignal.timeout(10000) });
  if (typeof response.verified !== "boolean") throw new Error("Integrity status unavailable.");
  return { verified: response.verified, message: response.verified ? "Current process-local chain verified." : "Integrity verification failed for the current process-local chain." };
}

export type HealthStatus = "ready" | "degraded" | "unavailable";

export interface HealthResponse {
  status: HealthStatus;
  mode: string;
  checks?: {
    trace: { ok: boolean; detail: string | null };
    capabilities: Record<string, { available: boolean; reason_code: string | null }>;
  };
}

// A 503 from /api/health still carries a HealthResponse body (trace unusable),
// so read it instead of reporting the backend as unreachable.
export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/api/health`, { cache: "no-store" });
  const body = (await response.json().catch(() => null)) as HealthResponse | null;
  if (body && typeof body.status === "string") return body;
  throw new Error(`Request failed (${response.status})`);
}
