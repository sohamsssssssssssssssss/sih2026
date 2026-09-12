import type { AnalysisResponse, ResolutionReport, SarReport, TraceRecord } from "./types";
import { GOLDEN_SCENE, analysisFailure, validateAnalysis } from "./workspace-contract";

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

export async function analyzeGolden(question: string, signal?: AbortSignal): Promise<AnalysisResponse> {
  let response: Response;
  try { response = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    signal,
    body: JSON.stringify({
      scene_id: GOLDEN_SCENE.id,
      question,
      sensor: "Unknown",
    }),
  }); } catch { throw new Error(signal?.aborted ? "Analysis request stopped or timed out. No answer was displayed." : analysisFailure(0)); }
  if (!response.ok) throw new Error(analysisFailure(response.status));
  const payload: unknown = await response.json().catch(() => null);
  return validateAnalysis(payload, question);
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

export function getHealth() {
  return request<{ status: string; mode: string }>("/api/health");
}
