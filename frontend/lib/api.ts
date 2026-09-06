import type { AnalysisResponse, ResolutionReport, SarReport, TraceRecord } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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

export function analyzeGolden(question: string) {
  return request<AnalysisResponse>("/api/analyze", {
    method: "POST",
    body: JSON.stringify({
      scene_id: "loveda_LoveDA_images_png_0_gsd0.3",
      question,
      sensor: "LoveDA",
    }),
  });
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

export function verifyTraces() {
  return request<{ verified: boolean; message: string }>("/api/traces/verify", { method: "POST" });
}

export function getHealth() {
  return request<{ status: string; mode: string }>("/api/health");
}
