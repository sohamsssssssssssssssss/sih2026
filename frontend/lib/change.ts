import type { PlanResponse } from "./types";

/** The capability a genuine t1/t2 change request routes to. */
export const CHANGE_CAPABILITY = "change_vqa";

/** What the workspace is allowed to claim right now.
 *
 * Every variant is derived from the planner's own response. Nothing here
 * infers, scores, or summarises change: when no Change-VQA provider is
 * registered the only honest outcome is CAPABILITY_UNAVAILABLE.
 */
export type ChangeReadiness =
  | { kind: "AWAITING_PAIR"; missing: ("t1" | "t2")[]; message: string }
  | { kind: "AWAITING_QUESTION"; message: string }
  | { kind: "AWAITING_PLAN"; message: string }
  | { kind: "MISSING_INPUT"; inputs: string[]; message: string }
  | { kind: "NOT_A_CHANGE_REQUEST"; capability: string; message: string }
  | { kind: "CAPABILITY_UNAVAILABLE"; capability: string; reason: string; message: string }
  | { kind: "READY"; capability: string; provider: string; message: string };

export function changeReadiness(
  plan: PlanResponse | null,
  { t1, t2, question }: { t1: string | null; t2: string | null; question: string },
): ChangeReadiness {
  const missing: ("t1" | "t2")[] = [];
  if (!t1) missing.push("t1");
  if (!t2) missing.push("t2");
  if (missing.length > 0) {
    return {
      kind: "AWAITING_PAIR",
      missing,
      message: `Load both epochs to plan a change request. Missing: ${missing.join(", ").toUpperCase()}.`,
    };
  }
  if (!question.trim()) {
    return { kind: "AWAITING_QUESTION", message: "Ask a change question about this pair." };
  }
  if (!plan) {
    return { kind: "AWAITING_PLAN", message: "The planner has not run for this pair and question." };
  }
  if (plan.missing_inputs.length > 0) {
    return {
      kind: "MISSING_INPUT",
      inputs: plan.missing_inputs,
      message: `The planner reports missing input: ${plan.missing_inputs.join(", ")}.`,
    };
  }
  if (plan.selected_capability !== CHANGE_CAPABILITY) {
    return {
      kind: "NOT_A_CHANGE_REQUEST",
      capability: plan.selected_capability,
      message: `The planner routed this question to ${plan.selected_capability}, which reads a single scene. Only T1 would be analysed.`,
    };
  }
  if (!plan.executable || plan.unavailable_capabilities.length > 0) {
    return {
      kind: "CAPABILITY_UNAVAILABLE",
      capability: plan.selected_capability,
      reason: plan.unavailable_reason ?? "No provider is registered for the selected capability.",
      message: "No Change-VQA provider is registered, so no answer can be produced for this pair.",
    };
  }
  return {
    kind: "READY",
    capability: plan.selected_capability,
    provider: plan.provider ?? "UNKNOWN",
    message: `Change analysis is executable via ${plan.provider ?? "UNKNOWN"}.`,
  };
}

/** True only when a real model could run. Guards the Run button. */
export function canRunChangeAnalysis(readiness: ChangeReadiness): boolean {
  return readiness.kind === "READY";
}

export interface PipelineStage {
  step_id: string;
  capability: string;
  provider: string | null;
  available: boolean;
  dependsOn: string[];
  requiredInputs: string[];
}

/** The execution plan's own steps, unmodified.
 *
 * The combined temporal+localization rule decomposes into
 * change_vqa → grounding; each stage reports its own real provider state,
 * so an available downstream stage never implies the chain can run. */
export function changePipeline(plan: PlanResponse | null): PipelineStage[] {
  if (!plan) return [];
  return plan.steps.map((step) => ({
    step_id: step.step_id,
    capability: step.capability,
    provider: step.provider,
    available: step.provider_available,
    dependsOn: step.depends_on,
    requiredInputs: step.required_inputs,
  }));
}

/** A pair is comparable pixel-for-pixel only at identical dimensions.
 *  Mismatched epochs are still shown; the caller is told they are not aligned. */
export function pairAlignment(
  t1: { width: number; height: number } | null,
  t2: { width: number; height: number } | null,
): { aligned: boolean; note: string | null } {
  if (!t1 || !t2) return { aligned: false, note: null };
  if (t1.width === t2.width && t1.height === t2.height) return { aligned: true, note: null };
  return {
    aligned: false,
    note: `Epoch dimensions differ (${t1.width}×${t1.height} vs ${t2.width}×${t2.height}). Synchronised views are not pixel-aligned and no co-registration is performed.`,
  };
}
