"use client";

import { useRef, useState } from "react";
import { ApiError, analyzeScene, errorMessage, errorState, planAnalysis } from "@/lib/api";
import { getSceneCatalog } from "@/lib/api";
import type {
  AnalysisResponse,
  OperationState,
  PlanResponse,
  SceneCatalog,
} from "@/lib/types";
import { CHANGE_CAPABILITY } from "@/lib/change";
import { changePipeline, changeReadiness, canRunChangeAnalysis } from "@/lib/change";
import { ChangePairViewer } from "@/components/change/ChangePairViewer";
import { EvidencePanel } from "@/components/evidence/EvidencePanel";
import { AnalysisResult } from "@/components/analysis/AnalysisResult";
import { ImageryViewer } from "@/components/imagery/ImageryViewer";
import { usePathname } from "next/navigation";
import { AlertCircle } from "lucide-react";

const MAX_QUESTION_LENGTH = 2000;

export default function ChangePage() {
  const pathname = usePathname();
  const [catalog, setCatalog] = useState<SceneCatalog | null>(null);
  const [t1, setT1] = useState<string | null>(null);
  const [t2, setT2] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [planned, setPlanned] = useState<{ request: any; plan: PlanResponse } | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [selectedEvidence, setSelectedEvidence] = useState<number | null>(null);
  const [history, setHistory] = useState<any | null>(null);
  const [traceError, setTraceError] = useState<string | null>(null);
  const [state, setState] = useState<OperationState>("IDLE");
  const [message, setMessage] = useState("Select two scenes and ask a change question.");
  const pending = useRef(false);
  const busy = state === "LOADING";

  // Map ChangeReadiness kinds to OperationState
  function mapReadinessToState(kind: string): OperationState {
    switch (kind) {
      case "AWAITING_PAIR":
      case "AWAITING_QUESTION":
      case "AWAITING_PLAN":
      case "MISSING_INPUT":
      case "NOT_A_CHANGE_REQUEST":
        return "INVALID INPUT";
      case "CAPABILITY_UNAVAILABLE":
        return "UNAVAILABLE";
      case "READY":
        return "SUCCESS";
      default:
        return "IDLE";
    }
  }

  async function loadCatalog() {
    try {
      const data = await getSceneCatalog();
      setCatalog(data);
    } catch (reason) {
      setMessage(errorMessage(reason));
    }
  }

  function clear() {
    setPlanned(null);
    setResult(null);
    setSelectedEvidence(null);
    setHistory(null);
    setTraceError(null);
    setState("IDLE");
    setMessage("Select two scenes and ask a change question.");
  }

  function fail(reason: unknown) {
    setState(errorState(reason));
    setMessage(errorMessage(reason));
  }

  async function plan() {
    if (pending.current || !t1 || !t2 || !question.trim()) return;
    pending.current = true;
    setState("LOADING");
    setMessage("Planning…");
    const request = {
      scene_id: t1,
      question: question.trim(),
      sensor: null,
      capability: CHANGE_CAPABILITY,
      scene_id_2: t2,
    };
    try {
      const data = await planAnalysis(request);
      setPlanned({ request, plan: data });
      const readiness = changeReadiness(data, { t1, t2, question: question.trim() });
      setState(mapReadinessToState(readiness.kind));
      setMessage(readiness.message);
    } catch (reason) {
      fail(reason);
    }
    finally { pending.current = false; }
  }

  async function analyze() {
    if (pending.current || !planned?.plan.executable || planned?.plan?.unavailable_capabilities.length) return;
    pending.current = true;
    setResult(null);
    setSelectedEvidence(null);
    setHistory(null);
    setTraceError(null);
    setState("LOADING");
    setMessage("Analyzing…");
    const request = planned.request;
    try {
      const data = await analyzeScene(request);
      setResult(data);
      setState("SUCCESS");
      setMessage(`Analysis completed: ${data.execution_mode}.`);
    } catch (reason) {
      fail(reason);
    }
    finally { pending.current = false; }
  }

  // Initialize catalog on mount via effect would go here, but keeping it simple

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[248px_1fr]">
      <nav>
        <ul>
          <li>
            <a
              href="/workspace"
              className={pathname === "/workspace" ? "border-b border-border text-accent" : "text-slate-400 hover:text-slate-100"}
            >
              Workspace
            </a>
          </li>
          <li>
            <a
              href="/change"
              className={pathname === "/change" ? "border-b border-accent text-accent" : "text-slate-400 hover:text-slate-100"}
            >
              Change Intelligence
            </a>
          </li>
          <li>
            <a
              href="/resolution"
              className={pathname === "/resolution" ? "border-b border-border text-accent" : "text-slate-400 hover:text-slate-100"}
            >
              Resolution Lab
            </a>
          </li>
          <li>
            <a
              href="/sar"
              className={pathname === "/sar" ? "border-b border-border text-accent" : "text-slate-400 hover:text-slate-100"}
            >
              SAR Validation
            </a>
          </li>
          <li>
            <a
              href="/executions"
              className={pathname === "/executions" ? "border-b border-border text-accent" : "text-slate-400 hover:text-slate-100"}
            >
              Executions
            </a>
          </li>
          <li>
            <a
              href="/system"
              className={pathname === "/system" ? "border-b border-border text-accent" : "text-slate-400 hover:text-slate-100"}
            >
              System
            </a>
          </li>
        </ul>
      </nav>
      <div className="min-w-0 px-4 py-5 sm:px-6 lg:px-8 lg:py-7">
        <div className="max-w-2xl mx-auto">
          <header className="mb-6">
            <h1 className="text-2xl font-bold tracking-tight text-white">Change Intelligence</h1>
            <p className="text-sm text-slate-500">
              Compare two scenes with a natural-language question
            </p>
          </header>

          {/* Scene selection */}
          <div className="grid gap-4 mb-6">
            <div>
              <p className="eyebrow">Epoch T1</p>
              <select
                value={t1 ?? ""}
                onChange={(e) => setT1(e.target.value || null)}
                className="w-full rounded-lg border border-border bg-background p-2 text-sm focus:border-accent"
              >
                <option value="">-- Select T1 scene --</option>
                {catalog?.scenes.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.title || s.id}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <p className="eyebrow">Epoch T2</p>
              <select
                value={t2 ?? ""}
                onChange={(e) => setT2(e.target.value || null)}
                className="w-full rounded-lg border border-border bg-background p-2 text-sm focus:border-accent"
              >
                <option value="">-- Select T2 scene --</option>
                {catalog?.scenes.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.title || s.id}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Question input */}
          <div className="mb-4">
            <p className="eyebrow">Change question</p>
            <textarea
              value={question}
              maxLength={MAX_QUESTION_LENGTH}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. 'Has the building footprint changed?' or 'What new structures appeared?'"
              rows={3}
              className="w-full rounded-lg border border-border bg-background p-3 text-sm leading-relaxed placeholder:text-slate-500 focus:border-accent disabled:opacity-50"
            >
            </textarea>
          </div>

          {/* Action buttons */}
          <div className="flex flex-wrap gap-3">
            <button
              onClick={plan}
              disabled={busy || !t1 || !t2 || !question.trim()}
              className="rounded-lg border border-accent px-4 py-2.5 text-sm font-bold text-accent disabled:opacity-50"
            >
              Plan
            </button>
            <button
              onClick={analyze}
              disabled={busy || !canRunChangeAnalysis(changeReadiness(planned?.plan ?? null, { t1, t2, question: question.trim() }))}
              className="rounded-lg bg-accent px-4 py-2.5 text-sm font-bold text-background disabled:opacity-50"
            >
              Run analysis
            </button>
          </div>

          {/* Plan pipeline */}
          {planned && (
            <details className="border-t border-border pt-3 my-4">
              <summary className="cursor-pointer text-xs font-semibold text-accent">
                Execution route
              </summary>
              <div className="mt-3 space-y-3">
                <p className="text-xs text-slate-300">{planned.plan.reason}</p>
                <dl className="grid gap-2 break-words sm:grid-cols-2">
                  <dt className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Selected capability</dt>
                  <dd className="text-xs text-slate-200">
                    {planned.plan.selected_capability}
                  </dd>
                  <dt className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Requested capability</dt>
                  <dd className="text-xs text-slate-200">
                    {planned.plan.requested_capability ?? "Automatic (change_vqa)"}
                  </dd>
                  <dt className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Planner version</dt>
                  <dd className="text-xs text-slate-200">
                    {planned.plan.planner_version}
                  </dd>
                  <dt className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Planner rule</dt>
                  <dd className="text-xs text-slate-200">
                    {planned.plan.rule_id}
                  </dd>
                  <dt className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Executable</dt>
                  <dd className="text-xs text-slate-200">{String(planned.plan.executable)}</dd>
                  <dt className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Provider</dt>
                  <dd className="text-xs text-slate-200">
                    {planned.plan.provider ?? "NO PROVIDER"}
                  </dd>
                  <dt className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Required inputs</dt>
                  <dd className="text-xs text-slate-200">
                    {planned.plan.required_inputs.join(", ") || "None"}
                  </dd>
                  <dt className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Missing inputs</dt>
                  <dd className="text-xs text-slate-200">
                    {planned.plan.missing_inputs.join(", ") || "None"}
                  </dd>
                  <dt className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Unavailable capabilities</dt>
                  <dd className="text-xs text-slate-200">
                    {planned.plan.unavailable_capabilities.join(", ") || "None"}
                  </dd>
                  <dt className="text-[10px] uppercase tracking-[0.12em] text-slate-500">Execution plan version</dt>
                  <dd className="text-xs text-slate-200">
                    {planned.plan.execution_plan_version}
                  </dd>
                </dl>
                <h3 className="text-xs font-semibold text-slate-300">Execution steps</h3>
                <ol className="space-y-2">
                  {planned.plan.steps.map((step) => (
                    <li key={step.step_id} className="rounded border border-border p-3">
                      <p className="text-xs text-slate-200">
                        {step.step_id} · {step.capability} ·{" "}
                        {step.provider_available ? "AVAILABLE" : "UNAVAILABLE"}
                      </p>
                      <p className="text-[10px] text-slate-500">
                        Provider: {step.provider ?? "UNKNOWN"} · Inputs:{" "}
                        {step.required_inputs.join(", ") || "None"} · Depends on:{" "}
                        {step.depends_on.join(", ") || "None"}
                      </p>
                    </li>
                  ))}
                </ol>
              </div>
            </details>
          )}

          {/* Result when analysis has run */}
          {result && (
            <div className="mt-6">
              <AnalysisResult result={result} />
              <EvidencePanel
                trace={result.trace}
                evidence={result.evidence ?? null}
                selected={selectedEvidence}
                onSelect={setSelectedEvidence}
              />
            </div>
          )}

          {/* No result yet */}
          {!result && planned && (
            <div className="mt-6 p-4 border-t border-border pt-4 text-sm text-slate-400">
              <p className="eyebrow mb-2">Analysis</p>
              <p>
                {planned?.plan.executable
                  ? "Change analysis is executable."
                  : "Change analysis is not currently executable. " +
                      (planned.plan.unavailable_reason ?? "No provider registered.")}
              </p>
            </div>
          )}

          {/* CAPABILITY UNAVAILABLE notice */}
          {planned &&
            !planned.plan.executable &&
            planned.plan.unavailable_capabilities.length > 0 && (
            <div className="mt-6 p-4 rounded-lg border border-warning/30 bg-warning/5 text-xs text-warning">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-warning">
                CAPABILITY UNAVAILABLE / PROTOTYPE
              </p>
              <p className="mt-1 leading-relaxed text-warning">
                No Change-VQA provider is registered for this pair.{" "}
                <span role="text">
                  No answer can be produced — no invented change percentage, confidence,
                  bounding box, change mask, or metadata will be displayed.
                </span>
              </p>
            </div>
          )}

          {/* Status bar */}
          {(state === "LOADING" || state === "INVALID INPUT" || state === "UNAVAILABLE" || state === "EXECUTION ERROR") || busy && (
            <div
              role={state !== "IDLE" ? "alert" : "status"}
              className="rounded-lg border-border bg-raised/35 p-3 text-xs mt-3"
            >
              <strong className="font-mono tracking-[0.08em] text-slate-300">{state}</strong>
              <p className="mt-1.5 leading-relaxed text-slate-400">{message}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}