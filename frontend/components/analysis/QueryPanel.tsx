"use client";

import { useId, useState } from "react";
import { ArrowRight, DatabaseZap, LoaderCircle } from "lucide-react";
import { AnalysisError, analyze, type AnalyzeRequest } from "@/lib/api";
import type { AnalysisResponse, ExecutionMode } from "@/lib/types";
import { GOLDEN_OPTION, PAIR_QUESTIONS, type SceneOption } from "@/lib/scenes";
import { isGoldenRequest, type FailureDetail } from "@/lib/workspace-contract";
import { AnalysisResult } from "./AnalysisResult";
import { AnalysisErrorNotice } from "./AnalysisErrorNotice";
import { EvidencePanel } from "@/components/evidence/EvidencePanel";
import { ResultEvidence } from "@/components/evidence/ResultEvidence";

const MAX_QUESTION_LENGTH = 2000;
const KIND_BADGE: Record<SceneOption["kind"], string> = { golden: "GOLDEN SCENE", curated: "CURATED SCENE", upload: "UPLOADED SCENE" };

type Attempt = Omit<AnalyzeRequest, "signal">;
interface Failure { status: number; detail: FailureDetail; attempt: Attempt }

export interface QueryPanelProps {
  scene?: SceneOption;
  /** Candidates for the optional second scene (pair questions). */
  scenes?: SceneOption[];
  onResult?: (result: AnalysisResponse | null) => void;
  /** Called with the second scene id whenever it changes (for previews). */
  onSecondScene?: (sceneId: string | null) => void;
}

export function QueryPanel({ scene = GOLDEN_OPTION, scenes = [], onResult, onSecondScene }: QueryPanelProps) {
  const ids = useId();
  const [question, setQuestion] = useState(scene.suggestedQuestion ?? "");
  const [sceneId2, setSceneId2] = useState("");
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [loading, setLoading] = useState<ExecutionMode | null>(null);
  const others = scenes.filter(option => option.id !== scene.id && option.available);

  function show(next: AnalysisResponse | null) { setResult(next); onResult?.(next); }

  async function run(attempt: Attempt) {
    setLoading(attempt.executionMode); setFailure(null); show(null);
    try { show(await analyze(attempt)); }
    catch (reason) {
      const detail: FailureDetail = reason instanceof AnalysisError ? reason.detail : { kind: "message", message: reason instanceof Error ? reason.message : "Analysis failed safely. No answer was generated." };
      setFailure({ status: reason instanceof AnalysisError ? reason.status : 0, detail, attempt });
    }
    finally { setLoading(null); }
  }

  function ask() {
    if (!question.trim() || loading) return;
    // Always live first. Cached replay is only offered afterwards, never substituted.
    void run({ sceneId: scene.id, question, sceneId2: sceneId2 || null, executionMode: "live", sensor: scene.requestSensor });
  }

  // Offered only when a *live* request for the pinned golden scene + question returned 503.
  const replayable = failure !== null && failure.status === 503 && failure.attempt.executionMode === "live"
    && isGoldenRequest(failure.attempt.sceneId, failure.attempt.question, failure.attempt.sceneId2);

  return (
    <div className="space-y-4">
      <section className="panel p-5 sm:p-6">
        <div className="flex items-start justify-between gap-4">
          <div><p className="eyebrow">Ask SatQuery</p><h2 className="mt-2 text-2xl font-[600] tracking-tight text-ivory">Interrogate this scene</h2></div>
          <span className={`whitespace-nowrap rounded-full border px-2.5 py-1 text-[10px] font-[600] ${scene.kind === "golden" ? "border-success/25 bg-success/10 text-success" : "border-border bg-surface text-deepgray"}`}>{KIND_BADGE[scene.kind]}</span>
        </div>
        <label htmlFor={`${ids}-question`} className="mt-6 block text-xs font-[500] uppercase tracking-wider text-midgray">Question</label>
        <textarea
          id={`${ids}-question`}
          value={question}
          onChange={event => setQuestion(event.target.value)}
          onKeyDown={event => { if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) { event.preventDefault(); ask(); } }}
          maxLength={MAX_QUESTION_LENGTH}
          rows={3}
          placeholder="Ask about this scene, e.g. Is there a building in this image?"
          className="mt-2 w-full resize-y rounded-lg border border-border bg-background px-4 py-3 text-sm leading-relaxed text-ink outline-none focus:border-accent"
        />
        {scene.suggestedQuestion && question.trim() !== scene.suggestedQuestion && (
          <button type="button" onClick={() => setQuestion(scene.suggestedQuestion ?? "")} className="mt-1 text-left text-xs text-linkblue hover:underline">Use curated question: {scene.suggestedQuestion}</button>
        )}

        <label htmlFor={`${ids}-scene2`} className="mt-5 block text-xs font-[500] uppercase tracking-wider text-midgray">Second scene <span className="normal-case tracking-normal text-midgray">(optional · pair questions)</span></label>
        <select
          id={`${ids}-scene2`}
          value={sceneId2}
          onChange={event => { setSceneId2(event.target.value); onSecondScene?.(event.target.value || null); }}
          className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2.5 text-sm text-ink outline-none focus:border-accent"
        >
          <option value="">None · single-scene question</option>
          {others.map(option => <option key={option.id} value={option.id}>{option.label}{option.kind === "upload" ? ` · ${option.format ?? "upload"}` : ""}</option>)}
        </select>
        {sceneId2 && (
          <div className="mt-2 space-y-1 text-[11px] leading-5 text-midgray">
            <p>Change detection needs two uploaded TIFF scenes with acquisition times (this scene is T1); optical–SAR needs one optical and one SAR TIFF.</p>
            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={() => setQuestion(PAIR_QUESTIONS.change)} className="rounded-full border border-border px-2.5 py-1 text-deepgray hover:bg-surface">{PAIR_QUESTIONS.change}</button>
              <button type="button" onClick={() => setQuestion(PAIR_QUESTIONS.opticalSar)} className="rounded-full border border-border px-2.5 py-1 text-deepgray hover:bg-surface">{PAIR_QUESTIONS.opticalSar}</button>
            </div>
          </div>
        )}

        <button onClick={ask} disabled={loading !== null || !question.trim() || !scene.available} className="mt-5 flex w-full items-center justify-center gap-2 rounded-pill bg-cobalt px-4 py-3.5 text-sm font-[500] tracking-wide text-white transition hover:bg-[#0077ed] focus:outline-none focus:ring-2 focus:ring-cobalt/60 focus:ring-offset-2 focus:ring-offset-background disabled:opacity-60">
          {loading ? <><LoaderCircle className="animate-spin" size={17} /> {loading === "live" ? "ANALYZING" : "LOADING CACHED REPLAY"}</> : <>ASK SATQUERY <ArrowRight size={17} /></>}
        </button>
        {!scene.available && <p className="mt-2 text-xs text-warning">This scene is unavailable{scene.unavailableReason ? `: ${scene.unavailableReason}` : "."}</p>}
        <p className="mt-2 text-[11px] text-midgray">Runs live inference. Answers are never substituted from cached results.</p>
      </section>

      {failure && (
        <AnalysisErrorNotice detail={failure.detail} status={failure.status || undefined}>
          {replayable && (
            <div className="mt-3 border-t border-error/20 pt-3 text-deepgray">
              <p className="text-xs leading-5">Live inference is unavailable here. This pinned scene and question have a committed, measured result you can replay instead. It will be labelled as cached, not live.</p>
              <button
                type="button"
                onClick={() => void run({ ...failure.attempt, executionMode: "cached_result" })}
                disabled={loading !== null}
                className="mt-2 inline-flex items-center gap-2 rounded-pill border border-warning/40 bg-background px-3.5 py-2 text-xs font-[600] text-warning transition hover:bg-warning/10 disabled:opacity-60"
              >
                {loading === "cached_result" ? <LoaderCircle className="animate-spin" size={14} /> : <DatabaseZap size={14} />} Show committed result (cached replay)
              </button>
            </div>
          )}
        </AnalysisErrorNotice>
      )}
      {result && (
        <>
          <AnalysisResult result={result} />
          {Array.isArray(result.evidence) && <ResultEvidence evidence={result.evidence} />}
          <EvidencePanel trace={result.trace} />
        </>
      )}
    </div>
  );
}
