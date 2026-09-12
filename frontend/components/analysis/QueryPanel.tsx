"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowRight, LoaderCircle } from "lucide-react";
import { analyzeGolden } from "@/lib/api";
import type { AnalysisResponse } from "@/lib/types";
import { AnalysisResult } from "./AnalysisResult";
import { EvidencePanel } from "@/components/evidence/EvidencePanel";
import { UploadScene } from "@/components/imagery/UploadScene";
import { GOLDEN_SCENE } from "@/lib/workspace-contract";

const GOLDEN_QUESTION = GOLDEN_SCENE.question;

export function QueryPanel() {
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [question, setQuestion] = useState<string>(GOLDEN_QUESTION);
  const pending = useRef<AbortController | null>(null);
  useEffect(() => () => { pending.current?.abort(); }, []);

  async function runAnalysis() {
    if (pending.current || !question.trim()) return;
    const controller = new AbortController();
    pending.current = controller;
    const timeout = setTimeout(() => controller.abort(), 60000);
    setLoading(true); setError(null); setResult(null);
    try { setResult(await analyzeGolden(question.trim(), controller.signal)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Analysis failed safely. No answer was generated."); }
    finally { clearTimeout(timeout); pending.current = null; setLoading(false); }
  }

  return (
    <div className="space-y-4">
    <form onSubmit={event => { event.preventDefault(); void runAnalysis(); }} className="surface-elevated rounded-16 border border-border p-5" aria-busy={loading}>
      <div className="flex items-start justify-between gap-3">
        <div>
<h2 className="text-sm font-medium uppercase tracking-[0.1em] text-subtitle">Ask SatQuery</h2>
<p className="mt-1 text-lg font-medium text-primary">Interrogate this scene</p>
        </div>
        <span className="rounded-full border border-primary/15 bg-primary/10 px-2.5 py-1 text-[10px] font-bold text-primary">VERIFIED DEMO SCENE</span>
      </div>
      <label htmlFor="question" className="mt-4 block text-xs font-medium uppercase tracking-wider text-subtitle">Question</label>
      <textarea
        id="question"
        value={question}
        onChange={event => { setQuestion(event.target.value); setResult(null); setError(null); }}
        disabled={loading}
        required
        rows={3}
        className="mt-2 w-full resize-none rounded-16 border border-border bg-surface px-4 py-3 text-base leading-relaxed text-primary outline-none focus:border-primary"
      />
      <button type="submit" disabled={loading || !question.trim()} className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-3 text-base font-medium text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-60">
        {loading ? <><LoaderCircle className="animate-spin" size={17} /> Running analysis…</> : <>RUN ANALYSIS <ArrowRight size={17} /></>}
      </button>
      <button type="button" disabled={loading} onClick={() => { setQuestion(GOLDEN_QUESTION); setResult(null); setError(null); }} className="mt-3 text-sm text-subtitle underline underline-offset-4">Use verified demo question</button>
      <p className="mt-2 text-xs leading-relaxed text-tertiary">Other questions require live inference or an exact cached match.</p>
      <div className="mt-4"><UploadScene /></div>
    </form>
    {loading && <p role="status" className="text-sm text-subtitle">Running analysis…</p>}
    {error && <div role="alert" className="rounded-lg border border-error/40 p-4 text-sm text-error">{error}</div>}
    {!loading && !error && !result && <p className="text-sm text-tertiary">No analysis yet. Submit a question about the selected scene.</p>}
    {result && <><div role="status"><AnalysisResult result={result} /></div><EvidencePanel key={result.trace.record_hash} trace={result.trace} /></>}
    </div>
  );
}
