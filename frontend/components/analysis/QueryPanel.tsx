"use client";

import { useState } from "react";
import { ArrowRight, LoaderCircle } from "lucide-react";
import { analyzeGolden } from "@/lib/api";
import type { AnalysisResponse } from "@/lib/types";
import { AnalysisResult } from "./AnalysisResult";
import { EvidencePanel } from "@/components/evidence/EvidencePanel";
import { UploadScene } from "@/components/imagery/UploadScene";

const GOLDEN_QUESTION = "Is there a building in this image?";

export function QueryPanel() {
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function runAnalysis() {
    setLoading(true); setError(null); setResult(null);
    try { setResult(await analyzeGolden(GOLDEN_QUESTION)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Analysis failed safely. No answer was generated."); }
    finally { setLoading(false); }
  }

  return (
    <div className="space-y-4">
      <section className="panel p-5 sm:p-6">
        <div className="flex items-start justify-between gap-4"><div><p className="eyebrow">Ask SatQuery</p><h2 className="mt-2 text-2xl font-bold tracking-tight text-white">Interrogate this scene</h2></div><span className="rounded-full border border-success/25 bg-success/10 px-2.5 py-1 text-[10px] font-bold text-success">GOLDEN SCENE</span></div>
        <label htmlFor="question" className="mt-6 block text-xs font-semibold uppercase tracking-wider text-slate-400">Question</label>
        <textarea id="question" value={GOLDEN_QUESTION} readOnly rows={3} className="mt-2 w-full resize-none rounded-lg border border-border bg-background px-4 py-3 text-sm leading-relaxed text-slate-100 outline-none focus:border-accent" />
        <button onClick={runAnalysis} disabled={loading} className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-4 py-3 text-sm font-black tracking-wide text-[#031013] transition hover:bg-[#79edf1] focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-background disabled:opacity-60">
          {loading ? <><LoaderCircle className="animate-spin" size={17} /> ANALYZING</> : <>ASK SATQUERY <ArrowRight size={17} /></>}
        </button>
        <div className="mt-4"><UploadScene /></div>
      </section>
      {error && <div role="alert" className="rounded-xl border border-error/30 bg-error/10 p-4 text-sm leading-relaxed text-error"><strong>No fabricated answer.</strong> {error}</div>}
      {result && <><AnalysisResult result={result} /><EvidencePanel trace={result.trace} /></>}
    </div>
  );
}
