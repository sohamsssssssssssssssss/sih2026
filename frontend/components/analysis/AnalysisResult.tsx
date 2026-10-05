import type { AnalysisResponse } from "@/lib/types";
import { ExecutionBadge } from "./ExecutionBadge";

export function AnalysisResult({ result }: { result: AnalysisResponse }) {
  const long = result.answer.length > 40;
  return (
    <section className="rounded-cards bg-surface p-6" aria-live="polite">
      <div className="flex flex-wrap items-center justify-between gap-3"><p className="eyebrow">Answer</p><ExecutionBadge mode={result.execution_mode} /></div>
      <p className={`mt-5 font-[600] text-ivory ${long ? "text-xl leading-snug tracking-tight" : "text-5xl tracking-[-0.06em] sm:text-6xl"}`}>{result.answer}</p>
      <p className="mt-4 text-sm text-deepgray">{result.model.name} · <span className="font-mono text-xs">{result.model.version}</span></p>
      {result.notice && <p className="mt-2 text-xs leading-5 text-midgray">{result.notice}</p>}
      {result.results_artifact && <p className="mt-4 break-all border-t border-border pt-3 font-mono text-[10px] text-midgray">{result.results_artifact}</p>}
    </section>
  );
}
