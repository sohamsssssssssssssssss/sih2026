import type { AnalysisResponse } from "@/lib/types";
import { ExecutionBadge } from "./ExecutionBadge";

export function AnalysisResult({ result }: { result: AnalysisResponse }) {
  return (
    <section className="rounded-16 bg-surface-elevated border border-border p-6">
      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between gap-3">
          <p className="eyebrow">Answer</p>
          <ExecutionBadge mode={result.execution_mode} />
        </div>
        <p className="break-words text-3xl font-semibold tracking-tight text-white">{result.answer}</p>
        <div><p className="eyebrow">Model</p><p className="mt-1 break-words text-base text-subtitle">{result.model.version}</p></div>
        {result.results_artifact && <p className="mt-4 break-all font-mono text-[10px] text-tertiary">{result.results_artifact}</p>}
      </div>
    </section>
  );
}
