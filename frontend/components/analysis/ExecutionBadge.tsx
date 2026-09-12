import { DatabaseZap, Radio, TriangleAlert } from "lucide-react";
import { executionLabel, isExecutionMode } from "@/lib/workspace-contract";

export function ExecutionBadge({ mode }: { mode: unknown }) {
  if (!isExecutionMode(mode)) return <span className="inline-flex items-center gap-2 text-xs font-medium text-error"><TriangleAlert size={14} />{executionLabel(mode)}</span>;
  const live = mode === "live";
  const base = "inline-flex items-center gap-2 rounded-24 border px-3 py-1.5 text-sm font-medium tracking-[0.04em]";
  const liveClass = "border-primary/15 bg-primary/10 text-primary";
  const defaultClass = "border-border bg-surface-elevated text-secondary";
  const cls = live ? liveClass : defaultClass;
  return (
    <span className={`${base} ${cls}`}>
      {live ? <Radio size={13} /> : <DatabaseZap size={13} />}
      {executionLabel(mode)}
    </span>
  );
}
