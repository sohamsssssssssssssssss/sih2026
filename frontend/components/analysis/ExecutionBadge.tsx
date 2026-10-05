import { CircleHelp, DatabaseZap, Radio } from "lucide-react";
import { executionLabel, isExecutionMode } from "@/lib/workspace-contract";

export function ExecutionBadge({ mode }: { mode: unknown }) {
  const known = isExecutionMode(mode);
  const live = mode === "live";
  const tone = !known ? "border-error/30 bg-error/10 text-error" : live ? "border-success/30 bg-success/10 text-success" : "border-warning/30 bg-warning/10 text-warning";
  const Icon = !known ? CircleHelp : live ? Radio : DatabaseZap;
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-[600] tracking-[0.08em] ${tone}`}>
      <Icon size={13} />{executionLabel(mode)}
    </span>
  );
}
