import { DatabaseZap, Radio } from "lucide-react";
import type { ExecutionMode } from "@/lib/types";

export function ExecutionBadge({ mode }: { mode: ExecutionMode }) {
  const live = mode === "live";
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-black tracking-[0.08em] ${live ? "border-success/30 bg-success/10 text-success" : "border-warning/30 bg-warning/10 text-warning"}`}>
      {live ? <Radio size={13} /> : <DatabaseZap size={13} />}{live ? "LIVE INFERENCE" : "VERIFIED CACHED RESULT"}
    </span>
  );
}
