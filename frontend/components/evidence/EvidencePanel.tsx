import type { TraceRecord } from "@/lib/types";
import { formatTimestamp, shortHash } from "@/lib/utils";
import { IntegrityBadge } from "./IntegrityBadge";
import { TraceDrawer } from "./TraceDrawer";

export function EvidencePanel({ trace }: { trace: TraceRecord }) {
  const scenes = [trace.params.scene_id, trace.params.scene_id_2].filter(Boolean).join(" + ");
  return (
    <section className="panel p-5">
      <div className="mb-4 flex items-center justify-between"><p className="eyebrow">Execution evidence</p><IntegrityBadge /></div>
      <div className="grid gap-3 sm:grid-cols-2">
        <Evidence label={trace.params.scene_id_2 ? "Scenes" : "Identity"} value={scenes || "Not recorded"} mono />
        <Evidence label="Execution" value={`${trace.model_name} · ${trace.params.execution_mode}`} detail={trace.params.capability ? `Capability: ${trace.params.capability}` : undefined} />
        <Evidence label="Question" value={trace.input_summary.question} />
        <Evidence label="Recorded" value={formatTimestamp(trace.timestamp_iso)} />
        <Evidence label="Record hash" value={shortHash(trace.record_hash)} mono />
        <Evidence label="Previous hash" value={trace.prev_hash ? shortHash(trace.prev_hash) : "None · first record in chain"} mono />
      </div>
      <div className="mt-4"><TraceDrawer trace={trace} /></div>
    </section>
  );
}

function Evidence({ label, value, detail, mono = false }: { label: string; value: string; detail?: string; mono?: boolean }) {
  return <div className="rounded-lg border border-border bg-raised/45 p-3"><p className="text-[10px] font-[600] uppercase tracking-[0.14em] text-midgray">{label}</p><p className={`mt-2 break-words text-xs leading-5 text-ink ${mono ? "font-mono" : ""}`}>{value}</p>{detail && <p className="mt-1 text-[10px] text-midgray">{detail}</p>}</div>;
}
