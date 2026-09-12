"use client";

import { useState } from "react";
import { verifyTraces } from "@/lib/api";
import type { TraceRecord } from "@/lib/types";
import { formatTimestamp, shortHash } from "@/lib/utils";
import { IntegrityBadge } from "./IntegrityBadge";
import { TraceDrawer } from "./TraceDrawer";

export function EvidencePanel({ trace }: { trace: TraceRecord }) {
  const [verified, setVerified] = useState<boolean | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState(false);
  async function verify() {
    setChecking(true); setVerified(null); setError(false);
    try { setVerified((await verifyTraces()).verified); }
    catch { setError(true); }
    finally { setChecking(false); }
  }
  return (
    <section className="rounded-16 bg-surface-elevated border border-border p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="eyebrow">Execution evidence</p>
        <IntegrityBadge verified={verified} />
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {[
          ["Scene", trace.params.scene_id ?? "Unknown"],
          ["Model", trace.model_name], ["Model version", trace.model_version],
          ["Execution mode", trace.params.execution_mode],
          ["Question", trace.input_summary.question],
          ["Recorded at", formatTimestamp(trace.timestamp_iso)],
          ["Results artifact", trace.params.results_artifact ?? "Not recorded"],
        ].map(([label, value]) => <div key={label} className="min-w-0 text-xs"><p className="text-tertiary">{label}</p><p className="mt-1 break-words text-subtitle">{value}</p></div>)}
      </div>
      <div className="mt-4 space-y-2 text-xs text-subtitle">
        <p>Record hash: <span className="font-mono" title={trace.record_hash}>{shortHash(trace.record_hash)}</span></p>
        <p>Previous hash: <span className="font-mono" title={trace.prev_hash}>{trace.prev_hash ? shortHash(trace.prev_hash) : "None — first in process-local chain"}</span></p>
      </div>
      <button type="button" disabled={checking} onClick={verify} className="mt-4 rounded border border-border px-3 py-2 text-xs text-white focus-visible:outline focus-visible:outline-accent disabled:opacity-60">{checking ? "Checking integrity…" : "Check current process-local chain"}</button>
      <div role="status">{verified !== null && <p className={`mt-2 text-xs ${verified ? "text-success" : "text-error"}`}>{verified ? "Current process-local chain verified." : "Integrity verification failed."}</p>}</div>
      {error && <p role="alert" className="mt-2 text-xs text-error">Integrity status unavailable. The chain has not been verified.</p>}
      <p className="mt-3 text-xs leading-relaxed text-tertiary">Checks detect modifications within the current API process’s chain, which starts empty after restart. This check does not certify the answer or artifact contents, or establish that this record is still in that process. No external signature is provided.</p>
      <div className="mt-4"><TraceDrawer trace={trace} /></div>
    </section>
  );
}
