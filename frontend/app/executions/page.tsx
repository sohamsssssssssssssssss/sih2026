"use client";

import { useCallback, useEffect, useState } from "react";
import { History, RefreshCw, ShieldCheck } from "lucide-react";
import { IntegrityBadge } from "@/components/evidence/IntegrityBadge";
import { getTraces, verifyTraces } from "@/lib/api";
import type { TraceRecord } from "@/lib/types";
import { ExecutionBadge } from "@/components/analysis/ExecutionBadge";
import { formatTimestamp, shortHash } from "@/lib/utils";

export default function ExecutionsPage() {
  const [records, setRecords] = useState<TraceRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [verification, setVerification] = useState<string | null>(null);
  const [verified, setVerified] = useState<boolean | null>(null);
  const load = useCallback(() => { getTraces().then((data) => { setRecords(data.records); setError(null); }).catch((reason) => setError(reason.message)); }, []);
  useEffect(() => { load(); }, [load]);
  async function verify() { setVerified(null); setVerification(null); try { const result = await verifyTraces(); setVerified(result.verified); setVerification(result.message); } catch { setVerification("Integrity status unavailable. The current chain has not been verified."); } }
  return (
    <div><div className="mb-6 flex flex-wrap items-end justify-between gap-4"><div><p className="eyebrow">Execution history</p><h1 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">Audit trail</h1><p className="muted mt-2">Process-local execution records, cryptographically linked in invocation order.</p></div><div className="flex gap-2"><button onClick={load} aria-label="Refresh executions" className="rounded-lg border border-border bg-surface p-2.5 text-slate-300 hover:text-accent"><RefreshCw size={17} /></button><button onClick={verify} className="flex items-center gap-2 rounded-lg border border-accent/30 bg-accent/10 px-4 py-2 text-sm font-bold text-accent"><ShieldCheck size={16} /> Verify chain</button></div></div>
      {verification && <div role="status" className={`mb-5 flex flex-wrap items-center gap-2 rounded-xl border p-4 text-sm ${verified === true ? "border-success/25 text-success" : verified === false ? "border-error/40 text-error" : "border-border text-subtitle"}`}><IntegrityBadge verified={verified} /> {verification}</div>}
      {error && <div role="alert" className="rounded-xl border border-error/30 bg-error/10 p-5 text-error">{error}</div>}
      {!error && records.length === 0 && <div className="panel grid min-h-72 place-items-center p-8 text-center"><div><History className="mx-auto text-slate-600" size={32} /><h2 className="mt-4 font-semibold text-slate-300">No executions in this API process yet</h2><p className="mt-2 text-sm text-slate-500">Run the golden analysis in Workspace, then refresh this view.</p></div></div>}
      <div className="space-y-3">{records.map((record, index) => <article key={record.record_hash} className="panel grid gap-4 p-5 md:grid-cols-[72px_1fr_auto] md:items-center"><div className="font-mono text-xs text-slate-500">#{records.length - index}</div><div><p className="text-sm font-semibold text-white">{record.input_summary.question}</p><p className="mt-2 font-mono text-[10px] text-slate-500">{record.params.scene_id} · {shortHash(record.record_hash)} · {formatTimestamp(record.timestamp_iso)}</p></div><ExecutionBadge mode={record.params.execution_mode} /></article>)}</div>
    </div>
  );
}
