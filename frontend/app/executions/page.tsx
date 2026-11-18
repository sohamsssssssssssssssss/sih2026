"use client";

import { useCallback, useEffect, useState } from "react";
import { getAskTrace } from "@/lib/api";

export default function ExecutionsPage() {
  const [trace, setTrace] = useState<Awaited<ReturnType<typeof getAskTrace>>>();
  const [error, setError] = useState("");
  const load = useCallback(async () => {
    try {
      setTrace(await getAskTrace());
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The trace is unavailable.");
    }
  }, []);
  useEffect(() => { void load(); }, [load]);
  const executions = trace?.events.filter((event) =>
    typeof event.trace_id === "string" && event.phase === "answer" && event.rule === "status").reverse() ?? [];
  return (
    <div>
      <h1 className="text-[42px] sm:text-[54px]">Audit trail</h1>
      <div className="mt-4 flex items-center gap-4">
        {trace && <p role="status" className={trace.verified ? "text-green-800" : "text-error"}>{trace.verified ? "Chain verified ✓" : "✗ chain broken"}</p>}
        <button onClick={load} className="rounded-full border border-border bg-white px-4 py-2 text-sm">Refresh and verify</button>
      </div>
      {error && <p role="alert" className="mt-4 text-error">{error}</p>}
      {trace?.verified && executions.length === 0 && <p className="mt-4">No Ask executions yet.</p>}
      {executions.length > 0 && (
        <ul className="mt-5 space-y-2">
          {executions.map((event) => (
            <li key={String(event.trace_id)} className="panel p-4">
              <p className="text-sm capitalize">{String(event.status).replaceAll("_", " ")}</p>
              <p className="mt-1 break-all font-mono text-xs text-deepgray">{String(event.trace_id)}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
