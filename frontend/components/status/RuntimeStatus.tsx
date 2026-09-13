"use client";

import { useEffect, useState } from "react";
import { getHealth, verifyTraces } from "@/lib/api";

type RuntimeState = {
  health: "CHECKING" | "READY" | "UNAVAILABLE" | "UNKNOWN";
  mode: string;
  integrity: "CHECKING" | "VERIFIED" | "UNVERIFIED" | "UNKNOWN";
  integrityMessage: string;
};

export function RuntimeStatus() {
  const [state, setState] = useState<RuntimeState>({ health: "CHECKING", mode: "NOT REPORTED", integrity: "CHECKING", integrityMessage: "Verification pending." });

  useEffect(() => {
    void Promise.allSettled([getHealth(), verifyTraces()]).then(([health, integrity]) => setState({
      health: health.status === "rejected" ? "UNKNOWN" : health.value.status === "ready" ? "READY" : "UNAVAILABLE",
      mode: health.status === "fulfilled" ? health.value.mode || "NOT REPORTED" : "UNKNOWN",
      integrity: integrity.status === "fulfilled" ? integrity.value.verified ? "VERIFIED" : "UNVERIFIED" : "UNKNOWN",
      integrityMessage: integrity.status === "fulfilled" ? integrity.value.message : "Trace integrity could not be checked.",
    }));
  }, []);

  return <section className="panel p-5" aria-label="Runtime status">
    <h2 className="eyebrow">Runtime</h2>
    <dl className="mt-4 divide-y divide-border">
      <RuntimeField label="API health" value={state.health} detail={`Mode: ${state.mode}`} />
      <RuntimeField label="Trace / integrity" value={state.integrity} detail={state.integrityMessage} />
    </dl>
  </section>;
}

function RuntimeField({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="flex items-start justify-between gap-4 py-3 first:pt-0 last:pb-0"><div><dt className="text-xs font-semibold text-slate-300">{label}</dt><dd className="mt-1 text-[11px] leading-5 text-slate-500">{detail}</dd></div><span className="text-xs font-bold text-accent">{value}</span></div>;
}
