"use client";

import { useEffect, useState } from "react";
import { getHealth, type HealthStatus } from "@/lib/api";

export type RuntimeState = "checking" | "offline" | HealthStatus;

const LABELS: Record<RuntimeState, string> = {
  checking: "Checking runtime…",
  ready: "Operational",
  degraded: "Operational · some capabilities unavailable",
  unavailable: "Trace unavailable · analysis disabled",
  offline: "Not connected",
};

export function useRuntimeState(): RuntimeState {
  const [state, setState] = useState<RuntimeState>("checking");
  useEffect(() => { getHealth().then((health) => setState(health.status)).catch(() => setState("offline")); }, []);
  return state;
}

export function runtimeDotColour(state: RuntimeState) {
  return state === "ready" ? "bg-success shadow-[0_0_14px_#55d68a]"
    : state === "checking" || state === "degraded" ? "bg-warning"
    : "bg-error";
}

export function RuntimeStatus() {
  const state = useRuntimeState();
  return (
    <div className="panel flex items-center justify-between p-5">
      <div><p className="eyebrow">Local API</p><p className="mt-2 font-[500]">{LABELS[state]}</p></div>
      <span className={`size-3 rounded-full ${runtimeDotColour(state)}`} />
    </div>
  );
}
