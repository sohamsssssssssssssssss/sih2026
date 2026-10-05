"use client";

import { useEffect, useState } from "react";
import { getHealth, type HealthStatus } from "@/lib/api";

type State = "checking" | "offline" | HealthStatus;

const LABELS: Record<State, string> = {
  checking: "Checking runtime…",
  ready: "Operational",
  degraded: "Operational · some capabilities unavailable",
  unavailable: "Trace unavailable · analysis disabled",
  offline: "Not connected",
};

export function RuntimeStatus() {
  const [state, setState] = useState<State>("checking");
  useEffect(() => { getHealth().then((health) => setState(health.status)).catch(() => setState("offline")); }, []);
  return (
    <div className="panel flex items-center justify-between p-5">
      <div><p className="eyebrow">Local API</p><p className="mt-2 font-[500]">{LABELS[state]}</p></div>
      <span className={cnDot(state)} />
    </div>
  );
}

function cnDot(state: State) {
  const colour =
    state === "ready" ? "bg-success shadow-[0_0_14px_#55d68a]"
    : state === "checking" || state === "degraded" ? "bg-warning"
    : "bg-error";
  return `size-3 rounded-full ${colour}`;
}
