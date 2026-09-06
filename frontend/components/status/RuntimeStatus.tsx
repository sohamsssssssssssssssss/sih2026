"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";

export function RuntimeStatus() {
  const [state, setState] = useState<"checking" | "ready" | "offline">("checking");
  useEffect(() => { getHealth().then(() => setState("ready")).catch(() => setState("offline")); }, []);
  const ready = state === "ready";
  return (
    <div className="panel flex items-center justify-between p-5">
      <div><p className="eyebrow">Local API</p><p className="mt-2 font-semibold">{state === "checking" ? "Checking runtime…" : ready ? "Operational" : "Not connected"}</p></div>
      <span className={cnDot(ready, state)} />
    </div>
  );
}

function cnDot(ready: boolean, state: string) {
  return `size-3 rounded-full ${state === "checking" ? "bg-warning" : ready ? "bg-success shadow-[0_0_14px_#55d68a]" : "bg-error"}`;
}
