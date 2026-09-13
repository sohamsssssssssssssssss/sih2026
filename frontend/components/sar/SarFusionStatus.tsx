import { useEffect, useState } from "react";
import { getCapabilities } from "@/lib/api";
import type { CapabilityStatus } from "@/lib/types";

const ROWS = [
  {
    state: "REAL" as const,
    title: "Sentinel-1 acquisitions & RTC processing",
    detail: "Sentinel-1 IW GRD scenes ordered via ASF + HyP3 and rendered by the committed pipeline in data/sar_gate/process_scenes.py. Renders must be regenerated in this checkout to display pixels.",
  },
  {
    state: "CACHED REAL" as const,
    title: "Human analyst interpretation",
    detail: "Written by a human from the rendered scenes, committed verbatim in data/sar_gate/annotation_template.md — including confidence levels and uncertainty notes.",
  },
  {
    state: "PROTOTYPE" as const,
    title: "Optical–SAR fusion",
    detail: "The optical_sar capability has no registered provider and cannot execute. No fused product exists anywhere in this repository, and none is simulated here.",
  },
];

/** Tracks the live /api/capabilities snapshot so the prototype claim is
 *  verified against the backend rather than asserted statically. */
export function SarFusionStatus() {
  const [opticalSar, setOpticalSar] = useState<CapabilityStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    getCapabilities()
      .then((data) => {
        if (!cancelled) setOpticalSar(data.capabilities.find((c) => c.name === "optical_sar") ?? null);
      })
      .catch(() => {
        /* Status panel stays honest with what it knows; availability is not fabricated. */
      });
    return () => { cancelled = true; };
  }, []);

  const providerAvailable = opticalSar?.available === true;

  return (
    <section className="panel space-y-4 p-5" aria-label="Real vs prototype status">
      <p className="eyebrow">What is real here</p>
      <div className="space-y-3">
        {ROWS.map((row) => (
          <article key={row.title} className="rounded-lg border border-border bg-raised/45 p-4">
            <div className="flex flex-wrap items-center gap-3">
              <StateBadge state={row.state} />
              <h2 className="text-sm font-bold text-white">{row.title}</h2>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">{row.detail}</p>
          </article>
        ))}
      </div>
      <p role="status" data-testid="optical-sar-status" className="text-xs text-slate-400">
        {opticalSar === null
          ? "optical_sar capability status: could not be verified (backend status unavailable)."
          : providerAvailable
            ? `optical_sar capability: registered to provider "${opticalSar.provider}". This page still displays human interpretation only.`
            : "Verified against /api/capabilities: optical_sar has no registered provider and cannot execute in this build."}
      </p>
    </section>
  );
}

function StateBadge({ state }: { state: "REAL" | "CACHED REAL" | "PROTOTYPE" }) {
  const styles = {
    REAL: "border-success/30 bg-success/10 text-success",
    "CACHED REAL": "border-success/30 bg-success/10 text-success",
    PROTOTYPE: "border-warning/30 bg-warning/10 text-warning",
  } as const;
  return <span className={`rounded-full border px-2.5 py-1 text-[10px] font-black tracking-[0.08em] ${styles[state]}`}>{state}</span>;
}
