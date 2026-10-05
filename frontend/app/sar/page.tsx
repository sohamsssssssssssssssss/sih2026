"use client";

import { useEffect, useState } from "react";
import { BadgeAlert, LoaderCircle } from "lucide-react";
import { getSar } from "@/lib/api";
import type { SarReport } from "@/lib/types";
import { SarViewer } from "@/components/sar/SarViewer";
import { SarInterpretation } from "@/components/sar/SarInterpretation";
import type { RegionKey } from "@/components/sar/regions";

export default function SarPage() {
  const [report, setReport] = useState<SarReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState<RegionKey | null>(null);
  useEffect(() => {
    getSar().then(setReport).catch((reason) => setError(reason.message));
  }, []);
  return (
    <div>
      <div className="mb-4">
        <p className="eyebrow">SAR validation</p>
        <h1 className="mt-2 text-[40px] font-[600] tracking-[-0.02em] text-ink sm:text-[56px]">
          Mumbai coastal interpretation
        </h1>
        <p className="muted mt-2">
          Human interpretation of a processed Sentinel-1 scene. For model output, pair an optical and a SAR GeoTIFF in the workspace (deterministic optical–SAR baseline).
        </p>
      </div>
      <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-warning/25 bg-warning/[0.06] px-3 py-1.5 text-[11px] font-[500] uppercase tracking-wider text-warning">
        <BadgeAlert size={14} /> Human SAR validation — not AI model output
      </div>
      {!report && !error && (
        <div className="panel grid h-64 place-items-center">
          <LoaderCircle className="animate-spin text-ash" />
        </div>
      )}
      {error && (
        <div role="alert" className="rounded-xl border border-error/30 bg-error/10 p-5 text-error">
          Analyst annotation unavailable: {error}
        </div>
      )}
      {report && (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(420px,0.9fr)]">
          <SarViewer
            scene={report.scene}
            available={report.render_available}
            active={active}
            onActive={setActive}
          />
          <SarInterpretation report={report} active={active} onActive={setActive} />
        </div>
      )}
    </div>
  );
}
