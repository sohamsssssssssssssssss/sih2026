"use client";

import { useEffect, useState } from "react";
import { BadgeAlert, LoaderCircle } from "lucide-react";
import { errorMessage, getSar } from "@/lib/api";
import type { SarReport } from "@/lib/types";
import { SarViewer } from "@/components/sar/SarViewer";
import { SarInterpretation } from "@/components/sar/SarInterpretation";
import { SarProvenance } from "@/components/sar/SarProvenance";
import { SarFusionStatus } from "@/components/sar/SarFusionStatus";

/** The five scenes with committed human annotations in
 *  data/sar_gate/annotation_template.md. Selecting one fetches that scene's
 *  annotation from the backend; unknown scenes fail closed server-side. */
const SAR_SCENES = [
  { id: "mumbai-coastal", label: "Mumbai coastal" },
  { id: "maharashtra-farmland", label: "Maharashtra farmland" },
  { id: "western-ghats-forest", label: "Western Ghats forest" },
  { id: "konkan-coast", label: "Konkan coast" },
  { id: "flat-inland-plain", label: "Flat inland plain" },
];

export default function SarPage() {
  const [scene, setScene] = useState("mumbai-coastal");
  const [report, setReport] = useState<SarReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getSar(scene)
      .then((data) => { if (!cancelled) setReport(data); })
      .catch((reason) => { if (!cancelled) { setReport(null); setError(errorMessage(reason)); } })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [scene]);

  const ready = report !== null && report.human_validation;

  return (
    <div>
      <div className="mb-6">
        <p className="eyebrow">Optical/SAR prototype</p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">Sentinel-1 SAR, human-validated</h1>
        <p className="muted mt-2 max-w-3xl">Real Sentinel-1 acquisitions, processed to RTC false color, interpreted by a human analyst. Optical–SAR fusion is a prototype: the optical_sar capability has no registered provider and produces no output in this build.</p>
      </div>

      <div className="mb-5 flex items-center gap-3 rounded-xl border border-warning/30 bg-warning/[0.07] px-4 py-3 text-sm font-black tracking-wide text-warning">
        <BadgeAlert size={18} /> HUMAN SAR VALIDATION — NOT AI MODEL OUTPUT
      </div>

      <div className="mb-5 flex flex-wrap gap-2" aria-label="Annotated SAR scenes">
        {SAR_SCENES.map(({ id, label }) => (
          <button key={id} onClick={() => setScene(id)} disabled={loading} aria-pressed={scene === id}
            className={`rounded-lg border px-3 py-2 text-sm font-semibold disabled:opacity-50 ${scene === id ? "border-accent/50 bg-accent/10 text-accent" : "border-border text-slate-300 hover:bg-raised"}`}>
            {label}
          </button>
        ))}
      </div>

      {loading && <div className="panel grid h-64 place-items-center"><LoaderCircle className="animate-spin text-accent" /></div>}

      {!loading && error && (
        <div role="alert" className="rounded-xl border border-error/30 bg-error/10 p-5 text-error">
          <p className="font-semibold">SAR scene data unavailable. This view fails closed — no substitute annotation or imagery is shown.</p>
          <p className="mt-2 text-sm">{error}</p>
        </div>
      )}

      {!loading && report && !report.human_validation && (
        <p role="alert" className="text-warning">Human validation unavailable. Interpretation is not displayed.</p>
      )}

      {!loading && ready && (
        <>
          <div className="mb-5">
            <h2 className="text-xl font-bold text-white">{report.title}</h2>
            <p className="text-xs text-slate-400">scene key: {report.scene}</p>
          </div>
          <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(420px,0.9fr)]">
            <SarViewer key={report.scene} scene={report.scene} available={report.render_available} />
            <SarInterpretation report={report} />
          </div>
          <div className="mt-5 space-y-5">
            <SarProvenance report={report} />
            <SarFusionStatus />
          </div>
        </>
      )}
    </div>
  );
}
