"use client";

import { useEffect, useState } from "react";
import { BadgeAlert, LoaderCircle } from "lucide-react";
import { getSar } from "@/lib/api";
import type { SarReport } from "@/lib/types";
import { SarViewer } from "@/components/sar/SarViewer";
import { SarInterpretation } from "@/components/sar/SarInterpretation";

export default function SarPage() {
  const [report, setReport] = useState<SarReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { getSar().then(setReport).catch((reason) => setError(reason.message)); }, []);
  return <div><div className="mb-6"><p className="eyebrow">SAR validation</p><h1 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">Mumbai coastal interpretation</h1><p className="muted mt-2">Human interpretation of a processed Sentinel-1 scene. Optical–SAR fusion remains in development.</p></div><div className="mb-5 flex items-center gap-3 rounded-xl border border-warning/30 bg-warning/[0.07] px-4 py-3 text-sm font-black tracking-wide text-warning"><BadgeAlert size={18} /> HUMAN SAR VALIDATION — NOT AI MODEL OUTPUT</div>{!report && !error && <div className="panel grid h-64 place-items-center"><LoaderCircle className="animate-spin text-accent" /></div>}{error && <div role="alert" className="rounded-xl border border-error/30 bg-error/10 p-5 text-error">Analyst annotation unavailable: {error}</div>}{report && <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(420px,0.9fr)]"><SarViewer scene={report.scene} available={report.render_available} /><SarInterpretation report={report} /></div>}</div>;
}
