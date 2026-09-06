"use client";

import { useEffect, useState } from "react";
import { LoaderCircle } from "lucide-react";
import { getResolution } from "@/lib/api";
import type { ResolutionReport } from "@/lib/types";
import { ResolutionChart } from "@/components/resolution/ResolutionChart";
import { ResolutionTable } from "@/components/resolution/ResolutionTable";
import { DegeneracyWarning } from "@/components/resolution/DegeneracyWarning";

export default function ResolutionPage() {
  const [report, setReport] = useState<ResolutionReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { getResolution().then(setReport).catch((reason) => setError(reason.message)); }, []);
  return (
    <div>
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4"><div><p className="eyebrow">Resolution lab</p><h1 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">Robustness under resolution loss</h1><p className="muted mt-2 max-w-2xl">A frozen 2,000-sample evaluation across five ground-sample-distance rungs.</p></div>{report && <div className="text-right"><p className="text-2xl font-bold text-white">{report.n_samples.toLocaleString()}</p><p className="text-xs uppercase tracking-wider text-slate-500">evaluated samples</p></div>}</div>
      {!report && !error && <div className="panel grid h-64 place-items-center"><LoaderCircle className="animate-spin text-accent" /></div>}
      {error && <div role="alert" className="rounded-xl border border-error/30 bg-error/10 p-5 text-error">Resolution artifact unavailable: {error}</div>}
      {report && <div className="space-y-5"><ResolutionChart report={report} /><DegeneracyWarning report={report} /><ResolutionTable report={report} /><p className="font-mono text-[10px] leading-relaxed text-slate-600">MODEL {report.model} · RUN {report.timestamp}</p></div>}
    </div>
  );
}
