import type { SarReport } from "@/lib/types";

/** Provenance facts exactly as recorded in committed artifacts. Fields the
 *  repository does not record are shown as UNKNOWN — never inferred, never
 *  filled from maps or external lookups. */
export function SarProvenance({ report }: { report: SarReport }) {
  const hasCoordinates = report.latitude !== null && report.longitude !== null;
  return (
    <section className="panel space-y-4 p-5" aria-label="SAR scene provenance">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="eyebrow">Scene provenance</p>
        <span className="rounded-full border border-success/30 bg-success/10 px-3 py-1 text-xs font-black tracking-[0.08em] text-success">REAL DATA · SENTINEL-1 RTC</span>
      </div>
      <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Provenance label="Sensor" value={report.sensor} />
        <Provenance label="Data source" value="Sentinel-1 IW GRD → ASF HyP3 RTC (gamma-0)" />
        <Provenance label="Recorded location" value={report.location} unknown={report.location === "UNKNOWN"} />
        <Provenance label="Acquisition date" value={report.acquisition_date ?? "UNKNOWN"} unknown={report.acquisition_date === null} />
        <Provenance
          label="RTC processing job"
          value={report.processing_job_id ?? "Not recorded in this repository"}
          mono
          unknown={report.processing_job_id === null}
        />
        <Provenance
          label="Render in this checkout"
          value={report.render_available ? "Committed render available" : "Not committed — viewer fails closed"}
          unknown={!report.render_available}
        />
      </dl>
      {hasCoordinates && (
        <p className="text-xs text-slate-400">
          Coordinates {report.latitude}, {report.longitude} are recorded in the analyst annotation (data/sar_gate/annotation_template.md).
        </p>
      )}
      <div>
        <p className="text-xs font-bold uppercase tracking-wider text-slate-500">Processing chain (as implemented)</p>
        <p className="mt-1 break-words font-mono text-[11px] leading-relaxed text-slate-300">{report.processing_chain}</p>
      </div>
    </section>
  );
}

function Provenance({ label, value, mono = false, unknown = false }: { label: string; value: string; mono?: boolean; unknown?: boolean }) {
  return (
    <div className="rounded-lg border border-border bg-raised/45 p-3">
      <dt className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-500">{label}</dt>
      <dd className={`mt-2 break-words text-xs leading-5 ${unknown ? "text-slate-500" : "text-slate-200"} ${mono ? "font-mono" : ""}`}>{value}</dd>
    </div>
  );
}
