import type { EvidenceItem } from "@/lib/types";
import { evidenceRows, formatValue, humanize } from "@/lib/evidence";

/** The model's structured supporting evidence, shown as returned (no re-interpretation). */
export function ResultEvidence({ evidence }: { evidence: EvidenceItem[] }) {
  return (
    <section className="panel p-5" aria-label="Supporting evidence">
      <div className="flex items-center justify-between"><p className="eyebrow">Supporting evidence</p><span className="text-xs text-midgray">{evidence.length} item{evidence.length === 1 ? "" : "s"}</span></div>
      {evidence.length === 0
        ? <p className="mt-3 text-sm text-midgray">The model returned no structured evidence for this answer.</p>
        : <ol className="mt-4 space-y-3">{evidence.map((item, index) => <EvidenceEntry key={index} index={index + 1} item={item} />)}</ol>}
    </section>
  );
}

function EvidenceEntry({ item, index }: { item: EvidenceItem; index: number }) {
  const type = typeof item.type === "string" ? item.type : "evidence";
  if (type === "bounding_box") {
    const confidence = typeof item.confidence === "number" ? item.confidence : null;
    return (
      <li className="rounded-lg border border-border bg-raised/45 p-3 text-xs">
        <p className="font-[600] text-ink">
          <span className="mr-2 inline-grid size-5 place-items-center rounded-full bg-cobalt text-[10px] text-white">{index}</span>
          {typeof item.label === "string" && item.label ? item.label : "Region"}
          {confidence !== null && <span className="ml-2 font-[500] text-deepgray">score {formatValue(confidence)}</span>}
        </p>
        <p className="mt-1 font-mono text-[10px] text-midgray">{formatValue(item.coordinate_space)} [{formatValue(item.coordinates)}]</p>
      </li>
    );
  }
  return (
    <li className="rounded-lg border border-border bg-raised/45 p-3 text-xs">
      <p className="font-[600] text-ink"><span className="mr-2 inline-grid size-5 place-items-center rounded-full bg-deepgray text-[10px] text-white">{index}</span>{humanize(type)}</p>
      <dl className="mt-2 grid gap-x-3 gap-y-1 sm:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        {evidenceRows(item).map(([label, value]) => <div key={label} className="contents"><dt className="text-midgray">{label}</dt><dd className="break-all font-mono text-[11px] text-deepgray">{value}</dd></div>)}
      </dl>
    </li>
  );
}
