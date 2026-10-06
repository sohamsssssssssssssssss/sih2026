import type { EvidenceItem } from "@/lib/types";
import { UNKNOWN, formatNumber, formatPercent, formatValue } from "@/lib/evidence";

const ATTRIBUTION = "Village boundaries: DataMeet (Census 2001), ODbL";
const HEADERS = ["Village", "Sub-district", "District", "Flooded (ha)", "% of village flooded", "% observed", "Extends past scene edge"];

const text = (value: unknown) => typeof value === "string" && value.trim() ? value : UNKNOWN;
const yesNo = (value: unknown) => typeof value === "boolean" ? (value ? "Yes" : "No") : UNKNOWN;

function villageCells(village: Record<string, unknown>): string[] {
  return [
    text(village.name), text(village.sub_district), text(village.district), formatNumber(village.flooded_ha),
    formatPercent(village.flooded_fraction), formatPercent(village.observed_fraction), yesNo(village.partially_outside_scene),
  ];
}

function VillageRows({ item }: { item: EvidenceItem }) {
  const villages = Array.isArray(item.flooded_villages) ? item.flooded_villages.filter(village => village && typeof village === "object") as Array<Record<string, unknown>> : [];
  return (
    <>
      <p className="mt-2 text-sm font-[600] text-ink">{formatNumber(item.flooded_village_count)} of {formatNumber(item.villages_in_scene)} villages in the scene gained open water</p>
      {item.truncated === true && <p className="mt-1 text-xs text-warning">List truncated: showing {villages.length} of {formatNumber(item.flooded_village_count)} villages.</p>}
      {villages.length > 0 && (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-left text-xs">
            <caption className="sr-only">Villages that gained open water, most flooded area first</caption>
            <thead>
              <tr className="border-b border-border text-midgray">
                {HEADERS.map((header, column) => <th key={header} scope="col" aria-sort={column === 3 ? "descending" : undefined} className={`px-2 py-1.5 font-[500] ${column >= 3 && column <= 5 ? "text-right" : ""}`}>{header}</th>)}
              </tr>
            </thead>
            <tbody>
              {villages.map((village, row) => (
                <tr key={row} className="border-b border-border/60">
                  {villageCells(village).map((cell, column) => <td key={column} className={`px-2 py-1.5 ${column === 0 ? "font-[500] text-ink" : "text-deepgray"} ${column >= 3 && column <= 5 ? "text-right font-mono" : ""}`}>{cell}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-[11px] leading-5 text-midgray">Sorted by flooded area, largest first. Percentages are of each village&apos;s area inside the scene; % observed is the share of that area with valid radar pixels.</p>
        </div>
      )}
    </>
  );
}

/** Villages that gained open water between the two dates, as returned by the village overlay. */
export function VillageTable({ item }: { item: EvidenceItem }) {
  return (
    <section className="panel p-5" aria-label="Flooded villages">
      <p className="eyebrow">Villages with new water</p>
      {item.status === "measured" ? <VillageRows item={item} />
        : item.status === "no_boundaries" ? <p className="mt-2 text-sm text-deepgray">No village boundaries cover this scene, so villages are not named.</p>
        : <p className="mt-2 text-sm text-deepgray">Village overlay status is {formatValue(item.status)}, so villages are not named.</p>}
      <p className="mt-3 border-t border-border pt-2 text-[11px] text-midgray">{ATTRIBUTION}</p>
    </section>
  );
}
