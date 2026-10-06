import type { EvidenceItem } from "@/lib/types";
import { UNKNOWN, formatHectares, formatNumber, formatValue } from "@/lib/evidence";

// Plain-language meaning of the provider's abstention codes (docs/change-baseline.md).
const REASONS: Record<string, string> = {
  NO_OPEN_WATER_MODE: "No tile in the scene showed a clear split between open water and land, so no water threshold could be set.",
};

function sensitivityRange(value: unknown): string {
  const range = value && typeof value === "object" ? value as Record<string, unknown> : {};
  const low = formatNumber(range.threshold_minus_1db);
  const high = formatNumber(range.threshold_plus_1db);
  return low === UNKNOWN || high === UNKNOWN ? UNKNOWN : `${low}–${high} ha`;
}

/** Sentinel-1 water change in hectares, or an explicit abstention with no numbers. */
export function WaterChangeSummary({ item }: { item: EvidenceItem }) {
  if (item.status !== "measured") {
    const reason = typeof item.reason_code === "string" ? item.reason_code : null;
    return (
      <section className="panel p-5" aria-label="Water change">
        <p className="eyebrow">Water change · Sentinel-1</p>
        <p role="status" className="mt-2 text-sm font-[600] text-warning">
          {item.status === "abstained" ? "Abstained: no water areas are reported." : `Not measured (status ${formatValue(item.status)}): no water areas are reported.`}
        </p>
        <p className="mt-2 text-xs text-deepgray">Reason code: <code className="font-mono">{reason ?? UNKNOWN}</code></p>
        {reason && REASONS[reason] && <p className="mt-1 text-xs text-midgray">{REASONS[reason]}</p>}
      </section>
    );
  }
  const threshold = formatNumber(item.threshold_db);
  const rows: Array<[string, string]> = [
    ["New water", formatHectares(item.new_water_ha)],
    ["New water with threshold ±1 dB", sensitivityRange(item.new_water_ha_sensitivity)],
    ["Receded water", formatHectares(item.receded_water_ha)],
    ["VV water threshold", threshold === UNKNOWN ? UNKNOWN : `${threshold} dB`],
    ["Bimodal tiles used", formatNumber(item.bimodal_tiles)],
  ];
  return (
    <section className="panel p-5" aria-label="Water change">
      <p className="eyebrow">Water change · Sentinel-1</p>
      <dl className="mt-3 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1.5 text-sm">
        {rows.map(([label, value]) => <div key={label} className="contents"><dt className="text-midgray">{label}</dt><dd className="text-right font-mono text-ink">{value}</dd></div>)}
      </dl>
      <p className="mt-3 text-[11px] leading-5 text-midgray">Change between the two dates, not total flood extent. One VV threshold (split-based Otsu) defines water at both dates.</p>
    </section>
  );
}
