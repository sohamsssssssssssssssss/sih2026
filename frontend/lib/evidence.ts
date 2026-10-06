import type { EvidenceItem } from "./types";

/** An evidence region in normalized image coordinates (0..1, origin top-left). */
export interface OverlayBox {
  index: number;
  kind: string;
  label: string;
  confidence: number | null;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

const finiteUnit = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;

/** Boxes the API reports in `normalized_xyxy`; anything malformed is skipped, never guessed. */
export function overlayBoxes(evidence: EvidenceItem[] | undefined | null): OverlayBox[] {
  if (!Array.isArray(evidence)) return [];
  const boxes: OverlayBox[] = [];
  evidence.forEach((item, position) => {
    const coordinates = item.coordinates;
    if (item.coordinate_space !== "normalized_xyxy" || !Array.isArray(coordinates) || coordinates.length !== 4 || !coordinates.every(finiteUnit)) return;
    const [x0, y0, x1, y1] = coordinates as number[];
    if (x1 < x0 || y1 < y0) return;
    const kind = typeof item.type === "string" ? item.type : "region";
    const label = typeof item.label === "string" && item.label.trim() ? item.label : humanize(kind);
    const confidence = typeof item.confidence === "number" && Number.isFinite(item.confidence) ? item.confidence : null;
    boxes.push({ index: position + 1, kind, label, confidence, x0, y0, x1, y1 });
  });
  return boxes;
}

export function humanize(value: string) {
  const spaced = value.replace(/[_-]+/g, " ").trim();
  return spaced ? spaced[0].toUpperCase() + spaced.slice(1) : value;
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "Not recorded";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(4)));
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "string") return value;
  if (Array.isArray(value) && value.every(entry => entry === null || ["string", "number", "boolean"].includes(typeof entry))) {
    return value.map(formatValue).join(", ");
  }
  return JSON.stringify(value);
}

export const UNKNOWN = "unknown";
const NUMBER = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const PERCENT = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });

export const finiteNumber = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) ? value : null;

/** Grouped, one-decimal number; anything else is shown as unknown, never guessed. */
export function formatNumber(value: unknown): string {
  const number = finiteNumber(value);
  return number === null ? UNKNOWN : NUMBER.format(number);
}

export function formatHectares(value: unknown): string {
  const number = formatNumber(value);
  return number === UNKNOWN ? UNKNOWN : `${number} ha`;
}

/** A 0..1 fraction as a percentage. */
export function formatPercent(value: unknown): string {
  const number = finiteNumber(value);
  return number === null ? UNKNOWN : PERCENT.format(number);
}

const countOf = (list: unknown, noun: string, view: string) => Array.isArray(list) ? `${list.length} ${noun} (${view})` : formatValue(null);

// Bulky payloads with their own view in the workspace: rows show a count, not raw JSON.
const SUMMARIZED: Record<string, (value: unknown) => string> = {
  geojson: value => countOf((value as { features?: unknown } | null)?.features, "features", "drawn on the flood map"),
  flooded_villages: value => countOf(value, "villages", "listed in the village table"),
};

/** Flatten one evidence item into readable label/value rows (nested keys joined with "›"). */
export function evidenceRows(item: EvidenceItem, prefix = ""): Array<[string, string]> {
  const rows: Array<[string, string]> = [];
  for (const [key, value] of Object.entries(item)) {
    if (!prefix && key === "type") continue;
    const label = prefix ? `${prefix} › ${humanize(key)}` : humanize(key);
    if (!prefix && Object.hasOwn(SUMMARIZED, key)) rows.push([label, SUMMARIZED[key](value)]);
    else if (value && typeof value === "object" && !Array.isArray(value)) rows.push(...evidenceRows(value as EvidenceItem, label));
    else rows.push([label, formatValue(value)]);
  }
  return rows;
}
