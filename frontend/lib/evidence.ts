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

/** Flatten one evidence item into readable label/value rows (nested keys joined with "›"). */
export function evidenceRows(item: EvidenceItem, prefix = ""): Array<[string, string]> {
  const rows: Array<[string, string]> = [];
  for (const [key, value] of Object.entries(item)) {
    if (!prefix && key === "type") continue;
    const label = prefix ? `${prefix} › ${humanize(key)}` : humanize(key);
    if (value && typeof value === "object" && !Array.isArray(value)) rows.push(...evidenceRows(value as EvidenceItem, label));
    else rows.push([label, formatValue(value)]);
  }
  return rows;
}
