import type { SceneCatalog } from "./types";
import { GOLDEN_SCENE } from "./workspace-contract";

/** One selectable scene in the workspace, normalised from the curated pack or an upload. */
export interface SceneOption {
  id: string;
  label: string;
  kind: "golden" | "curated" | "upload";
  available: boolean;
  unavailableReason: string | null;
  /** A question the scene was curated for, if any. Never a promise of a cached answer. */
  suggestedQuestion: string | null;
  source: string | null;
  sensor: string | null;
  /** Sent as the analyze `sensor` field. */
  requestSensor: string | null;
  gsd: number | null;
  modality: string | null;
  format: string | null;
  width: number | null;
  height: number | null;
  acquisitionTime: string | null;
  georeferenced: boolean;
  hasNativeRaster: boolean;
  uploadedAt: string | null;
}

export const GOLDEN_OPTION: SceneOption = {
  id: GOLDEN_SCENE.id,
  label: "LoveDA golden scene",
  kind: "golden",
  available: true,
  unavailableReason: null,
  suggestedQuestion: GOLDEN_SCENE.question,
  source: GOLDEN_SCENE.source,
  sensor: null,
  requestSensor: "Unknown",
  gsd: GOLDEN_SCENE.gsd,
  modality: "optical",
  format: "PNG",
  width: null,
  height: null,
  acquisitionTime: null,
  georeferenced: false,
  hasNativeRaster: false,
  uploadedAt: null,
};

export function toSceneOptions(catalog: SceneCatalog | null): SceneOption[] {
  if (!catalog) return [GOLDEN_OPTION];
  // The golden scene is always addressed by its pinned identifier: that is the only
  // identifier the backend can replay from the committed artifact.
  const goldenEntry = catalog.scenes.find(scene => scene.source?.source_id === GOLDEN_SCENE.id);
  const golden: SceneOption = goldenEntry
    ? { ...GOLDEN_OPTION, available: goldenEntry.available, unavailableReason: goldenEntry.unavailable_reason }
    : GOLDEN_OPTION;
  const curated = catalog.scenes
    .filter(scene => scene.catalog_visible && scene !== goldenEntry)
    .map<SceneOption>(scene => ({
      id: scene.id,
      label: scene.title,
      kind: "curated",
      available: scene.available,
      unavailableReason: scene.unavailable_reason,
      suggestedQuestion: scene.question || null,
      source: [scene.source?.dataset, scene.source?.dataset_split].filter(Boolean).join(" · ") || null,
      sensor: scene.source?.sensor ?? null,
      requestSensor: scene.source?.sensor ?? null,
      gsd: typeof scene.source?.gsd === "number" ? scene.source.gsd : null,
      modality: null,
      format: null,
      width: null,
      height: null,
      acquisitionTime: scene.source?.acquisition_date ?? null,
      georeferenced: false,
      hasNativeRaster: false,
      uploadedAt: null,
    }));
  const uploads = catalog.uploads.map<SceneOption>(upload => ({
    id: upload.scene_id,
    label: upload.filename,
    kind: "upload",
    available: true,
    unavailableReason: null,
    suggestedQuestion: null,
    source: "Uploaded scene",
    sensor: upload.sensor,
    requestSensor: upload.sensor,
    gsd: null,
    modality: upload.modality,
    format: upload.format,
    width: upload.width,
    height: upload.height,
    acquisitionTime: upload.acquisition_time,
    georeferenced: upload.georeferenced,
    hasNativeRaster: upload.has_native_raster,
    uploadedAt: upload.uploaded_at,
  }));
  return [golden, ...curated, ...uploads];
}

export function sceneSummary(scene: SceneOption) {
  if (scene.kind === "upload") {
    return [scene.format, scene.modality && scene.modality !== "unknown" ? scene.modality : null, scene.acquisitionTime ? "timestamped" : null]
      .filter(Boolean).join(" · ");
  }
  return scene.kind === "golden" ? "Pinned · cached replay available" : "Curated";
}

export const PAIR_QUESTIONS = {
  change: "What changed between these images?",
  opticalSar: "Compare the optical and SAR evidence for this area.",
} as const;
