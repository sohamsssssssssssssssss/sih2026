"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import { getSceneImageUrl, getScenes } from "@/lib/api";
import { overlayBoxes } from "@/lib/evidence";
import { GOLDEN_OPTION, sceneSummary, toSceneOptions, type SceneOption } from "@/lib/scenes";
import type { AnalysisResponse, SceneCatalog, SceneUploadMetadata, SceneUploadResponse, UploadedScene } from "@/lib/types";
import { ImageryViewer } from "@/components/imagery/ImageryViewer";
import { SceneMetadata } from "@/components/imagery/SceneMetadata";
import { UploadScene } from "@/components/imagery/UploadScene";
import { QueryPanel } from "@/components/analysis/QueryPanel";
import { FloodMap } from "@/components/flood/FloodMap";
import { VillageTable } from "@/components/flood/VillageTable";
import { WaterChangeSummary } from "@/components/flood/WaterChangeSummary";

/** Stand-in listing for a just-uploaded scene until GET /api/scenes reports it. */
function provisionalUpload(scene: SceneUploadResponse, metadata: SceneUploadMetadata): UploadedScene {
  return {
    scene_id: scene.scene_id,
    filename: scene.filename,
    format: scene.format,
    width: scene.width,
    height: scene.height,
    modality: metadata.modality ?? "unknown",
    sensor: scene.sensor,
    acquisition_time: scene.acquisition_date,
    has_native_raster: scene.format === "TIFF",
    georeferenced: false,
    uploaded_at: new Date().toISOString(),
  };
}

export function Workspace() {
  const [catalog, setCatalog] = useState<SceneCatalog | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [loadingCatalog, setLoadingCatalog] = useState(true);
  const [pending, setPending] = useState<UploadedScene[]>([]);
  const [sceneId, setSceneId] = useState<string>(GOLDEN_OPTION.id);
  const [sceneId2, setSceneId2] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  const load = useCallback(async () => {
    setLoadingCatalog(true);
    try { setCatalog(await getScenes()); setCatalogError(null); }
    catch (reason) { setCatalogError(reason instanceof Error ? reason.message : "Scene catalog unavailable."); }
    finally { setLoadingCatalog(false); }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const options = useMemo(() => {
    const listed = new Set(catalog?.uploads.map(upload => upload.scene_id) ?? []);
    const extra = pending.filter(upload => !listed.has(upload.scene_id));
    const merged: SceneCatalog = { version: catalog?.version ?? "unknown", scenes: catalog?.scenes ?? [], uploads: [...extra, ...(catalog?.uploads ?? [])] };
    return toSceneOptions(merged);
  }, [catalog, pending]);

  const scene = options.find(option => option.id === sceneId) ?? GOLDEN_OPTION;
  const second = sceneId2 ? options.find(option => option.id === sceneId2) ?? null : null;
  const boxes = useMemo(() => overlayBoxes(result?.evidence), [result]);
  const flood = useMemo(() => {
    const evidence = Array.isArray(result?.evidence) ? result.evidence : [];
    const find = (type: string) => evidence.find(item => item.type === type);
    return { stats: find("water_change_statistics"), polygons: find("water_change_polygons"), villages: find("village_flooding") };
  }, [result]);

  function select(id: string) {
    setSceneId(id);
    setSceneId2(null);
    setResult(null);
  }

  function onUploaded(uploaded: SceneUploadResponse, metadata: SceneUploadMetadata) {
    setPending(current => [provisionalUpload(uploaded, metadata), ...current.filter(upload => upload.scene_id !== uploaded.scene_id)]);
    select(uploaded.scene_id);
    void load();
  }

  const groups: Array<[string, SceneOption[]]> = [
    ["Pinned", options.filter(option => option.kind === "golden")],
    ["Curated", options.filter(option => option.kind === "curated")],
    ["Uploads", options.filter(option => option.kind === "upload")],
  ];

  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1.45fr)_minmax(360px,0.75fr)]">
      <div>
        <ImageryViewer key={scene.id} sceneId={scene.id} label={scene.label} width={scene.width} height={scene.height} georeferenced={scene.georeferenced} boxes={boxes} />
        {second && (
          <div className="panel mt-4 flex items-center gap-4 p-4">
            <img src={getSceneImageUrl(second.id)} alt={`Preview of second scene ${second.label}`} className="size-20 shrink-0 rounded-md border border-border bg-surface object-cover" />
            <div className="min-w-0"><p className="eyebrow">Second scene</p><p className="mt-1 truncate text-sm font-[500] text-ink">{second.label}</p><p className="truncate font-mono text-[10px] text-midgray">{second.id}</p></div>
          </div>
        )}
        {(flood.stats || flood.polygons || flood.villages) && (
          <div className="mt-4 space-y-4">
            {flood.stats && <WaterChangeSummary item={flood.stats} />}
            {flood.polygons && <FloodMap item={flood.polygons} />}
            {flood.villages && <VillageTable item={flood.villages} />}
          </div>
        )}
        <SceneMetadata scene={scene} />
      </div>
      <div className="space-y-4">
        <section className="panel p-5 sm:p-6" aria-label="Scene selection">
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="scene-picker" className="eyebrow">Scene</label>
            <button type="button" onClick={() => void load()} disabled={loadingCatalog} className="inline-flex items-center gap-1 text-xs text-ash hover:text-ivory disabled:opacity-50" aria-label="Refresh scene list"><RefreshCw size={12} className={loadingCatalog ? "animate-spin" : ""} /> Refresh</button>
          </div>
          <select id="scene-picker" value={scene.id} onChange={event => select(event.target.value)} className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2.5 text-sm text-ink outline-none focus:border-accent">
            {groups.filter(([, items]) => items.length > 0).map(([label, items]) => (
              <optgroup key={label} label={label}>
                {items.map(option => <option key={option.id} value={option.id} disabled={!option.available}>{option.label}{option.available ? "" : " (unavailable)"}</option>)}
              </optgroup>
            ))}
          </select>
          <p className="mt-2 text-xs text-midgray">{sceneSummary(scene)}</p>
          {catalogError && <p role="status" className="mt-2 text-xs text-warning">Scene list unavailable ({catalogError}). Only the pinned scene is shown.</p>}
          <div className="mt-5 border-t border-border pt-4"><p className="eyebrow mb-3">Add a scene</p><UploadScene onUploaded={onUploaded} /></div>
        </section>
        <QueryPanel key={scene.id} scene={scene} scenes={options} onResult={setResult} onSecondScene={setSceneId2} />
      </div>
    </div>
  );
}
