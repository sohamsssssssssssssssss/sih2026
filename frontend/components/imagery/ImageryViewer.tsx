"use client";

import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import { Crosshair, Layers3 } from "lucide-react";
import { getSceneImageUrl } from "@/lib/api";
import type { OverlayBox } from "@/lib/evidence";
import { GOLDEN_SCENE } from "@/lib/workspace-contract";

// Scenes are NOT placed on Earth here. The golden LoveDA scene ships no geographic
// coordinates, and the backend deliberately reports `location: null` (backend/schemas.py);
// any lat/lon shown for it would be invented. Even a georeferenced upload is displayed in
// the same neutral frame, because this viewer has no basemap to align it with.
//
// The map is therefore a pan/zoom canvas in an arbitrary frame centred on the origin,
// never a claim about where on Earth the scene sits. Do not re-introduce a real-world
// centre or a coordinate readout unless the backend actually returns one.
const NEUTRAL_CENTRE: [number, number] = [0, 0];
const E = 0.06; // half-extent of the arbitrary canvas frame (long side)

type GeoJSONData = Exclude<Parameters<maplibregl.GeoJSONSource["setData"]>[0], string>;

export interface ImageryViewerProps {
  sceneId?: string;
  label?: string;
  /** Pixel size when known, so the frame keeps the scene's aspect ratio. */
  width?: number | null;
  height?: number | null;
  georeferenced?: boolean;
  /** Evidence regions in normalized xyxy image coordinates. */
  boxes?: OverlayBox[];
}

/** Half-extents of the arbitrary frame: the long side spans 2E. */
export function frameExtent(width?: number | null, height?: number | null): [number, number] {
  if (!width || !height || width <= 0 || height <= 0) return [E, E];
  return width >= height ? [E, (E * height) / width] : [(E * width) / height, E];
}

/** Normalized image boxes → polygons in the arbitrary frame (y grows downward in the image). */
export function boxesToGeoJSON(boxes: OverlayBox[], [hw, hh]: [number, number]): GeoJSONData {
  const x = (value: number) => -hw + value * 2 * hw;
  const y = (value: number) => hh - value * 2 * hh;
  return {
    type: "FeatureCollection",
    features: boxes.map(box => ({
      type: "Feature",
      properties: { index: box.index, label: box.label, kind: box.kind },
      geometry: {
        type: "Polygon",
        coordinates: [[[x(box.x0), y(box.y0)], [x(box.x1), y(box.y0)], [x(box.x1), y(box.y1)], [x(box.x0), y(box.y1)], [x(box.x0), y(box.y0)]]],
      },
    })),
  };
}

export function ImageryViewer({ sceneId = GOLDEN_SCENE.id, label, width, height, georeferenced = false, boxes = [] }: ImageryViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const boxesRef = useRef(boxes);
  const [failed, setFailed] = useState(false);
  const [ready, setReady] = useState(false);
  const extent = frameExtent(width, height);
  const [hw, hh] = extent;
  boxesRef.current = boxes;

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: { version: 8, sources: {}, layers: [{ id: "background", type: "background", paint: { "background-color": "#f5f5f7" } }] },
      center: NEUTRAL_CENTRE,
      zoom: 11,
      attributionControl: false,
      interactive: true,
    });
    mapRef.current = map;
    map.on("load", () => {
      map.addSource("scene", {
        type: "image",
        url: getSceneImageUrl(sceneId),
        coordinates: [[-hw, hh], [hw, hh], [hw, -hh], [-hw, -hh]],
      });
      map.addLayer({ id: "scene", type: "raster", source: "scene", paint: { "raster-opacity": 0.94, "raster-fade-duration": 0 } });
      map.addSource("evidence", { type: "geojson", data: boxesToGeoJSON(boxesRef.current, [hw, hh]) });
      map.addLayer({ id: "evidence-fill", type: "fill", source: "evidence", paint: { "fill-color": "#0071e3", "fill-opacity": 0.12 } });
      map.addLayer({ id: "evidence-line", type: "line", source: "evidence", paint: { "line-color": "#0071e3", "line-width": 2 } });
      map.fitBounds([[-hw, -hh], [hw, hh]], { padding: 24, animate: false });
      setReady(true);
    });
    map.on("error", (event) => {
      if (String(event.error?.message ?? "").includes("image")) setFailed(true);
    });
    return () => { map.remove(); mapRef.current = null; };
    // The parent remounts this viewer (key) when the scene changes.
  }, []);

  useEffect(() => {
    if (!ready) return;
    mapRef.current?.getSource<maplibregl.GeoJSONSource>("evidence")?.setData(boxesToGeoJSON(boxes, [hw, hh]));
  }, [ready, boxes, hw, hh]);

  const golden = sceneId === GOLDEN_SCENE.id;
  return (
    <div className="panel relative min-h-[420px] overflow-hidden lg:min-h-[600px]">
      <div ref={containerRef} className="absolute inset-0" aria-label={`Interactive view of scene ${label ?? sceneId}`} />
      <div className="pointer-events-none absolute inset-0 grid-overlay opacity-30" />
      <div className="absolute left-4 top-4 flex max-w-[70%] items-center gap-2 truncate rounded-md border border-border bg-background/85 px-3 py-2 text-xs font-[500] backdrop-blur"><Layers3 size={14} className="shrink-0 text-ash" /> <span className="truncate">{golden ? "LoveDA RGB · 0.3 m" : label ?? sceneId}</span></div>
      {boxes.length > 0 && <div className="absolute right-4 top-4 rounded-md border border-cobalt/30 bg-background/85 px-3 py-2 text-xs font-[500] text-cobalt backdrop-blur">{boxes.length} evidence region{boxes.length === 1 ? "" : "s"}</div>}
      <div className="absolute bottom-4 left-4 rounded-md border border-border bg-background/85 px-3 py-2 font-mono text-[10px] text-midgray backdrop-blur" title="Shown in an arbitrary pan/zoom frame, not on a geographic map">
        {georeferenced ? "Georeferenced · shown in a neutral frame, not on a map" : "Location not recorded · not georeferenced"}
      </div>
      {boxes.length === 0 && <Crosshair className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-electric drop-shadow-[0_0_10px_rgba(255,255,255,0.9)]" size={28} strokeWidth={1.25} />}
      {failed && <div className="absolute inset-x-4 bottom-4 rounded-lg border border-warning/30 bg-background/90 p-3 text-sm text-warning">Local scene pixels are unavailable for this scene.</div>}
    </div>
  );
}
