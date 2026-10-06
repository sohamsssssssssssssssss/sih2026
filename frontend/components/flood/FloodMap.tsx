"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
// Type-only; @types/geojson ships with maplibre-gl, no added dependency.
import type { FeatureCollection } from "geojson";
import type { EvidenceItem } from "@/lib/types";
import { formatNumber } from "@/lib/evidence";
import { MAPLIBRE_WORKER_URL } from "@/lib/maplibre";

// The view is never a hard-coded place: the map opens on the bounds of the polygons the
// backend returned (EPSG:4326 lon/lat). With no usable polygons, no map is drawn at all.

type MapStyle = Exclude<maplibregl.MapOptions["style"], string | undefined>;
export type Bounds = [[number, number], [number, number]];

// Categorical pair validated for colour-vision deficiency; the outline style (solid vs dashed) is the second cue.
const NEW_WATER = "#2a78d6";
const RECEDED_WATER = "#eb6834";
const OSM_ATTRIBUTION = "© OpenStreetMap contributors";
const MAX_FIT_ZOOM = 15;

function featureCollection(value: unknown): FeatureCollection | null {
  const candidate = value as { type?: unknown; features?: unknown } | null;
  return candidate?.type === "FeatureCollection" && Array.isArray(candidate.features) ? candidate as FeatureCollection : null;
}

/** West-south and east-north corners over every finite [lon, lat] in the collection; null when there are none. */
export function geojsonBounds(collection: FeatureCollection | null): Bounds | null {
  // Running min/max, not Math.min(...all): 500 large polygons can exceed the argument limit.
  const box = { west: Infinity, south: Infinity, east: -Infinity, north: -Infinity };
  const visit = (node: unknown): void => {
    if (!Array.isArray(node)) return;
    if (typeof node[0] !== "number") { node.forEach(visit); return; }
    const [lon, lat] = node;
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) return;
    box.west = Math.min(box.west, lon); box.east = Math.max(box.east, lon);
    box.south = Math.min(box.south, lat); box.north = Math.max(box.north, lat);
  };
  collection?.features.forEach(feature => visit((feature as { geometry?: { coordinates?: unknown } | null } | null)?.geometry?.coordinates));
  return box.west <= box.east ? [[box.west, box.south], [box.east, box.north]] : null;
}

function floodStyle(data: FeatureCollection): MapStyle {
  return {
    version: 8,
    sources: {
      osm: { type: "raster", tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256, maxzoom: 19, attribution: OSM_ATTRIBUTION },
      water: { type: "geojson", data },
    },
    layers: [
      { id: "background", type: "background", paint: { "background-color": "#f5f5f7" } },
      { id: "osm", type: "raster", source: "osm", paint: { "raster-saturation": -0.7, "raster-opacity": 0.85 } },
      { id: "new-water-fill", type: "fill", source: "water", filter: ["==", ["get", "change"], "new_water"], paint: { "fill-color": NEW_WATER, "fill-opacity": 0.55 } },
      { id: "new-water-line", type: "line", source: "water", filter: ["==", ["get", "change"], "new_water"], paint: { "line-color": NEW_WATER, "line-width": 1.5 } },
      { id: "receded-water-fill", type: "fill", source: "water", filter: ["==", ["get", "change"], "receded_water"], paint: { "fill-color": RECEDED_WATER, "fill-opacity": 0.35 } },
      { id: "receded-water-line", type: "line", source: "water", filter: ["==", ["get", "change"], "receded_water"], paint: { "line-color": RECEDED_WATER, "line-width": 2, "line-dasharray": [2, 1] } },
    ],
  };
}

function polygonCount(item: EvidenceItem, shown: number): string {
  const total = formatNumber(item.feature_count);
  const ordering = typeof item.ordering === "string" && item.ordering ? ` (${item.ordering})` : "";
  return item.truncated === true ? `Showing ${formatNumber(shown)} of ${total} polygons${ordering}` : `Showing all ${formatNumber(shown)} polygons`;
}

const LEGEND = [
  { label: "New water", detail: "water at T2 that was not water at T1 · solid outline", color: NEW_WATER, border: "solid" },
  { label: "Receded water", detail: "water at T1 that is no longer water at T2 · dashed outline", color: RECEDED_WATER, border: "dashed" },
] as const;

/** Water-change polygons on an OpenStreetMap basemap, framed by the polygons' own extent. */
export function FloodMap({ item }: { item: EvidenceItem }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const geojson = useMemo(() => featureCollection(item.geojson), [item]);
  const bounds = useMemo(() => geojsonBounds(geojson), [geojson]);
  const shown = geojson?.features.length ?? 0;

  useEffect(() => {
    if (!containerRef.current || !geojson || !bounds) return;
    let map: maplibregl.Map;
    try {
      maplibregl.setWorkerUrl(MAPLIBRE_WORKER_URL);
      map = new maplibregl.Map({
        container: containerRef.current, style: floodStyle(geojson), bounds,
        fitBoundsOptions: { padding: 32, maxZoom: MAX_FIT_ZOOM }, attributionControl: false,
      });
    } catch (reason) {
      setMapError(`The map could not start in this browser (${reason instanceof Error ? reason.message : "WebGL unavailable"}).`);
      return;
    }
    map.on("error", () => setMapError("Some basemap tiles or layers failed to load; polygons may be drawn on a plain background."));
    return () => map.remove();
  }, [geojson, bounds]);

  return (
    <section className="panel p-5" aria-label="Water-change map">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="eyebrow">Water-change polygons</p>
        {bounds && <p className="text-xs text-deepgray">{polygonCount(item, shown)}</p>}
      </div>
      {bounds ? (
        <>
          <div className="relative mt-3 h-[380px] overflow-hidden rounded-xl border border-border bg-surface">
            <div ref={containerRef} className="absolute inset-0" role="region" aria-label="Interactive map of water-change polygons; the legend below describes the layers" />
            <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer" className="absolute bottom-0 right-0 rounded-tl-md bg-background/85 px-2 py-0.5 text-[10px] text-deepgray hover:underline">{OSM_ATTRIBUTION}</a>
          </div>
          <ul className="mt-3 space-y-1.5 text-xs text-deepgray" aria-label="Map legend">
            {LEGEND.map(entry => (
              <li key={entry.label} className="flex items-center gap-2">
                <span aria-hidden className="inline-block h-3 w-6 shrink-0 rounded-sm border-2" style={{ backgroundColor: `${entry.color}66`, borderColor: entry.color, borderStyle: entry.border }} />
                <span><span className="font-[600] text-ink">{entry.label}</span>: {entry.detail}</span>
              </li>
            ))}
          </ul>
          {mapError && <p role="status" className="mt-2 text-xs text-warning">{mapError}</p>}
        </>
      ) : (
        <p className="mt-2 text-sm text-deepgray">{shown === 0 ? "No water-change polygons were returned for this pair, so there is nothing to map." : "The returned polygons carry no usable coordinates, so no map is drawn."}</p>
      )}
    </section>
  );
}
