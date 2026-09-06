"use client";

import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import { Crosshair, Layers3 } from "lucide-react";
import { API_URL } from "@/lib/api";

const SCENE_ID = "loveda_LoveDA_images_png_0_gsd0.3";

export function ImageryViewer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: { version: 8, sources: {}, layers: [{ id: "background", type: "background", paint: { "background-color": "#08131d" } }] },
      center: [72.88, 19.08],
      zoom: 11,
      attributionControl: false,
      interactive: true,
    });
    mapRef.current = map;
    map.on("load", () => {
      map.addSource("golden-scene", {
        type: "image",
        url: `${API_URL}/api/scenes/${SCENE_ID}/image`,
        coordinates: [[72.82, 19.14], [72.94, 19.14], [72.94, 19.02], [72.82, 19.02]],
      });
      map.addLayer({ id: "golden-scene", type: "raster", source: "golden-scene", paint: { "raster-opacity": 0.94, "raster-fade-duration": 0 } });
    });
    map.on("error", (event) => {
      if (String(event.error?.message ?? "").includes("image")) setFailed(true);
    });
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  return (
    <div className="panel relative min-h-[420px] overflow-hidden lg:min-h-[600px]">
      <div ref={containerRef} className="absolute inset-0" aria-label="Interactive view of the verified golden satellite scene" />
      <div className="pointer-events-none absolute inset-0 grid-overlay opacity-30" />
      <div className="absolute left-4 top-4 flex items-center gap-2 rounded-md border border-border bg-background/85 px-3 py-2 text-xs font-semibold backdrop-blur"><Layers3 size={14} className="text-accent" /> LoveDA RGB · 0.3 m</div>
      <div className="absolute bottom-4 left-4 rounded-md border border-border bg-background/85 px-3 py-2 font-mono text-[10px] text-slate-300 backdrop-blur">19.08° N / 72.88° E</div>
      <Crosshair className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-accent drop-shadow-[0_0_8px_#47d7dd]" size={28} strokeWidth={1.25} />
      {failed && <div className="absolute inset-x-4 bottom-4 rounded-lg border border-warning/30 bg-background/90 p-3 text-sm text-warning">Local scene pixels are unavailable; cached analysis remains usable.</div>}
    </div>
  );
}
