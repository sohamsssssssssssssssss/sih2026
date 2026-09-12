"use client";

import { useEffect, useRef, useState } from "react";
import { Minus, Plus, Maximize2 } from "lucide-react";
import { getSceneImageUrl } from "@/lib/api";
import { GOLDEN_SCENE } from "@/lib/workspace-contract";

export type ImageLoadState = "loading" | "ready" | "failed";

export function reconcileImageLoad(current: ImageLoadState, complete: boolean, naturalWidth: number): ImageLoadState {
  if (!complete) return current;
  return naturalWidth > 0 ? "ready" : "failed";
}

export function ImageryViewer() {
  const viewport = useRef<HTMLDivElement>(null);
  const image = useRef<HTMLImageElement>(null);
  const transform = useRef({ scale: 1, x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number } | null>(null);
  const [state, setState] = useState<ImageLoadState>("loading");
  const [zoom, setZoom] = useState(100);

  const paint = () => {
    const box = viewport.current;
    const img = image.current;
    if (!box || !img) return;
    const t = transform.current;
    const fit = Math.min(box.clientWidth / (img.naturalWidth || 1), box.clientHeight / (img.naturalHeight || 1));
    const maxX = Math.max(0, (img.naturalWidth * fit * t.scale - box.clientWidth) / 2);
    const maxY = Math.max(0, (img.naturalHeight * fit * t.scale - box.clientHeight) / 2);
    t.x = Math.max(-maxX, Math.min(maxX, t.x));
    t.y = Math.max(-maxY, Math.min(maxY, t.y));
    img.style.transform = `translate(${t.x}px, ${t.y}px) scale(${t.scale})`;
  };
  const fit = () => { transform.current = { scale: 1, x: 0, y: 0 }; setZoom(100); paint(); };
  const changeZoom = (factor: number) => {
    transform.current.scale = Math.max(1, Math.min(6, transform.current.scale * factor));
    setZoom(Math.round(transform.current.scale * 100));
    paint();
  };

  useEffect(() => {
    const box = viewport.current;
    const img = image.current;
    if (!box || !img) return;
    const observer = new ResizeObserver(paint);
    observer.observe(box);
    if (img.complete) {
      const settled = reconcileImageLoad("loading", img.complete, img.naturalWidth);
      setState(settled);
      if (settled === "ready") fit();
    }
    const timeout = setTimeout(() => setState(current => current === "loading" ? "failed" : current), 15000);
    return () => { observer.disconnect(); clearTimeout(timeout); };
  }, []);

  return (
    <section className="overflow-hidden rounded-16 border border-border bg-background" aria-label="Golden scene image viewer">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-3 py-2">
        <span className="text-xs text-subtitle">Local scene image · not georeferenced</span>
        <div className="flex items-center gap-2">
          <button type="button" disabled={state !== "ready" || zoom <= 100} aria-label="Zoom out" onClick={() => changeZoom(1 / 1.25)} className="rounded p-2 text-white focus-visible:outline focus-visible:outline-accent disabled:opacity-40"><Minus size={15} /></button>
          <output className="w-10 text-center text-xs text-subtitle" aria-label="Image zoom">{zoom}%</output>
          <button type="button" disabled={state !== "ready" || zoom >= 600} aria-label="Zoom in" onClick={() => changeZoom(1.25)} className="rounded p-2 text-white focus-visible:outline focus-visible:outline-accent disabled:opacity-40"><Plus size={15} /></button>
          <button type="button" disabled={state !== "ready"} onClick={fit} className="flex items-center gap-1 rounded p-2 text-xs text-white focus-visible:outline focus-visible:outline-accent disabled:opacity-40"><Maximize2 size={14} /> Fit</button>
        </div>
      </div>
      <div ref={viewport} tabIndex={0} role="region" aria-label="Scene image. Use plus and minus to zoom, arrow keys to pan, and zero to fit."
        className="relative h-[420px] overflow-hidden outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent lg:h-[clamp(360px,52vh,640px)]"
        style={{ touchAction: zoom > 100 ? "none" : "pan-y", cursor: zoom > 100 ? "grab" : "default" }}
        onKeyDown={event => {
          if (state !== "ready") return;
          if (["+", "=", "-", "0", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) event.preventDefault();
          if (event.key === "+" || event.key === "=") changeZoom(1.25);
          if (event.key === "-") changeZoom(1 / 1.25);
          if (event.key === "0") fit();
          if (event.key === "ArrowLeft") transform.current.x += 35;
          if (event.key === "ArrowRight") transform.current.x -= 35;
          if (event.key === "ArrowUp") transform.current.y += 35;
          if (event.key === "ArrowDown") transform.current.y -= 35;
          paint();
        }}
        onPointerDown={event => {
          if (state !== "ready" || zoom <= 100 || event.button !== 0) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          drag.current = { x: event.clientX, y: event.clientY };
        }}
        onPointerMove={event => {
          if (!drag.current) return;
          transform.current.x += event.clientX - drag.current.x;
          transform.current.y += event.clientY - drag.current.y;
          drag.current = { x: event.clientX, y: event.clientY };
          paint();
        }}
        onPointerUp={() => { drag.current = null; }}
        onPointerCancel={() => { drag.current = null; }}
        onLostPointerCapture={() => { drag.current = null; }}
      >
        <img ref={image} src={getSceneImageUrl(GOLDEN_SCENE.id)} alt="Verified LoveDA golden scene" draggable={false}
          className="pointer-events-none absolute inset-0 h-full w-full select-none object-contain"
          style={{ visibility: state === "ready" ? "visible" : "hidden" }}
          onLoad={() => { setState("ready"); fit(); }} onError={() => setState("failed")} />
        {state !== "ready" && <div role="status" className="absolute inset-0 grid place-items-center p-8 text-center text-sm text-subtitle">
          {state === "loading" ? "Loading local scene image…" : "Scene image unavailable"}
        </div>}
      </div>
      <p className="border-t border-border px-3 py-2 text-[11px] text-tertiary">Zoom with + / −, drag to pan, or use arrow keys. Fit resets the image.</p>
    </section>
  );
}
