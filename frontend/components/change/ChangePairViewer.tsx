"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { sceneImageUrl } from "@/lib/api";

export type CompareMode = "side_by_side" | "swipe" | "blink";

type EpochState = "LOADING" | "SUCCESS" | "UNAVAILABLE";

export interface Transform { scale: number; x: number; y: number }

const IDENTITY: Transform = { scale: 1, x: 0, y: 0 };
const MIN_SCALE = 1;
const MAX_SCALE = 8;

export const clampTransform = (next: Transform): Transform => ({
  scale: Math.min(MAX_SCALE, Math.max(MIN_SCALE, next.scale)),
  x: next.x,
  y: next.y,
});

/** Synchronised t1/t2 viewer.
 *
 * Both epochs share one transform, so panning or zooming either pane moves
 * the other by exactly the same amount. No imagery is synthesised, warped,
 * co-registered, or differenced: each pane shows only the pixels the API
 * served for that scene. */
export function ChangePairViewer({ t1, t2, mode, onMode }: {
  t1: string | null;
  t2: string | null;
  mode: CompareMode;
  onMode: (mode: CompareMode) => void;
}) {
  const [transform, setTransform] = useState<Transform>(IDENTITY);
  const [swipe, setSwipe] = useState(50);
  const [blinkOnT2, setBlinkOnT2] = useState(false);
  const drag = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);

  // Blink alternates the two epochs in place; the strongest way to see a
  // difference without ever computing one.
  useEffect(() => {
    if (mode !== "blink" || !t1 || !t2) return;
    const timer = window.setInterval(() => setBlinkOnT2((value) => !value), 700);
    return () => window.clearInterval(timer);
  }, [mode, t1, t2]);

  useEffect(() => { setTransform(IDENTITY); }, [t1, t2]);

  const zoomBy = useCallback((factor: number) => {
    setTransform((current) => clampTransform({ ...current, scale: current.scale * factor }));
  }, []);

  const onPointerDown = (event: React.PointerEvent) => {
    if (transform.scale === 1) return;
    drag.current = { x: event.clientX, y: event.clientY, ox: transform.x, oy: transform.y };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const onPointerMove = (event: React.PointerEvent) => {
    const origin = drag.current;
    if (!origin) return;
    setTransform((current) => ({ ...current, x: origin.ox + (event.clientX - origin.x), y: origin.oy + (event.clientY - origin.y) }));
  };
  const endDrag = () => { drag.current = null; };

  const paired = !!t1 && !!t2;

  return (
    <section className="panel overflow-hidden">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <p className="eyebrow">T1 / T2 comparison</p>
          <span className="font-mono text-[10px] text-slate-500">synchronised view</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div role="group" aria-label="Comparison mode" className="flex gap-1">
            {([["side_by_side", "Side by side"], ["swipe", "Swipe"], ["blink", "Blink"]] as [CompareMode, string][]).map(([value, label]) => (
              <button key={value} type="button" aria-pressed={mode === value} disabled={!paired}
                onClick={() => onMode(value)}
                className={`rounded border px-2.5 py-1 font-mono text-[11px] transition disabled:opacity-40 ${mode === value ? "border-accent bg-accent/10 text-accent" : "border-border text-slate-400 hover:border-accent/40"}`}>
                {label}
              </button>
            ))}
          </div>
          <div role="group" aria-label="Zoom" className="flex items-center gap-1">
            <button type="button" aria-label="Zoom out" disabled={!paired} onClick={() => zoomBy(1 / 1.5)} className="rounded border border-border px-2 py-1 font-mono text-[11px] text-slate-400 disabled:opacity-40">−</button>
            <span className="w-12 text-center font-mono text-[11px] text-accent">{transform.scale.toFixed(1)}×</span>
            <button type="button" aria-label="Zoom in" disabled={!paired} onClick={() => zoomBy(1.5)} className="rounded border border-border px-2 py-1 font-mono text-[11px] text-slate-400 disabled:opacity-40">+</button>
            <button type="button" disabled={!paired} onClick={() => setTransform(IDENTITY)} className="ml-1 rounded border border-border px-2 py-1 font-mono text-[11px] text-slate-400 disabled:opacity-40">Reset</button>
          </div>
        </div>
      </header>

      {mode === "side_by_side" ? (
        <div className="grid gap-px bg-border lg:grid-cols-2">
          <Pane epoch="T1" sceneId={t1} transform={transform} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endDrag} />
          <Pane epoch="T2" sceneId={t2} transform={transform} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endDrag} />
        </div>
      ) : (
        <div className="bg-surface p-4">
          <div data-testid="overlay-stage" className="grid-overlay relative mx-auto grid min-h-[340px] place-items-center overflow-hidden rounded-lg border border-border bg-background lg:min-h-[520px]"
            onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endDrag} onPointerCancel={endDrag}
            style={{ cursor: transform.scale > 1 ? "grab" : "default" }}>
            {paired ? <>
              <Layer sceneId={t1!} epoch="T1" transform={transform} />
              <Layer sceneId={t2!} epoch="T2" transform={transform}
                clip={mode === "swipe" ? `inset(0 0 0 ${swipe}%)` : undefined}
                hidden={mode === "blink" && !blinkOnT2} />
              {mode === "swipe" && <div aria-hidden className="pointer-events-none absolute inset-y-0 w-px bg-accent" style={{ left: `${swipe}%` }} />}
            </> : <p role="status" className="p-8 text-center text-sm text-slate-400">Load both epochs to compare. No substitute imagery is shown.</p>}
          </div>
          {mode === "swipe" && (
            <label className="mt-3 flex items-center gap-3 text-[11px] text-slate-400">
              <span className="font-mono">T1</span>
              <input type="range" min={0} max={100} value={swipe} disabled={!paired} aria-label="Swipe position"
                onChange={(event) => setSwipe(Number(event.target.value))} className="h-1 flex-1 accent-[#47d7dd]" />
              <span className="font-mono">T2</span>
            </label>
          )}
          {mode === "blink" && paired && (
            <p role="status" className="mt-3 text-center font-mono text-[11px] text-accent">showing {blinkOnT2 ? "T2" : "T1"}</p>
          )}
        </div>
      )}
    </section>
  );
}

function Layer({ sceneId, epoch, transform, clip, hidden }: {
  sceneId: string; epoch: string; transform: Transform; clip?: string; hidden?: boolean;
}) {
  return (
    <img
      src={sceneImageUrl(sceneId)}
      alt={`${epoch} scene ${sceneId}`}
      draggable={false}
      data-epoch={epoch}
      style={{ transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`, clipPath: clip, visibility: hidden ? "hidden" : undefined }}
      className="absolute max-h-full max-w-full select-none object-contain"
    />
  );
}

function Pane({ epoch, sceneId, transform, onPointerDown, onPointerMove, onPointerUp }: {
  epoch: string;
  sceneId: string | null;
  transform: Transform;
  onPointerDown: (event: React.PointerEvent) => void;
  onPointerMove: (event: React.PointerEvent) => void;
  onPointerUp: () => void;
}) {
  const [state, setState] = useState<EpochState>("LOADING");
  const imageRef = useRef<HTMLImageElement | null>(null);

  // A cached image can finish decoding before onLoad attaches; without this
  // the pane would sit in LOADING forever with real pixels on screen.
  useEffect(() => {
    const image = imageRef.current;
    if (!image || !image.complete) return;
    setState(image.naturalWidth > 0 && image.naturalHeight > 0 ? "SUCCESS" : "UNAVAILABLE");
  }, [sceneId]);

  return (
    <article className="bg-surface p-4">
      <div className="mb-2 flex items-center justify-between">
        <p className="font-mono text-[11px] font-bold tracking-[0.14em] text-accent">{epoch}</p>
        <p className="truncate pl-3 font-mono text-[10px] text-slate-500">{sceneId ?? "not loaded"}</p>
      </div>
      <div data-testid={`pane-${epoch}`} className="grid-overlay relative grid min-h-[300px] place-items-center overflow-hidden rounded-lg border border-border bg-background lg:min-h-[460px]"
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}
        style={{ cursor: transform.scale > 1 ? "grab" : "default" }}>
        {!sceneId ? (
          <p className="p-8 text-center text-sm text-slate-400">{epoch} not loaded.</p>
        ) : (
          <>
            <img
              ref={imageRef}
              src={sceneImageUrl(sceneId)}
              alt={`${epoch} scene ${sceneId}`}
              draggable={false}
              data-epoch={epoch}
              onLoad={(event) => {
                const image = event.currentTarget;
                setState(image.naturalWidth > 0 && image.naturalHeight > 0 ? "SUCCESS" : "UNAVAILABLE");
              }}
              onError={() => setState("UNAVAILABLE")}
              style={{ transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})` }}
              className={`max-h-full max-w-full select-none object-contain ${state === "SUCCESS" ? "" : "invisible"}`}
            />
            {state !== "SUCCESS" && (
              <p role="status" className="absolute p-8 text-center text-sm text-slate-400">
                {state === "LOADING" ? `Loading ${epoch} pixels…` : `${epoch} image unavailable. No substitute imagery is shown.`}
              </p>
            )}
          </>
        )}
      </div>
    </article>
  );
}
