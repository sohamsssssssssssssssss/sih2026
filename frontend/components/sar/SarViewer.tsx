"use client";

import { useState } from "react";
import { sarImageUrl } from "@/lib/api";

type ViewerState = "LOADING" | "SUCCESS" | "UNAVAILABLE";

export function SarViewer({ scene, available }: { scene: string; available: boolean }) {
  const [state, setState] = useState<ViewerState>("LOADING");

  if (!available) {
    return (
      <figure className="panel relative grid min-h-[420px] place-items-center overflow-hidden bg-[#090c16] p-8 text-center">
        <div role="status" className="max-w-md">
          <p className="text-sm font-semibold text-slate-200">Processed SAR render is not committed to this checkout.</p>
          <p className="mt-2 text-xs leading-relaxed text-slate-400">The human interpretation beside this panel is real; the scene pixels are not available here, so no substitute imagery is displayed. The render can be regenerated from the committed pipeline in data/sar_gate/process_scenes.py.</p>
        </div>
      </figure>
    );
  }

  return (
    <figure className="panel relative min-h-[420px] overflow-hidden bg-[#090c16]">
      <img
        src={sarImageUrl(scene)}
        alt={`Processed Sentinel-1 RTC false-color render: ${scene}`}
        onLoad={() => setState("SUCCESS")}
        onError={() => setState("UNAVAILABLE")}
        className={`absolute inset-0 size-full object-contain ${state === "SUCCESS" ? "" : "invisible"}`}
      />
      {state !== "SUCCESS" && (
        <div role="status" className="absolute inset-0 grid place-items-center p-8 text-center text-sm text-slate-300">
          {state === "LOADING" ? "Loading processed SAR render…" : "Processed SAR render unavailable. No substitute imagery is shown."}
        </div>
      )}
      <div className="pointer-events-none absolute inset-0 grid-overlay opacity-20" />
      <figcaption className="absolute bottom-3 left-3 rounded-md border border-border bg-background/85 px-3 py-2 text-xs font-semibold backdrop-blur">
        Sentinel-1 · RTC gamma-0 false color (R=VV, G=VH, B=VV-VH)
      </figcaption>
    </figure>
  );
}
