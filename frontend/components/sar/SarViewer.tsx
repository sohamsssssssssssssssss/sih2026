"use client";

import { API_URL } from "@/lib/api";
import { cn } from "@/lib/utils";
import { SAR_REGIONS, type RegionKey } from "./regions";

interface SarViewerProps {
  scene: string;
  available: boolean;
  active: RegionKey | null;
  onActive: (key: RegionKey | null) => void;
}

export function SarViewer({ scene, available, active, onActive }: SarViewerProps) {
  // Genuine processed Sentinel-1 render — unchanged real-image path.
  if (available) {
    return (
      <div className="panel relative min-h-[420px] overflow-hidden bg-canvas">
        <img
          src={`${API_URL}/api/sar/${scene}/image`}
          alt={`Processed Sentinel-1 SAR scene: ${scene}`}
          className="absolute inset-0 size-full object-cover"
        />
        <div className="pointer-events-none absolute inset-0 grid-overlay opacity-20" />
        <div className="absolute left-4 top-4 rounded-md border border-border bg-background/85 px-3 py-2 text-xs font-[500] backdrop-blur">
          Sentinel-1 · RTC false color
        </div>
      </div>
    );
  }

  // No genuine render locally — educational reference visualization only.
  return <SarInterpretationGuide active={active} onActive={onActive} />;
}

const LAYOUT: RegionKey[] = ["water", "built_up", "vegetation", "terrain"];

function SarInterpretationGuide({
  active,
  onActive,
}: {
  active: RegionKey | null;
  onActive: (key: RegionKey | null) => void;
}) {
  return (
    <div>
      <div className="relative flex min-h-[420px] flex-col overflow-hidden rounded-cards bg-[#0b0e13] text-white/90">
        <div className="flex flex-col items-start justify-between gap-3 sm:flex-row border-b border-white/10 px-5 py-4">
          <div>
            <p className="text-[11px] font-[500] uppercase tracking-[0.18em] text-white/70">
              Sentinel-1 · SAR Interpretation Guide
            </p>
            <p className="mt-1 text-xs text-white/45">How radar backscatter patterns are interpreted</p>
          </div>
          <span className="max-w-full break-words rounded-full border border-white/15 px-2.5 py-1 text-[10px] font-[500] uppercase tracking-wider text-white/60">
            Reference visualization
          </span>
        </div>

        <div className="grid flex-1 grid-cols-2 grid-rows-2 gap-px bg-white/10">
          {LAYOUT.map((key) => {
            const region = SAR_REGIONS.find((r) => r.key === key)!;
            const isActive = active === key;
            const dimmed = active !== null && !isActive;
            return (
              <button
                type="button"
                key={key}
                onMouseEnter={() => onActive(key)}
                onMouseLeave={() => onActive(null)}
                onFocus={() => onActive(key)}
                onBlur={() => onActive(null)}
                aria-label={`${region.label} — ${region.category}`}
                className={cn(
                  "group relative overflow-hidden text-left outline-none transition-opacity duration-200",
                  dimmed ? "opacity-40" : "opacity-100",
                )}
              >
                <RegionTexture region={key} />
                <span
                  className={cn(
                    "pointer-events-none absolute inset-0 transition-colors duration-200",
                    isActive ? "bg-white/[0.06] ring-1 ring-inset ring-electric/70" : "bg-transparent",
                  )}
                />
                <span className="pointer-events-none absolute bottom-0 left-0 p-3">
                  <span
                    className={cn(
                      "block text-[11px] font-[600] uppercase tracking-[0.14em] transition-opacity duration-200",
                      isActive ? "text-white opacity-100" : "text-white/70 opacity-80",
                    )}
                  >
                    {region.label}
                  </span>
                  <span
                    className={cn(
                      "block text-[10px] uppercase tracking-wider transition-opacity duration-200",
                      isActive ? "text-electric opacity-100" : "text-white/40 opacity-0 group-hover:opacity-100",
                    )}
                  >
                    {region.category}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="mt-3 rounded-cards bg-canvas px-4 py-3">
        <p className="text-[11px] font-[600] uppercase tracking-[0.08em] text-ember">
          Reference visualization — not satellite imagery
        </p>
        <p className="mt-1 text-xs leading-5 text-midgray">
          The processed Sentinel-1 render is not available in this local environment. This diagram illustrates the
          radar signatures referenced by the committed analyst interpretation.
        </p>
      </div>
    </div>
  );
}

// --- Conceptual SAR textures (deterministic SVG/CSS, no imagery) ------------

// Deterministic PRNG so server and client render identical markup.
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Bright geometric blocks for the built-up quadrant (high backscatter).
const BUILT_UP_BLOCKS = (() => {
  const rng = mulberry32(42);
  const blocks: { x: number; y: number; s: number; o: number }[] = [];
  for (let gx = 0; gx < 11; gx++) {
    for (let gy = 0; gy < 11; gy++) {
      if (rng() > 0.45) {
        blocks.push({
          x: gx * 9 + rng() * 3,
          y: gy * 9 + rng() * 3,
          s: 2.5 + rng() * 4,
          o: 0.25 + rng() * 0.7,
        });
      }
    }
  }
  return blocks;
})();

function RegionTexture({ region }: { region: RegionKey }) {
  const common = "absolute inset-0 h-full w-full";
  if (region === "water") {
    return (
      <svg className={common} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <rect width="100" height="100" fill="#070a0e" />
        <g stroke="#182230" strokeWidth="0.4" opacity="0.6">
          <line x1="0" y1="26" x2="100" y2="28" />
          <line x1="0" y1="46" x2="100" y2="45" />
          <line x1="0" y1="66" x2="100" y2="68" />
          <line x1="0" y1="84" x2="100" y2="83" />
        </g>
      </svg>
    );
  }
  if (region === "built_up") {
    return (
      <svg className={common} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <rect width="100" height="100" fill="#0a0e13" />
        {BUILT_UP_BLOCKS.map((b, i) => (
          <rect key={i} x={b.x} y={b.y} width={b.s} height={b.s} fill="#e6edf3" opacity={b.o} />
        ))}
      </svg>
    );
  }
  if (region === "vegetation") {
    return (
      <svg className={common} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <filter id="sar-veg-noise" x="0" y="0" width="100%" height="100%">
            <feTurbulence type="fractalNoise" baseFrequency="0.75" numOctaves="2" seed="11" result="n" />
            <feColorMatrix in="n" type="saturate" values="0" />
          </filter>
        </defs>
        <rect width="100" height="100" fill="#161d24" />
        <rect width="100" height="100" filter="url(#sar-veg-noise)" opacity="0.5" />
      </svg>
    );
  }
  // terrain — directional ridges + slope gradient
  return (
    <svg className={common} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id="sar-terr-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#2b3744" />
          <stop offset="1" stopColor="#090d12" />
        </linearGradient>
        <pattern id="sar-terr-ridges" width="7" height="7" patternTransform="rotate(35)" patternUnits="userSpaceOnUse">
          <rect width="7" height="7" fill="url(#sar-terr-grad)" />
          <line x1="0" y1="0" x2="0" y2="7" stroke="#3c4855" strokeWidth="1.4" />
        </pattern>
      </defs>
      <rect width="100" height="100" fill="url(#sar-terr-ridges)" />
    </svg>
  );
}
