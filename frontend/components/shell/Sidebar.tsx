"use client";

import { Activity, Aperture, History, Radar, Satellite, Settings2 } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { runtimeDotColour, useRuntimeState, type RuntimeState } from "@/components/status/RuntimeStatus";

const navigation = [
  { href: "/workspace", label: "Workspace", icon: Aperture },
  { href: "/resolution", label: "Resolution Lab", icon: Activity },
  { href: "/sar", label: "SAR Validation", icon: Radar },
  { href: "/executions", label: "Executions", icon: History },
  { href: "/system", label: "System", icon: Settings2 },
];

const POSTURE: Record<RuntimeState, { label: string; note: string }> = {
  checking: { label: "Checking runtime…", note: "Contacting the local API." },
  ready: { label: "All capabilities ready", note: "Live inference and cached replay are available." },
  degraded: { label: "Partially available", note: "Some live models are not ready here. Cached replay stays explicit." },
  unavailable: { label: "Analysis disabled", note: "The execution trace failed verification. See System." },
  offline: { label: "API not connected", note: "Start the backend on port 8000." },
};

export function Sidebar() {
  const pathname = usePathname();
  const state = useRuntimeState();
  const posture = POSTURE[state];
  return (
    <aside className="bg-onyx lg:sticky lg:top-0 lg:h-screen">
      <div className="flex h-16 items-center gap-3 px-4 lg:h-24 lg:px-6">
        <div className="grid size-10 place-items-center rounded-cards bg-obsidian text-ivory"><Satellite size={19} /></div>
        <div><p className="text-sm tracking-[0.14em] text-ivory" style={{ fontWeight: 480 }}>SATQUERY AI</p><p className="text-[11px] text-ash">GEOSPATIAL VQA</p></div>
      </div>
      <nav aria-label="Primary navigation" className="flex gap-1.5 overflow-x-auto px-3 pb-3 lg:block lg:space-y-1.5 lg:px-4 lg:pb-0">
        {navigation.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link key={href} href={href} className={cn("flex shrink-0 items-center gap-3 rounded-pill px-4 py-2.5 text-[15px] transition focus:outline-none focus:ring-2 focus:ring-cobalt/60", active ? "bg-cobalt text-white" : "text-ash hover:bg-graphite hover:text-ivory") }>
              <Icon size={17} /><span>{label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="absolute bottom-0 hidden w-[248px] p-6 lg:block">
        <p className="eyebrow">Runtime posture</p>
        <div className="mt-3 flex items-center gap-2 text-[13px] text-ivory"><span className={cn("size-2 rounded-full", runtimeDotColour(state))} /> {posture.label}</div>
        <p className="mt-2 text-[12px] leading-relaxed text-ash">{posture.note}</p>
      </div>
    </aside>
  );
}
