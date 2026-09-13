"use client";

import { Activity, AlertCircle, Aperture, History, Radar, Satellite, Settings2 } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const navigation = [
  { href: "/workspace", label: "Workspace", icon: Aperture },
  { href: "/change", label: "Change Intelligence", icon: AlertCircle },
  { href: "/resolution", label: "Resolution Lab", icon: Activity },
  { href: "/sar", label: "SAR Validation", icon: Radar },
  { href: "/executions", label: "Executions", icon: History },
  { href: "/system", label: "System", icon: Settings2 },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="border-b border-border bg-[#08131d]/95 lg:sticky lg:top-0 lg:h-screen lg:border-b-0 lg:border-r">
      <div className="flex h-16 items-center gap-3 px-4 lg:h-24 lg:px-6">
        <div className="grid size-10 place-items-center rounded-lg border border-accent/40 bg-accent/10 text-accent shadow-glow"><Satellite size={20} /></div>
        <div><p className="text-sm font-black tracking-[0.14em] text-white">SATQUERY AI</p><p className="text-[11px] text-slate-500">GEOSPATIAL VQA</p></div>
      </div>
      <nav aria-label="Primary navigation" className="flex gap-1 overflow-x-auto px-3 pb-3 lg:block lg:space-y-1 lg:px-4 lg:pb-0">
        {navigation.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link key={href} href={href} className={cn("flex shrink-0 items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition focus:outline-none focus:ring-2 focus:ring-accent", active ? "border border-accent/25 bg-accent/10 text-accent" : "border border-transparent text-slate-400 hover:bg-raised hover:text-slate-100") }>
              <Icon size={17} /><span>{label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="absolute bottom-0 hidden w-[248px] border-t border-border p-5 lg:block">
        <p className="eyebrow">Runtime posture</p>
        <div className="mt-3 text-xs text-slate-300">Offline-first design</div>
        <p className="mt-2 text-[11px] leading-relaxed text-slate-500">Exact golden results require the local API and committed artifact. Check System for API status.</p>
      </div>
    </aside>
  );
}
