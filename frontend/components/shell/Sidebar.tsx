"use client";

import { Activity, Aperture, History, Radar, Satellite, Settings2 } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

const navigation = [
  { href: "/workspace", label: "Workspace", icon: Aperture },
  { href: "/resolution", label: "Resolution Lab", icon: Activity },
  { href: "/sar", label: "SAR Validation", icon: Radar },
  { href: "/executions", label: "Execution History", icon: History },
  { href: "/system", label: "System", icon: Settings2 },
];

function getDotClass(state: string) {
  if (state === "connecting") return "size-1.5 rounded-full";
  if (state === "online") return "size-1.5 rounded-full bg-success";
  return "size-1.5 rounded-full bg-error";
}

function getStatusText(state: string) {
  if (state === "connecting") return "CONNECTING";
  if (state === "online") return "ONLINE";
  return "OFFLINE";
}

export function Sidebar() {
  const pathname = usePathname();
  const [state, setState] = useState<"connecting" | "online" | "offline">("connecting");

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE || 'http://127.0.0.1:8000'}/api/health`);
        if (res.ok) {
          const data = await res.json();
          if (data.status === "ready") {
            setState("online");
          } else {
            setState("offline");
          }
        } else {
          setState("offline");
        }
      } catch {
        setState("offline");
      }
    }
    checkHealth();
  }, []);

  return (
    <aside className="border-b border-border bg-surface-elevated lg:sticky lg:top-0 lg:flex lg:h-screen lg:flex-col lg:border-b-0 lg:border-r">
      <Link href="/workspace" className="flex h-20 shrink-0 items-center gap-3 border-b border-border px-4 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">
        <span className="grid size-9 place-items-center rounded border border-primary/40 bg-primary/10 text-cyan"><Satellite size={18} aria-hidden="true" /></span>
        <div><p className="text-sm font-bold tracking-[0.08em] text-primary">SATQUERY <span className="text-cyan">AI</span></p><p className="mt-0.5 text-[8px] tracking-[0.12em] text-subtitle">GEOSPATIAL INTELLIGENCE</p></div>
      </Link>
      <nav aria-label="Primary navigation" className="flex gap-1 overflow-x-auto px-3 py-3 lg:block lg:space-y-1">
        {navigation.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link key={href} href={href} aria-current={active ? "page" : undefined} className={cn("flex shrink-0 items-center gap-3 rounded px-3 py-2.5 text-[13px] focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent", active ? "border border-primary/25 bg-primary/10 font-semibold text-cyan" : "border border-transparent text-subtitle hover:bg-raised hover:text-white") }>
              <Icon size={17} /><span>{label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto hidden border-t border-border p-4 lg:block">
        <div className="flex items-center gap-2 rounded border border-border bg-surface px-3 py-2 font-mono text-[10px] text-subtitle">
          <span className={getDotClass(state)} />
          {getStatusText(state)}
        </div>
        <p className="mt-3 font-mono text-[9px] tracking-[0.08em] text-tertiary">SIH26167 · LIVE API CONNECTED</p>
        <p className="mt-2 text-[11px] leading-relaxed text-subtitle">Ask your satellite data anything.</p>
      </div>
    </aside>
  );
}
