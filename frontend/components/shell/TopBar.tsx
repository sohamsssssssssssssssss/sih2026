import { ShieldCheck } from "lucide-react";

export function TopBar() {
  return (
    <header className="hidden h-16 items-center justify-between border-b border-border bg-background/80 px-8 backdrop-blur lg:flex">
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Evidence-backed geospatial intelligence</p>
      <div className="flex items-center gap-2 rounded-full border border-success/25 bg-success/10 px-3 py-1.5 text-xs font-bold text-success"><ShieldCheck size={14} /> AUDITABLE</div>
    </header>
  );
}
