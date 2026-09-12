import Link from "next/link";

export function TopBar() {
  return (
    <header className="flex min-h-14 flex-wrap items-center justify-between gap-2 border-b border-border bg-surface-elevated/95 px-4 py-3 lg:h-14 lg:px-7">
      <div className="flex items-center gap-3"><span className="font-mono text-[9px] tracking-[0.16em] text-cyan">EARTH OBSERVATION</span><span className="h-3 w-px bg-border" /><p className="text-[11px] text-subtitle">Evidence-backed geospatial intelligence</p></div>
      <Link href="/system" className="rounded border border-border bg-surface px-3 py-1.5 font-mono text-[9px] tracking-[0.1em] text-cyan hover:border-primary/40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">SYSTEM / CAPABILITIES <span aria-hidden="true">→</span></Link>
    </header>
  );
}
