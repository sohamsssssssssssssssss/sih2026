import { Building2, Leaf, Mountain, Waves } from "lucide-react";
import type { SarReport } from "@/lib/types";

const cards = [
  { key: "water" as const, label: "Water", icon: Waves },
  { key: "built_up" as const, label: "Built-up", icon: Building2 },
  { key: "vegetation" as const, label: "Vegetation", icon: Leaf },
  { key: "terrain" as const, label: "Terrain", icon: Mountain },
];

export function SarInterpretation({ report }: { report: SarReport }) {
  return (
    <div className="space-y-5">
      <div className="grid gap-3 sm:grid-cols-2">
        {cards.map(({ key, label, icon: Icon }) => <article key={key} className="panel p-4"><div className="flex items-center gap-2 text-accent"><Icon size={16} /><h2 className="text-xs font-bold uppercase tracking-wider">{label}</h2></div><p className="mt-3 line-clamp-6 text-sm leading-6 text-slate-300">{report.summaries[key]}</p></article>)}
      </div>
      <details className="panel group p-5"><summary className="cursor-pointer list-none text-sm font-semibold text-accent">View full analyst annotation <span className="ml-1 group-open:hidden">+</span><span className="ml-1 hidden group-open:inline">−</span></summary><div className="prose-annotation mt-5 whitespace-pre-wrap border-t border-border pt-4 text-sm leading-6 text-slate-300">{report.annotation}</div></details>
    </div>
  );
}
