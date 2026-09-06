import { MapPin, ScanLine } from "lucide-react";

export function SceneMetadata() {
  return (
    <div className="panel mt-4 grid gap-4 p-4 sm:grid-cols-[1fr_auto_auto] sm:items-center">
      <div className="min-w-0"><p className="eyebrow">Scene</p><p className="mt-1 truncate font-mono text-xs text-slate-300">loveda_LoveDA_images_png_0_gsd0.3</p></div>
      <div className="flex items-center gap-2 text-sm text-slate-300"><MapPin size={15} className="text-accent" /> LoveDA</div>
      <div className="flex items-center gap-2 text-sm text-slate-300"><ScanLine size={15} className="text-accent" /> 0.3 m GSD</div>
    </div>
  );
}
