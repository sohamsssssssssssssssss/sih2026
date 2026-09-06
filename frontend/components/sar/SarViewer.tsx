import { API_URL } from "@/lib/api";

export function SarViewer({ scene, available }: { scene: string; available: boolean }) {
  return (
    <div className="panel relative min-h-[420px] overflow-hidden bg-[#090c16]">
      {available ? <img src={`${API_URL}/api/sar/${scene}/image`} alt={`Processed Sentinel-1 SAR scene: ${scene}`} className="absolute inset-0 size-full object-cover" /> : <div className="absolute inset-0 grid place-items-center p-8 text-center"><div><div className="mx-auto mb-5 size-28 rounded-full border border-accent/20 bg-[radial-gradient(circle,#47d7dd22_1px,transparent_1px)] bg-[length:8px_8px]" /><p className="font-semibold text-slate-300">Processed render not present locally</p><p className="mt-2 max-w-sm text-sm text-slate-500">The committed analyst annotation remains available. No remote imagery is requested.</p></div></div>}
      <div className="pointer-events-none absolute inset-0 grid-overlay opacity-20" />
      <div className="absolute left-4 top-4 rounded-md border border-border bg-background/85 px-3 py-2 text-xs font-semibold backdrop-blur">Sentinel-1 · RTC false color</div>
    </div>
  );
}
