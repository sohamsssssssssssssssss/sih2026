import { GOLDEN_SCENE } from "@/lib/workspace-contract";

export function SceneMetadata() {
  return (
    <div className="mt-4 p-4 bg-surface-elevated border border-border rounded-16">
      <p className="eyebrow">Scene</p><p className="mt-1 break-all font-mono text-sm text-subtitle">{GOLDEN_SCENE.id}</p>
      <dl className="mt-4 grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">{[["Dataset / source", GOLDEN_SCENE.source], ["GSD", `${GOLDEN_SCENE.gsd} m`], ["Sensor", "Unknown"], ["Location", "Unknown"], ["Acquisition date", "Unknown"]].map(([label, value]) => <div key={label}><dt className="text-tertiary">{label}</dt><dd className="mt-1 text-subtitle">{value}</dd></div>)}</dl>
    </div>
  );
}
