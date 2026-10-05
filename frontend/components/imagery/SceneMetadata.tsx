// Deliberately not a map pin: a dataset scene is not a place, and none of these scenes
// carry coordinates the UI could honestly show. See ImageryViewer for the same reasoning.
import { GOLDEN_OPTION, type SceneOption } from "@/lib/scenes";
import { formatTimestamp } from "@/lib/utils";

export function SceneMetadata({ scene = GOLDEN_OPTION }: { scene?: SceneOption }) {
  const rows: Array<[string, string]> = [
    ["Dataset / source", scene.source ?? "Unknown"],
    ["Sensor", scene.sensor ?? "Unknown"],
    ["GSD", scene.gsd != null ? `${scene.gsd} m` : "Unknown"],
    ["Location", "Unknown"],
    ["Acquisition", scene.acquisitionTime ? formatTimestamp(scene.acquisitionTime) : "Unknown"],
  ];
  if (scene.kind === "upload") {
    rows.push(
      ["Modality", scene.modality ?? "Unknown"],
      ["Format", scene.format && scene.width && scene.height ? `${scene.format} · ${scene.width}×${scene.height}` : scene.format ?? "Unknown"],
      ["Native raster", scene.hasNativeRaster ? (scene.georeferenced ? "Yes · georeferenced" : "Yes") : "No · preview only"],
    );
  }
  return (
    <div className="panel mt-4 p-4">
      <p className="eyebrow">Scene</p>
      <p className="mt-1 break-all font-mono text-xs text-deepgray">{scene.id}</p>
      <dl className="mt-4 grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
        {rows.map(([label, value]) => <div key={label} className="min-w-0"><dt className="text-midgray">{label}</dt><dd className="mt-1 break-words text-deepgray">{value}</dd></div>)}
      </dl>
    </div>
  );
}
