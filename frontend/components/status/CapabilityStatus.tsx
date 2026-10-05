const available = ["Single-image VQA", "Scene upload (PNG · JPEG · GeoTIFF)", "Resolution robustness", "Execution audit trace", "SAR analyst validation"];
// Implemented with a registered provider, but they only run where that provider is
// ready (weights, GPU, native rasters). The workspace reports the exact reason when not.
const providerGated = ["Grounding (Grounding DINO)", "Bi-temporal change · deterministic baseline", "Optical–SAR · deterministic baseline"];
const development = ["RS fine-tuning"];

export function CapabilityStatus({ compact = false }: { compact?: boolean }) {
  return (
    <section className={compact ? "space-y-4" : "panel grid gap-5 p-5 md:grid-cols-3"} aria-label="Capability status">
      <div><p className="eyebrow text-success">Available now</p><p className="mt-2 text-sm leading-6 text-deepgray">{available.join(" · ")}</p></div>
      <div><p className="eyebrow text-deepgray">Requires a ready local provider</p><p className="mt-2 text-sm leading-6 text-deepgray">{providerGated.join(" · ")}</p></div>
      <div><p className="eyebrow text-warning">In development</p><p className="mt-2 text-sm leading-6 text-midgray">{development.join(" · ")}</p></div>
    </section>
  );
}
