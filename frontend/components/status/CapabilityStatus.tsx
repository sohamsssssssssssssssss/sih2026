const available = ["Single-image VQA", "Resolution robustness", "Execution audit trace", "SAR analyst validation"];
const development = ["Grounding", "Bi-temporal Change-VQA", "Optical–SAR fusion", "RS fine-tuning"];

export function CapabilityStatus({ compact = false }: { compact?: boolean }) {
  return (
    <section className={compact ? "space-y-4" : "panel grid gap-5 p-5 md:grid-cols-2"} aria-label="Capability status">
      <div><p className="eyebrow text-success">Available now</p><p className="mt-2 text-sm leading-6 text-slate-300">{available.join(" · ")}</p></div>
      <div><p className="eyebrow text-warning">In development</p><p className="mt-2 text-sm leading-6 text-slate-400">{development.join(" · ")}</p></div>
    </section>
  );
}
