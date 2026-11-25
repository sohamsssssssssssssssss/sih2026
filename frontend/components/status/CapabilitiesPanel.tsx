const labels = {
  live: "Live",
  experimental: "Experimental",
  off: "Off",
  "in-progress": "In progress",
};

const colors = {
  live: "bg-emerald-100 text-emerald-900",
  experimental: "bg-amber-100 text-amber-900",
  off: "bg-slate-200 text-slate-800",
  "in-progress": "bg-sky-100 text-sky-900",
};

export type Capability = {
  name: string;
  status: keyof typeof labels;
  why: string;
  evidence_path: string;
  commit: string;
};

export function CapabilitiesPanel({ rows }: { rows: Capability[] }) {
  return (
    <ul aria-label="Demo capabilities" className="mt-6 divide-y rounded-lg border">
      {rows.map(({ name, status, why, evidence_path, commit }) => (
        <li key={name} className="grid items-start gap-2 px-5 py-4 md:grid-cols-[15rem_1fr]">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="font-semibold">{name}</h2>
            <span className={`rounded-full px-2 py-1 text-xs font-semibold ${colors[status]}`}>{labels[status]}</span>
          </div>
          <div>
            <p>{why}</p>
            <a className="break-all text-sm underline underline-offset-2"
               href={`https://github.com/sohamsssssssssssssssss/sih2026/blob/${commit}/${evidence_path}`}>
              {evidence_path} · {commit.slice(0, 7)}
            </a>
          </div>
        </li>
      ))}
    </ul>
  );
}
