import type { ResolutionReport } from "@/lib/types";

export function ResolutionTable({ report }: { report: ResolutionReport }) {
  return (
    <div className="panel overflow-hidden">
      <div className="border-b border-border p-5"><p className="eyebrow">Rung detail</p></div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[700px] text-left text-sm">
          <thead className="bg-raised/60 text-[10px] uppercase tracking-[0.12em] text-slate-500"><tr><th className="px-5 py-3">GSD</th><th className="px-5 py-3">Status</th><th className="px-5 py-3">Open accuracy</th><th className="px-5 py-3">Aggregate</th><th className="px-5 py-3">Predicted yes</th><th className="px-5 py-3">n</th></tr></thead>
          <tbody className="divide-y divide-border">
            {Object.entries(report.per_rung).map(([gsd, metrics]) => {
              const flagged = report.degenerate_rungs.includes(gsd);
              return <tr key={gsd} className={flagged ? "bg-warning/[0.045]" : ""}><td className="px-5 py-4 font-mono text-white">{Number(gsd)} m</td><td className="px-5 py-4"><span className={`rounded-full border px-2.5 py-1 text-[10px] font-bold ${flagged ? "border-warning/30 bg-warning/10 text-warning" : "border-success/25 bg-success/10 text-success"}`}>{flagged ? "DEGENERATE" : "VALID"}</span></td><td className="px-5 py-4 font-semibold text-accent">{(metrics.open_accuracy * 100).toFixed(1)}%</td><td className="px-5 py-4 text-slate-300">{(metrics.accuracy * 100).toFixed(1)}%</td><td className="px-5 py-4 text-slate-300">{(metrics.pred_yes_rate_on_binary * 100).toFixed(1)}%</td><td className="px-5 py-4 text-slate-500">{metrics.n}</td></tr>;
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
