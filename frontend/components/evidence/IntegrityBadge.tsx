import { ShieldCheck } from "lucide-react";

export function IntegrityBadge() {
  return <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-success"><ShieldCheck size={14} /> Hash chained</span>;
}
