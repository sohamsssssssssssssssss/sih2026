import type { Metadata } from "next";
import { Cpu, Database, LockKeyhole, WifiOff } from "lucide-react";
import { CapabilityStatus } from "@/components/status/CapabilityStatus";
import { RuntimeStatus } from "@/components/status/RuntimeStatus";

export const metadata: Metadata = { title: "System" };

export default function SystemPage() {
  return (
    <div><div className="mb-6"><p className="eyebrow">System</p><h1 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">Capability and runtime status</h1><p className="muted mt-2">A small, explicit view of what is operational today and what is not.</p></div><div className="grid gap-5 lg:grid-cols-[1fr_360px]"><div className="panel p-6"><CapabilityStatus compact /></div><div className="space-y-5"><RuntimeStatus /><div className="panel divide-y divide-border p-5">{[{ icon: WifiOff, label: "Network", value: "Not required for golden path" }, { icon: Database, label: "Artifact", value: "Committed and local" }, { icon: Cpu, label: "Live model", value: "CUDA + cached weights required" }, { icon: LockKeyhole, label: "Trace", value: "SHA-256 hash chain" }].map(({ icon: Icon, label, value }) => <div key={label} className="flex items-center gap-3 py-3 first:pt-0 last:pb-0"><Icon size={16} className="text-accent" /><div><p className="text-xs font-semibold text-slate-300">{label}</p><p className="text-[11px] text-slate-500">{value}</p></div></div>)}</div></div></div></div>
  );
}
