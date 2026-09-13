import type { Metadata } from "next";
import { CapabilityStatus } from "@/components/status/CapabilityStatus";
import { RuntimeStatus } from "@/components/status/RuntimeStatus";

export const metadata: Metadata = { title: "System" };

export default function SystemPage() {
  return <div>
    <div className="mb-6">
      <p className="eyebrow">System</p>
      <h1 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">Capability and runtime status</h1>
      <p className="muted mt-2">Reported facts only. Unknown and unreported values are never promoted to available.</p>
    </div>
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
      <div className="panel p-6"><CapabilityStatus compact /></div>
      <RuntimeStatus />
    </div>
  </div>;
}
