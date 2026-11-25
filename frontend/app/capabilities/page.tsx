import type { Metadata } from "next";
import { CapabilitiesPanel, type Capability } from "@/components/status/CapabilitiesPanel";
import { API_URL } from "@/lib/api";

export const metadata: Metadata = { title: "Capabilities" };

export default async function CapabilitiesPage() {
  const apiBase = process.env.SATQUERY_INTERNAL_API_URL ?? (API_URL || "http://127.0.0.1:8000");
  const response = await fetch(`${apiBase}/capabilities`, { cache: "no-store" }).catch(() => null);
  const rows: Capability[] | null = response?.ok ? (await response.json()).evidence : null;
  return (
    <main>
      <h1 className="text-[42px] sm:text-[54px]">Capabilities</h1>
      {rows ? <CapabilitiesPanel rows={rows} /> : (
        <p role="status" className="mt-6">The local capabilities API is unavailable.</p>
      )}
    </main>
  );
}
