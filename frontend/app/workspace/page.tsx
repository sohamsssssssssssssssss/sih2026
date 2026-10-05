import type { Metadata } from "next";
import { Workspace } from "@/components/workspace/Workspace";

export const metadata: Metadata = { title: "Workspace" };

export default function WorkspacePage() {
  return (
    <div>
      <div className="mb-6"><p className="eyebrow">Analysis workspace</p><h1 className="mt-2 text-[40px] font-[600] tracking-[-0.02em] text-ink sm:text-[56px]">Verified scene intelligence</h1><p className="muted mt-2 max-w-2xl">Pick a curated scene or upload your own, ask a question, and inspect the provenance and evidence behind every result. Pair two uploaded scenes for change detection or optical–SAR comparison.</p></div>
      <Workspace />
    </div>
  );
}
