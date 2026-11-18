import type { Metadata } from "next";
import { AskWorkspace } from "@/components/analysis/AskWorkspace";

export const metadata: Metadata = { title: "Ask" };

export default function AskPage() {
  return (
    <div className="mx-auto max-w-6xl pb-10">
      <h1 className="text-[42px] sm:text-[54px]">Ask about a scene</h1>
      <p className="muted mt-3 max-w-2xl">Choose a scene or upload a GeoTIFF. Inspect the answer, the evidence, and the decisions behind it.</p>
      <AskWorkspace />
    </div>
  );
}
