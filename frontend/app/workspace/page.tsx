import type { Metadata } from "next";
import { ImageryViewer } from "@/components/imagery/ImageryViewer";
import { SceneMetadata } from "@/components/imagery/SceneMetadata";
import { QueryPanel } from "@/components/analysis/QueryPanel";

export const metadata: Metadata = { title: "Workspace" };

export default function WorkspacePage() {
  return (
    <div>
      <div className="mb-6"><p className="eyebrow">Analysis workspace</p><h1 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">Verified scene intelligence</h1><p className="muted mt-2 max-w-2xl">Ask a frozen Qwen vision-language model about a known LoveDA scene and inspect the provenance behind every result.</p></div>
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.45fr)_minmax(360px,0.75fr)]">
        <div><ImageryViewer /><SceneMetadata /></div>
        <QueryPanel />
      </div>
    </div>
  );
}
