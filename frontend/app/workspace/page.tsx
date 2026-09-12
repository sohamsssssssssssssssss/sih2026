import type { Metadata } from "next";
import { ImageryViewer } from "@/components/imagery/ImageryViewer";
import { SceneMetadata } from "@/components/imagery/SceneMetadata";
import { QueryPanel } from "@/components/analysis/QueryPanel";

export const metadata: Metadata = { title: "Workspace" };

export default function WorkspacePage() {
  return (
    <div>
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4"><div><p className="eyebrow text-cyan">Analysis workspace</p><h1 className="mt-1.5 text-2xl font-semibold tracking-tight text-white sm:text-3xl">Verified scene intelligence</h1><p className="mt-1 max-w-2xl text-sm text-subtitle">Ask the frozen Qwen vision-language model about a known LoveDA scene and inspect its execution provenance.</p></div><div className="flex gap-2 font-mono text-[9px]"><span className="rounded border border-success/30 bg-success/10 px-2 py-1.5 text-success">GOLDEN SCENE</span><span className="rounded border border-border bg-surface px-2 py-1.5 text-subtitle">0.3 M GSD</span></div></div>
      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_340px] 2xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="min-w-0">
          <ImageryViewer />
          <SceneMetadata />
        </div>
        <div className="min-w-0">
          <QueryPanel />
        </div>
      </div>
    </div>
  );
}
