import { UploadCloud } from "lucide-react";

export function UploadScene() {
  return (
    <button disabled className="flex w-full cursor-not-allowed items-center justify-center gap-2 rounded-lg border border-dashed border-border px-4 py-3 text-sm text-slate-500" title="Upload support is a later checkpoint">
      <UploadCloud size={16} /> Upload pipeline · in development
    </button>
  );
}
