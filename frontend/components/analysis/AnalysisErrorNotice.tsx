import type { ReactNode } from "react";
import type { FailureDetail } from "@/lib/workspace-contract";

/** Readable rendering of an /api/analyze failure. Never invents an answer. */
export function AnalysisErrorNotice({ detail, status, children }: { detail: FailureDetail; status?: number; children?: ReactNode }) {
  return (
    <div role="alert" className="rounded-xl border border-error/30 bg-error/10 p-4 text-sm leading-relaxed text-error">
      <p><strong>No fabricated answer.</strong>{status ? <span className="ml-2 font-mono text-[11px] opacity-80">HTTP {status}</span> : null}</p>
      {detail.kind === "message" && <p className="mt-1">{detail.message}</p>}
      {detail.kind === "provider" && (
        <div className="mt-2 space-y-1">
          <p>The {detail.capability ? <strong>{detail.capability}</strong> : "selected"} capability is not ready on this machine.</p>
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
            {detail.provider && <><dt className="opacity-75">Provider</dt><dd className="font-mono">{detail.provider}</dd></>}
            <dt className="opacity-75">Reason</dt><dd className="font-mono">{detail.reasonCode}</dd>
            {detail.detail && <><dt className="opacity-75">Detail</dt><dd>{detail.detail}</dd></>}
          </dl>
        </div>
      )}
      {detail.kind === "pair" && (
        <div className="mt-2 space-y-2">
          <p>This scene pair is not eligible{detail.workflow ? <> for <strong>{detail.workflow}</strong></> : null}.</p>
          {detail.failedChecks.length > 0 && (
            <ul className="list-disc space-y-1 pl-5 text-xs">
              {detail.failedChecks.map((check, index) => <li key={`${check.code}-${index}`}>{check.message} <span className="font-mono opacity-75">({check.code})</span></li>)}
            </ul>
          )}
          {detail.warnings.length > 0 && (
            <div className="text-xs text-warning"><p className="font-[600]">Warnings</p><ul className="list-disc pl-5">{detail.warnings.map((check, index) => <li key={`${check.code}-${index}`}>{check.message}</li>)}</ul></div>
          )}
          {detail.unknownMetadata.length > 0 && <p className="text-xs opacity-80">Unknown metadata: <span className="font-mono">{detail.unknownMetadata.join(", ")}</span></p>}
        </div>
      )}
      {children}
    </div>
  );
}
