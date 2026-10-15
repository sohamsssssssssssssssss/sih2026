"use client";

import { useEffect, useState } from "react";
import { getKosiSarChange, getKosiSarRasterUrl } from "@/lib/api";
import type { KosiSarChange } from "@/lib/types";
import { formatValue, formatPercent } from "@/lib/evidence";
import { ResultEvidence } from "@/components/evidence/ResultEvidence";

export function KosiChangePanel() {
  const [result, setResult] = useState<KosiSarChange | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    getKosiSarChange().then(value => { if (active) setResult(value); })
      .catch(reason => { if (active) setError(reason instanceof Error ? reason.message : "Could not reach the SAR change API."); });
    return () => { active = false; };
  }, []);

  return (
    <section aria-labelledby="kosi-change-title" className="space-y-4">
      <div className="panel p-5 sm:p-6">
        <p className="eyebrow">Preserved satellite analysis · previously computed</p>
        <h2 id="kosi-change-title" className="mt-2 text-2xl font-[600] text-ink">Kosi 2024 — Sentinel-1 Change Analysis</h2>
        <p className="mt-3 text-sm font-[600] text-warning">Measured SAR backscatter change; flood interpretation unvalidated.</p>
        <p className="mt-2 text-sm text-midgray">September 24 represents pre-breach conditions, not a confirmed dry baseline.</p>
        <p className="mt-2 text-xs text-midgray">This view retrieves a preserved result. No live inference or satellite acquisition runs.</p>
        {!result && !error && <p role="status" className="mt-4 text-sm text-midgray">Loading preserved Kosi result…</p>}
        {error && <p role="alert" className="mt-4 text-sm text-error">
          {/integrity|metadata/i.test(error) ? "Artifact integrity or metadata verification failed" : "Preserved Kosi result unavailable"}: {error}
        </p>}
        {result && <>
          <div className="mt-5 flex flex-wrap gap-x-6 gap-y-2 text-sm text-ink">
            <p>BEFORE: September 24, 2024</p><p>AFTER: October 6, 2024</p>
          </div>
          <p className="mt-2 text-sm text-midgray">{result.grid.width} × {result.grid.height} · {result.grid.crs} · {result.grid.resolution_m} m pixels</p>
          <p className="mt-2 text-sm text-midgray">Valid-pixel coverage: {result.coverage.valid_pixels.toLocaleString("en-US")} / {result.coverage.total_pixels.toLocaleString("en-US")} ({formatPercent(result.coverage.valid_fraction)})</p>
          <div className="mt-5">
            <p id="kosi-statistics-label" className="mb-2 text-xs text-midgray">Backscatter change in dB · AFTER minus BEFORE in logarithmic power</p>
            <div className="space-y-3 sm:hidden" aria-labelledby="kosi-statistics-label">
              {(["VV", "VH"] as const).map(band => {
                const stats = result.statistics[`${band}_change_dB`];
                return <section key={band} aria-label={`${band} change statistics`} className="border-t border-border pt-3">
                  <h3 className="text-sm text-ink">{band} · dB</h3>
                  <dl className="mt-2 grid grid-cols-2 gap-3 text-sm">
                    {([['mean', 'Mean'], ['std', 'Std. deviation'], ['min', 'Minimum'], ['max', 'Maximum']] as const).map(([key, label]) => <div key={key} className="min-w-0">
                      <dt className="text-xs text-midgray">{label}</dt>
                      <dd className="mt-1 break-words font-mono text-ink">{formatValue(stats[key])}</dd>
                    </div>)}
                  </dl>
                </section>;
              })}
            </div>
            <table aria-labelledby="kosi-statistics-label" className="hidden w-full text-left text-sm sm:table">
              <thead><tr>{["Band", "Mean", "Std. deviation", "Minimum", "Maximum"].map(label => <th key={label} scope="col" className="py-2 pr-4 font-[500] text-midgray">{label}</th>)}</tr></thead>
              <tbody>{(["VV", "VH"] as const).map(band => {
                const stats = result.statistics[`${band}_change_dB`];
                return <tr key={band} className="border-t border-border"><th scope="row" className="py-2 pr-4 text-ink">{band}</th>{(["mean", "std", "min", "max"] as const).map(key => <td key={key} className="pr-4 font-mono text-ink">{formatValue(stats[key])}</td>)}</tr>;
              })}</tbody>
            </table>
          </div>
          <a href={getKosiSarRasterUrl(result.raster_download_url)} className="mt-5 inline-flex rounded-pill border border-border px-4 py-2 text-sm text-ink hover:bg-raised">Download change GeoTIFF</a>
          <details className="mt-5 border-t border-border pt-4">
            <summary className="cursor-pointer text-sm text-ink">Processing and provenance limitations</summary>
            <ul className="mt-3 list-disc space-y-2 pl-5 text-xs text-midgray">{result.provenance_caveats.map(caveat => <li key={caveat}>{caveat}</li>)}</ul>
          </details>
        </>}
      </div>
      {result && <ResultEvidence evidence={[
        { type: "sar_backscatter_change_provenance", artifact_id: result.artifact_id, acquisitions: result.input_archives },
        { type: "sar_backscatter_change_method", formula: result.formula, grid: result.grid },
        { type: "sar_backscatter_change_artifacts", hashes: result.output_artifact_hashes, recorded_code_fingerprints: result.code_fingerprints },
      ]} />}
    </section>
  );
}
