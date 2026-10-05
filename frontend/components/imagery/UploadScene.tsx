"use client";

import { useId, useRef, useState, type DragEvent, type FormEvent } from "react";
import { CheckCircle2, LoaderCircle, UploadCloud, X } from "lucide-react";
import { MAX_UPLOAD_BYTES, UploadError, uploadScene } from "@/lib/api";
import type { SceneModality, SceneUploadMetadata, SceneUploadResponse } from "@/lib/types";

const EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"];
const MIME_TYPES = new Set(["image/png", "image/jpeg", "image/tiff", "image/geotiff"]);
const MODALITIES: SceneModality[] = ["optical", "multispectral", "sar", "unknown"];

export type TimeZoneChoice = "utc" | "local";

/** Client-side guard only; the API re-validates every byte. */
export function validateUploadFile(file: File): string | null {
  const name = file.name.toLowerCase();
  const extensionOk = EXTENSIONS.some(extension => name.endsWith(extension));
  if (!extensionOk || (file.type && !MIME_TYPES.has(file.type))) {
    return `Unsupported file type${file.type ? ` (${file.type})` : ""}. Upload a PNG, JPEG, TIFF or GeoTIFF image.`;
  }
  if (file.size === 0) return "The selected file is empty.";
  if (file.size > MAX_UPLOAD_BYTES) return `The file is ${formatBytes(file.size)}; the upload limit is 20 MiB.`;
  return null;
}

export function formatBytes(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}

function pad(value: number) { return String(value).padStart(2, "0"); }

/**
 * `datetime-local` value → ISO 8601 with an explicit offset, as the API requires.
 * "local" uses the browser's offset at that instant (so DST is respected).
 */
export function toIsoWithZone(value: string, zone: TimeZoneChoice): string | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?$/.exec(value.trim());
  if (!match) return null;
  const [, year, month, day, hour, minute, second = "00"] = match;
  const base = `${year}-${month}-${day}T${hour}:${minute}:${second}`;
  if (zone === "utc") return `${base}Z`;
  const local = new Date(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second));
  if (Number.isNaN(local.getTime())) return null;
  const offset = -local.getTimezoneOffset();
  const sign = offset >= 0 ? "+" : "-";
  return `${base}${sign}${pad(Math.floor(Math.abs(offset) / 60))}:${pad(Math.abs(offset) % 60)}`;
}

export function UploadScene({ onUploaded }: { onUploaded?: (scene: SceneUploadResponse, metadata: SceneUploadMetadata) => void }) {
  const ids = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [modality, setModality] = useState<SceneModality | "">("");
  const [sensor, setSensor] = useState("");
  const [polarization, setPolarization] = useState("");
  const [acquired, setAcquired] = useState("");
  const [zone, setZone] = useState<TimeZoneChoice>("utc");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState<SceneUploadResponse | null>(null);

  const timestamp = acquired ? toIsoWithZone(acquired, zone) : null;

  function choose(next: File | null | undefined) {
    setError(null); setUploaded(null);
    if (!next) return;
    const problem = validateUploadFile(next);
    setFileError(problem);
    setFile(problem ? null : next);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    if (pending) return;
    choose(event.dataTransfer.files?.[0]);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file || pending) return;
    if (acquired && !timestamp) { setError("Acquisition time is not a valid date and time."); return; }
    const metadata: SceneUploadMetadata = {};
    if (modality) metadata.modality = modality;
    if (sensor.trim()) metadata.sensor = sensor.trim();
    if (timestamp) metadata.acquisition_timestamp = timestamp;
    if (polarization.trim()) metadata.polarization = polarization.trim();
    setPending(true); setError(null); setUploaded(null);
    try {
      const response = await uploadScene(file, metadata);
      setUploaded(response);
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
      onUploaded?.(response, metadata);
    } catch (reason) {
      setError(reason instanceof UploadError || reason instanceof Error ? reason.message : "Upload failed. The scene was not stored.");
    } finally {
      setPending(false);
    }
  }

  const fieldClass = "mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-ink outline-none focus:border-accent disabled:opacity-60";
  const labelClass = "block text-[10px] font-[600] uppercase tracking-[0.14em] text-midgray";

  return (
    <form onSubmit={submit} className="space-y-3" aria-label="Upload a scene">
      <div
        role="button"
        tabIndex={0}
        aria-label="Choose or drop an image to upload"
        aria-disabled={pending}
        onClick={() => !pending && inputRef.current?.click()}
        onKeyDown={event => { if (!pending && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); inputRef.current?.click(); } }}
        onDragOver={event => { event.preventDefault(); if (!pending) setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex w-full cursor-pointer flex-col items-center justify-center gap-1 rounded-lg border border-dashed px-4 py-4 text-center text-sm transition focus:outline-none focus:ring-2 focus:ring-cobalt/60 ${dragging ? "border-cobalt bg-cobalt/5 text-cobalt" : "border-border text-deepgray hover:bg-surface"} ${pending ? "cursor-not-allowed opacity-60" : ""}`}
      >
        <span className="flex items-center gap-2 font-[500]"><UploadCloud size={16} /> {file ? file.name : "Drop an image or browse"}</span>
        <span className="text-[11px] text-midgray">{file ? formatBytes(file.size) : "PNG, JPEG, TIFF or GeoTIFF · up to 20 MiB"}</span>
      </div>
      <input
        ref={inputRef}
        id={`${ids}-file`}
        data-testid="upload-input"
        type="file"
        className="sr-only"
        tabIndex={-1}
        accept=".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff"
        disabled={pending}
        onChange={event => choose(event.target.files?.[0])}
      />
      {fileError && <p role="alert" className="text-xs text-error">{fileError}</p>}

      <details className="rounded-lg border border-border px-3 py-2">
        <summary className="cursor-pointer text-xs font-[500] text-deepgray">Optional metadata</summary>
        <p className="mt-2 text-[11px] leading-5 text-midgray">Declared by you and stored as-is. Change detection needs two TIFF scenes with acquisition times; optical–SAR needs one optical and one SAR TIFF.</p>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <div>
            <label htmlFor={`${ids}-modality`} className={labelClass}>Modality</label>
            <select id={`${ids}-modality`} value={modality} disabled={pending} onChange={event => setModality(event.target.value as SceneModality | "")} className={fieldClass}>
              <option value="">Not declared</option>
              {MODALITIES.map(value => <option key={value} value={value}>{value === "sar" ? "SAR" : value[0].toUpperCase() + value.slice(1)}</option>)}
            </select>
          </div>
          <div>
            <label htmlFor={`${ids}-sensor`} className={labelClass}>Sensor</label>
            <input id={`${ids}-sensor`} value={sensor} maxLength={256} disabled={pending} onChange={event => setSensor(event.target.value)} placeholder="e.g. Sentinel-2" className={fieldClass} />
          </div>
          <div>
            <label htmlFor={`${ids}-acquired`} className={labelClass}>Acquisition time</label>
            <input id={`${ids}-acquired`} type="datetime-local" step={1} value={acquired} disabled={pending} onChange={event => setAcquired(event.target.value)} className={fieldClass} />
          </div>
          <div>
            <label htmlFor={`${ids}-zone`} className={labelClass}>Time zone</label>
            <select id={`${ids}-zone`} value={zone} disabled={pending} onChange={event => setZone(event.target.value as TimeZoneChoice)} className={fieldClass}>
              <option value="utc">UTC</option>
              <option value="local">This browser&apos;s local time</option>
            </select>
          </div>
          {modality === "sar" && (
            <div className="sm:col-span-2">
              <label htmlFor={`${ids}-polarization`} className={labelClass}>Polarization</label>
              <input id={`${ids}-polarization`} value={polarization} disabled={pending} onChange={event => setPolarization(event.target.value)} placeholder="e.g. VV,VH" className={fieldClass} />
            </div>
          )}
        </div>
        {acquired && <p className="mt-2 font-mono text-[10px] text-midgray" data-testid="timestamp-preview">{timestamp ? `Sent as ${timestamp}` : "Invalid date and time"}</p>}
      </details>

      <button type="submit" disabled={!file || pending} className="flex w-full items-center justify-center gap-2 rounded-pill border border-cobalt/40 px-4 py-2.5 text-sm font-[500] text-cobalt transition hover:bg-cobalt/5 focus:outline-none focus:ring-2 focus:ring-cobalt/60 disabled:cursor-not-allowed disabled:opacity-50">
        {pending ? <><LoaderCircle className="animate-spin" size={15} /> Uploading…</> : <><UploadCloud size={15} /> Upload scene</>}
      </button>
      {error && <div role="alert" className="flex items-start gap-2 rounded-lg border border-error/30 bg-error/10 p-3 text-xs leading-5 text-error"><X size={14} className="mt-0.5 shrink-0" /><span><strong>Upload rejected.</strong> {error}</span></div>}
      {uploaded && <p role="status" className="flex items-center gap-2 text-xs text-success"><CheckCircle2 size={14} /> Uploaded {uploaded.filename} ({uploaded.format} · {uploaded.width}×{uploaded.height}) and selected it.</p>}
    </form>
  );
}
