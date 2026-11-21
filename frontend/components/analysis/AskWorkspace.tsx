"use client";

import { useEffect, useState, type FormEvent } from "react";
import {
  API_URL, askScene, fetchAskScene, getAskCapabilities, getAskHealth, getAskPreview, getAskTrace, uploadAskScene,
  type AskAnswer, type AskSceneRef, type FreshSceneInput, type ScenePreview,
} from "@/lib/api";

type Scene = { label: string; ref: AskSceneRef } | { label: string; fixture: string };
type Example = { label: string; scene: string; question: string; second?: string };

const SCENES: Record<string, Scene> = {
  punjab: { label: "Punjab · Sentinel-2", fixture: "punjab-s2.tif" },
  india: { label: "India flood chip · Sentinel-1", ref: { pack_id: "sen1floods11-hand-v1", patch_id: "India_900498" } },
  second: { label: "Second India chip · Sentinel-1", ref: { pack_id: "sen1floods11-hand-v1", patch_id: "India_956930" } },
  vienna: { label: "Vienna · WorldCover", ref: { pack_id: "worldcover", patch_id: "1800" } },
  mumbai: { label: "Cloudy Mumbai · Sentinel-2", fixture: "mumbai-s2-cloudy-v2.tif" },
};

const EXAMPLES: Example[] = [
  { label: "Punjab land cover", scene: "punjab", question: "What is the land cover?" },
  { label: "India flood", scene: "india", question: "Map flood water" },
  { label: "Two-scene change", scene: "india", second: "second", question: "What changed between these two dates?" },
  { label: "Outside India", scene: "vienna", question: "What is the land cover at 48.45058 N, 16.15819 E?" },
  { label: "Cloudy Mumbai", scene: "mumbai", question: "What is the land cover in this Mumbai scene?" },
];

const CAPABILITY_LABELS: Record<string, string> = {
  landcover: "Land cover", flood: "Flood water", change_vqa: "Change QA", single_image_vqa: "Qwen visual QA",
  grounding: "Grounding DINO", optical_sar: "Optical–SAR",
};
const OUTCOME = { answered: "Answer", low_confidence: "Low confidence", refused: "Refused" };
const field = "w-full rounded-xl border border-border bg-white px-3 py-2 text-sm focus:ring-2 focus:ring-cobalt";
const badge = "rounded-full border px-2.5 py-1 text-xs font-semibold";

type CapabilityData = Awaited<ReturnType<typeof getAskCapabilities>>;
type HealthData = Awaited<ReturnType<typeof getAskHealth>>;
type TraceData = Awaited<ReturnType<typeof getAskTrace>>;

function describeError(error: unknown) {
  return error instanceof Error ? error.message : "Could not complete the request.";
}

async function resolveScene(scene: Scene) {
  if ("ref" in scene) return scene.ref;
  const response = await fetch(`${API_URL}/scenes/demo/${encodeURIComponent(scene.fixture)}`);
  if (!response.ok) throw new Error("The demo GeoTIFF could not be loaded.");
  const file = new File([await response.blob()], scene.fixture, { type: "image/tiff" });
  return (await uploadAskScene(file)).upload_id;
}

export function AskCapabilities() {
  const [data, setData] = useState<{ catalog: CapabilityData; health: HealthData }>();
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let active = true;
    Promise.all([getAskCapabilities(), getAskHealth()]).then(([catalog, health]) => {
      if (active) setData({ catalog, health });
    }).catch(() => { if (active) setFailed(true); });
    return () => { active = false; };
  }, []);
  return (
    <aside aria-label="Capabilities" className="panel p-5">
      <h2 className="text-lg">Capabilities</h2>
      {failed && <p role="alert" className="mt-3 text-sm text-error">Could not load live capabilities.</p>}
      {!failed && !data && <p className="mt-3 text-sm">Checking the API…</p>}
      {data && (
        <ul className="mt-2 divide-y divide-border">
          {data.catalog.capabilities.flatMap((cap) => {
            const tool = data.catalog.tools.find((t) => t.name === cap.tool);
            const disabled = data.health.disabled_by_decision[cap.tool] ?? (tool?.status === "disabled" ? cap.reason : undefined);
            if (disabled || tool?.status === "disabled") return [];
            const hostLimit = data.health.unavailable_on_host[cap.tool];
            const fault = data.health.capability_checks[cap.capability];
            const state = hostLimit ? "Unavailable on this machine"
              : !cap.available || fault?.available === false ? "Unavailable"
              : tool?.status === "experimental_unvalidated" ? "Available" : "Supported · validated";
            const reason = hostLimit ?? (fault?.available === false ? fault.reason_code : cap.reason);
            return (
              <li key={cap.capability} className="py-3">
                <p className="flex flex-wrap items-center gap-2 text-sm font-semibold">
                  {CAPABILITY_LABELS[cap.capability] ?? cap.capability}
                  {tool?.status === "experimental_unvalidated" && <ExperimentalBadge />}
                </p>
                <p className="mt-0.5 text-xs font-medium text-deepgray">{state}</p>
                {reason && <p className="mt-1 text-xs leading-relaxed text-deepgray">{reason}</p>}
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}

const REFUSALS: Record<string, string> = {
  nodata: "Too much of the image is missing to answer reliably.",
  sensor_missing: "This question needs a different sensor.",
  bands_missing: "This image is missing bands needed to answer the question.",
  gsd_too_coarse: "The image resolution is too coarse to answer reliably.",
  tool_disabled: "This question cannot be answered with the tools available here.",
  tool_missing: "No available tool can answer this question.",
  tool_error: "The analysis failed before it could produce an answer.",
  tool_unvalidated: "This analysis has not been validated for the question.",
  checks_disagree: "The analysis checks disagreed, so there is no reliable answer.",
  out_of_distribution: "This image differs too much from the model’s training imagery.",
  second_scene_missing: "This question needs a second scene.",
  timeout: "The analysis ran out of time.",
};

function ExperimentalBadge() {
  return <span className={`${badge} border-amber-300 bg-amber-50 text-amber-900`}>Experimental</span>;
}

function Preview({ preview, fallback }: { preview: ScenePreview; fallback?: string }) {
  const { scene, provenance, preview_kind } = preview;
  return (
    <figure className="mt-4">
      {/* eslint-disable-next-line @next/next/no-img-element -- The API returns a small, display-only PNG data URI. */}
      <img src={preview.image} alt={`${scene.sensor} ${preview_kind}`} className="w-64 max-w-full rounded-lg border border-border" />
      <figcaption className="mt-2 text-xs leading-relaxed text-deepgray">
        {provenance
          ? <><strong>{provenance.label}</strong> · {provenance.source} · {provenance.collection} · {provenance.item_id} · <time dateTime={provenance.acquisition_datetime}>{provenance.acquisition_datetime}</time></>
          : <>{scene.source}{scene.acquired && ` · ${scene.acquired}`}</>}
        {scene.cloud_pct !== null && <> · {provenance?.cloud_pct_label ?? "Reported scene-level cloud cover"} {formatPct(scene.cloud_pct)}</>}
        {fallback && <p className="mt-1 font-medium">{fallback}</p>}
      </figcaption>
    </figure>
  );
}

function formatPct(value: number) {
  return value > 0 && value < 0.01 ? "<0.01%" : `${Number(value.toFixed(2))}%`;
}

function classNameFor(name: string) {
  return ({ built: "built areas", shrub_grass: "shrubs and grass" } as Record<string, string>)[name] ?? name.replaceAll("_", " ");
}

function AskResult({ answer, trace, traceError, preview, fallback, previewError }: {
  answer: AskAnswer; trace?: TraceData; traceError?: string; preview?: ScenePreview; fallback?: string; previewError?: string;
}) {
  const label = answer.evidence.find((e) => typeof e.summary.tool_status === "string")?.summary.tool_status;
  const sensors = [...new Set(answer.evidence.flatMap((e) => e.needs_sensor))];
  const warnings = [...new Set(answer.evidence.flatMap(({ summary }) =>
    typeof summary.warning === "string" ? [summary.warning] : []))];
  const cloud = answer.evidence.find((e) => e.summary.rule === "cloud_threshold")?.summary;
  const rawScores = answer.primary?.tool === "landcover" ? answer.primary.value.scores : undefined;
  const scores = typeof rawScores === "object" && rawScores !== null
    ? Object.entries(rawScores).filter((entry): entry is [string, number] => typeof entry[1] === "number")
        .sort((a, b) => b[1] - a[1])
    : [];
  const water = answer.primary?.value.water_fraction;
  const vqaLabel = answer.primary?.tool === "vqa_b13_mlx" ? answer.primary.value.label : undefined;
  const vqaRefusal = answer.evidence.some(({ summary }) => summary.policy === "vqa_unavailable");
  let summary = answer.text;
  if (answer.refusal_reason === "cloud") {
    const detail = typeof cloud?.observed === "number" && typeof cloud.threshold === "number"
      ? ` (scene-level cloud cover ${formatPct(cloud.observed)}, limit ${formatPct(cloud.threshold)})` : "";
    summary = `Too cloudy to answer reliably${detail}.`;
  } else if (answer.refusal_reason === "not_answerable" && !answer.evidence.some((e) => e.summary.detail)) {
    summary = "This question can’t be answered from the available satellite imagery.";
  } else if (answer.refusal_reason && !vqaRefusal) {
    summary = REFUSALS[answer.refusal_reason] ?? answer.text;
  } else if (scores.length > 0) {
    summary = `The model’s strongest land-cover signals were for ${scores.slice(0, 2).map(([name]) => classNameFor(name)).join(" and ")}.`;
  } else if (typeof water === "number") {
    summary = `The experimental radar mask marks ${formatPct(water * 100)} of the analysed window as possible open water.`;
  }
  return (
    <section aria-label="Result" className="panel min-w-0 p-5" aria-live="polite">
      <p className="text-lg font-semibold leading-relaxed">{summary}</p>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <h2 className="mr-1 text-sm font-medium text-deepgray">{OUTCOME[answer.status]}</h2>
        {label === "validated" && <span className={`${badge} border-green-300 bg-green-50 text-green-900`}>Validated</span>}
        {label === "experimental_unvalidated" && <ExperimentalBadge />}
      </div>
      {typeof vqaLabel === "string" && <p className="mt-2 text-sm text-deepgray">{vqaLabel}</p>}
      {sensors.length > 0 && <p className="mt-2 text-sm font-semibold">Try imagery from {sensors.map((sensor) => sensor === "S1" ? "Sentinel-1 (S1)" : sensor === "S2" ? "Sentinel-2 (S2)" : sensor).join(", ")}.</p>}
      {preview && <Preview preview={preview} fallback={fallback} />}
      {previewError && <p className="mt-2 text-sm text-deepgray">{previewError}</p>}
      {warnings.length > 0 && <p className="mt-2 text-xs text-deepgray">{warnings.map((w) => w === "cloud_unknown" ? "Scene-level cloud metadata unavailable" : w.replaceAll("_", " ")).join(" · ")}</p>}
      {scores.length > 0 && (
        <dl aria-label="Uncalibrated land-cover scores" className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-5">
          {scores.map(([name, value]) => (
            <div key={name} className="rounded-xl border border-border bg-white px-3 py-2">
              <dt className="text-xs capitalize text-deepgray">{classNameFor(name)}</dt>
              <dd><span className="text-lg font-semibold tabular-nums">{value.toFixed(3)}</span><span className="block text-xs text-deepgray">uncalibrated score</span></dd>
            </div>
          ))}
        </dl>
      )}
      <div className="mt-5 border-t border-border pt-3">
        {traceError ? <p role="alert" className="text-sm text-error">{traceError}</p> : !trace ? (
          <p className="text-sm">Loading trace…</p>
        ) : (
          <details>
            <summary className={`cursor-pointer text-sm font-medium ${trace.verified ? "text-green-800" : "text-error"}`}>
              {trace.verified ? "Chain verified ✓" : "✗ chain broken"}
            </summary>
            <p className="mt-2 break-all text-xs text-deepgray">Trace ID <code>{answer.trace_id}</code></p>
            {answer.refusal_reason && <p className="mt-1 text-xs text-deepgray">Reason code <code>{answer.refusal_reason}</code></p>}
            <p className="mt-2 text-xs text-deepgray">{trace.message}</p>
            <ol aria-label="Trace events" className="mt-2 max-h-64 space-y-2 overflow-y-auto">
              {trace.events.map((event, index) => (
                <li key={index} className="rounded-lg border border-border bg-white px-3 py-2 text-xs">
                  <p className="mb-1 font-semibold">{[event.phase, event.rule, event.outcome].filter((v) => typeof v === "string").join(" · ")}</p>
                  <pre className="whitespace-pre-wrap break-all text-deepgray">{JSON.stringify(event, null, 2)}</pre>
                </li>
              ))}
            </ol>
          </details>
        )}
      </div>
    </section>
  );
}

export function AskWorkspace() {
  const [scenes, setScenes] = useState(SCENES);
  const [scene, setScene] = useState("fresh");
  const [place, setPlace] = useState("");
  const [sensor, setSensor] = useState<"S1" | "S2">("S2");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [second, setSecond] = useState("");
  const [question, setQuestion] = useState(EXAMPLES[0].question);
  const [selectedExample, setSelectedExample] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [answer, setAnswer] = useState<AskAnswer>();
  const [trace, setTrace] = useState<TraceData>();
  const [traceError, setTraceError] = useState("");

  const [preview, setPreview] = useState<ScenePreview>();
  const [fallback, setFallback] = useState<string>();
  const [previewError, setPreviewError] = useState<string>();

  async function run(input: Pick<Example, "scene" | "question" | "second">) {
    if (busy || !input.question.trim()) return;
    setBusy(input.scene === "fresh" ? "Finding imagery…" : "Asking…");
    setError(""); setAnswer(undefined); setTrace(undefined); setTraceError("");
    setPreview(undefined); setFallback(undefined); setPreviewError(undefined);
    try {
      let result: AskAnswer;
      let previewTask: Promise<void> | undefined;
      if (input.scene === "fresh") {
        const coords = place.trim().match(/^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))$/);
        const location = coords ? { lat: Number(coords[1]), lon: Number(coords[2]) } : { place: place.trim() };
        const selection: FreshSceneInput = { ...location, sensor, start_date: startDate, end_date: endDate || undefined };
        const fresh = await fetchAskScene(selection);
        if ("refusal" in fresh) {
          result = fresh.refusal;
        } else {
          setPreview(fresh);
          if (fresh.fallback_reason) {
            setFallback(fresh.fallback_reason.startsWith("Live fetch timed out after ")
              ? fresh.fallback_reason
              : `${fresh.fallback_reason} Showing the ${fresh.backup_place} backup for this question.`);
          }
          setBusy("Asking…");
          const secondary = input.second ? await resolveScene(scenes[input.second]) : undefined;
          result = await askScene(input.question.trim(), fresh.scene_ref, secondary);
        }
      } else {
        const primary = await resolveScene(scenes[input.scene]);
        const secondary = input.second ? await resolveScene(scenes[input.second]) : undefined;
        result = await askScene(input.question.trim(), primary, secondary);
        previewTask = getAskPreview(primary).then(setPreview)
          .catch((err) => setPreviewError(`Preview unavailable. ${describeError(err)}`));
      }
      setAnswer(result);
      const traceTask = getAskTrace(result.trace_id).then(setTrace)
        .catch((err) => setTraceError(`Trace unavailable. ${describeError(err)}`));
      await Promise.all([traceTask, previewTask]);
    } catch (err) { setError(describeError(err)); }
    finally { setBusy(""); }
  }

  async function addUpload(file: File) {
    setError(""); setBusy("Uploading…");
    try {
      const upload = await uploadAskScene(file);
      setScenes((current) => ({ ...current, [upload.upload_id]: { label: `${file.name} · ${upload.scene.sensor}`, ref: upload.upload_id } }));
      setScene(upload.upload_id); setSelectedExample(""); setAnswer(undefined);
    } catch (err) { setError(describeError(err)); }
    finally { setBusy(""); }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSelectedExample("");
    void run({ scene, second, question });
  }

  return (
    <div className="mt-7 grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_300px]">
      <div className="min-w-0 space-y-4">
        <section aria-label="Demo examples" className="panel p-5">
          <h2 className="text-lg">Demo examples</h2>
          <div className="mt-3 flex flex-wrap gap-2">
            {EXAMPLES.map((example) => (
              <button key={example.label} type="button" aria-pressed={selectedExample === example.label} disabled={!!busy}
                className="rounded-full border border-border bg-white px-3 py-2 text-sm font-medium hover:border-cobalt focus-visible:ring-2 focus-visible:ring-cobalt aria-pressed:border-cobalt aria-pressed:bg-blue-50 disabled:opacity-60"
                onClick={() => {
                  setSelectedExample(example.label); setScene(example.scene); setSecond(example.second ?? ""); setQuestion(example.question);
                  void run(example);
                }}>{example.label}</button>
            ))}
          </div>
        </section>
        <form aria-label="Ask a question" onSubmit={submit} className="panel p-5">
          <fieldset disabled={!!busy} className="grid gap-3">
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="grid gap-1 text-sm font-medium">Scene
                <select className={field} value={scene} onChange={(e) => setScene(e.target.value)}>
                  <option value="fresh">Find a scene by place and date</option>
                  {Object.entries(scenes).map(([id, value]) => <option key={id} value={id}>{value.label}</option>)}
                </select>
              </label>
              <label className="grid gap-1 text-sm font-medium">Second scene (optional)
                <select className={field} value={second} onChange={(e) => setSecond(e.target.value)}>
                  <option value="">None</option>
                  {Object.entries(scenes).map(([id, value]) => <option key={id} value={id}>{value.label}</option>)}
                </select>
              </label>
            </div>
            {scene === "fresh" && <>
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="grid gap-1 text-sm font-medium">Place or latitude, longitude
                  <input required maxLength={200} className={field} placeholder="Ludhiana or 30.90, 75.86" value={place} onChange={(e) => setPlace(e.target.value)} />
                </label>
                <label className="grid gap-1 text-sm font-medium">Sensor
                  <select className={field} value={sensor} onChange={(e) => setSensor(e.target.value === "S1" ? "S1" : "S2")}>
                    <option value="S2">Sentinel-2 · optical</option>
                    <option value="S1">Sentinel-1 · radar</option>
                  </select>
                </label>
                <label className="grid gap-1 text-sm font-medium">Start date (UTC)
                  <input required type="date" className={field} value={startDate} onChange={(e) => setStartDate(e.target.value)} />
                </label>
                <label className="grid gap-1 text-sm font-medium">End date (optional)
                  <input type="date" min={startDate} className={field} value={endDate} onChange={(e) => setEndDate(e.target.value)} />
                </label>
              </div>
              <p className="text-xs text-deepgray">Place names use <a className="underline" href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>.</p>
            </>}
            <label className="grid gap-1 text-sm font-medium">Upload GeoTIFF
              <input type="file" accept=".tif,.tiff,image/tiff" className="text-sm file:mr-2 file:rounded-lg file:border file:border-border file:bg-white file:px-3 file:py-1.5"
                onChange={(e) => { const file = e.target.files?.[0]; e.target.value = ""; if (file) void addUpload(file); }} />
            </label>
            <label className="grid gap-1 text-sm font-medium">Question
              <textarea rows={2} required maxLength={2000} className={field} value={question} onChange={(e) => setQuestion(e.target.value)} />
            </label>
            <button type="submit" disabled={!question.trim()} className="w-fit rounded-full bg-cobalt px-5 py-2 text-sm font-semibold text-white focus-visible:ring-2 focus-visible:ring-cobalt focus-visible:ring-offset-2 disabled:opacity-60">Ask</button>
          </fieldset>
          {busy && <p role="status" className="mt-2 text-sm text-deepgray">{busy}</p>}
          {error && <p role="alert" className="mt-2 text-sm text-error">{error}</p>}
        </form>
        {answer && <AskResult key={answer.trace_id} answer={answer} trace={trace} traceError={traceError} preview={preview} fallback={fallback} previewError={previewError} />}
      </div>
      <AskCapabilities />
    </div>
  );
}
