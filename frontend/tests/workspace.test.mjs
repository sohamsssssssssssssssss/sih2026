import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { registerHooks } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { loadBindings, transformSync } from "next/dist/build/swc/index.js";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

// Reuse Next's installed TSX compiler and Node's test runner; no added dependency.
await loadBindings();
const root = fileURLToPath(new URL("../", import.meta.url));
registerHooks({
  resolve(specifier, context, next) {
    if (specifier.startsWith("@/")) specifier = pathToFileURL(root + specifier.slice(2)).href;
    if (specifier.startsWith("file:") || specifier.startsWith(".")) {
      const base = new URL(specifier, context.parentURL);
      for (const suffix of [".ts", ".tsx"]) {
        if (existsSync(fileURLToPath(base) + suffix)) return { url: base.href + suffix, shortCircuit: true };
      }
    }
    return next(specifier, context);
  },
  load(url, context, next) {
    if (/\.tsx?$/.test(url) && !url.includes("node_modules")) return {
      format: "module", shortCircuit: true,
      source: transformSync(readFileSync(new URL(url), "utf8"), { filename: fileURLToPath(url), jsc: { parser: { syntax: "typescript", tsx: url.endsWith(".tsx") }, target: "es2022", transform: { react: { runtime: "automatic" } } }, module: { type: "es6" } }).code,
    };
    return next(url, context);
  },
});

const { validateAnalysis, GOLDEN_SCENE, integrityState } = await import("../lib/workspace-contract.ts");
const { ExecutionBadge } = await import("../components/analysis/ExecutionBadge.tsx");
const { AnalysisResult } = await import("../components/analysis/AnalysisResult.tsx");
const { IntegrityBadge } = await import("../components/evidence/IntegrityBadge.tsx");
const { EvidencePanel } = await import("../components/evidence/EvidencePanel.tsx");
const { SceneMetadata } = await import("../components/imagery/SceneMetadata.tsx");
const { ImageryViewer, boxesToGeoJSON, frameExtent } = await import("../components/imagery/ImageryViewer.tsx");
const { analyzeGolden, getSceneImageUrl } = await import("../lib/api.ts");
const html = (component, props) => renderToStaticMarkup(React.createElement(component, props));
const artifact = JSON.parse(readFileSync(new URL("../../results/qwen2.5vl-3b__ladder__rescored__20260904.json", import.meta.url)));
const matches = artifact.results.filter(row => row.tile_id === GOLDEN_SCENE.id && row.question === GOLDEN_SCENE.question && row.gsd === GOLDEN_SCENE.gsd);
const fixture = () => ({
  answer: matches[0].prediction.answer, execution_mode: "cached_result", results_artifact: "results/test.json",
  model: { name: "qwen2.5vl-3b", version: "Qwen/Qwen2.5-VL-3B-Instruct" },
  trace: { model_name: "qwen2.5vl-3b", model_version: "Qwen/Qwen2.5-VL-3B-Instruct", params: { scene_id: GOLDEN_SCENE.id, execution_mode: "cached_result", results_artifact: "results/test.json" }, input_summary: { question: GOLDEN_SCENE.question, image_paths: [], n_images: 1 }, timestamp_iso: "2026-09-07T00:00:00Z", record_hash: "a".repeat(64), prev_hash: "" },
});

test("golden constants match a unique correct artifact result", () => { assert.equal(matches.length, 1); assert.equal(matches[0].correct, true); });
test("live and cached badges render exact labels", () => { assert.match(html(ExecutionBadge, { mode: "live" }), /LIVE INFERENCE/); assert.match(html(ExecutionBadge, { mode: "cached_result" }), /VERIFIED CACHED RESULT/); });
test("unknown modes never render a valid execution badge", () => { for (const mode of [undefined, null, "cached", "unknown"]) { const rendered = html(ExecutionBadge, { mode }); assert.match(rendered, /UNKNOWN EXECUTION MODE/); assert.doesNotMatch(rendered, /VERIFIED CACHED RESULT|LIVE INFERENCE/); } });
test("missing, unknown and inconsistent modes reject results", () => { for (const mode of [undefined, "cached", "live"]) { const result = fixture(); result.execution_mode = mode; assert.throws(() => validateAnalysis(result, GOLDEN_SCENE.question), /execution mode/); } });
test("cached and live structured results validate", () => { validateAnalysis(fixture(), GOLDEN_SCENE.question); const live = fixture(); live.execution_mode = live.trace.params.execution_mode = "live"; live.results_artifact = null; validateAnalysis(live, GOLDEN_SCENE.question); });
test("wrong question and incomplete evidence reject results", () => { assert.throws(() => validateAnalysis(fixture(), "different question"), /provenance/); const result = fixture(); delete result.trace; assert.throws(() => validateAnalysis(result, GOLDEN_SCENE.question), /incomplete/); });
test("result shows model and answer without placeholder confidence", () => { const rendered = html(AnalysisResult, { result: { ...fixture(), confidence: 1 } }); assert.match(rendered, /Yes/); assert.match(rendered, /Qwen2.5-VL-3B-Instruct/); assert.doesNotMatch(rendered, /confidence|100%/i); });
test("integrity badge never claims a verification that did not run", () => { assert.equal(integrityState("true"), "unknown"); assert.equal(integrityState(true), "verified"); assert.equal(integrityState(false), "failed"); const rendered = html(IntegrityBadge); assert.match(rendered, /Hash chained/); assert.doesNotMatch(rendered, /Chain verified/); });
test("evidence includes hashes", () => { const rendered = html(EvidencePanel, { trace: fixture().trace }); assert.match(rendered, /Record hash/); assert.match(rendered, /Previous hash/); assert.match(rendered, /raw execution evidence/); });
test("metadata separates source and unknown sensor/location", () => { const rendered = html(SceneMetadata); assert.match(rendered, /Dataset \/ source/); assert.match(rendered, /Sensor<\/dt><dd[^>]*>Unknown/); assert.match(rendered, /Location<\/dt><dd[^>]*>Unknown/); });
test("image viewer has no geographic position", () => { const rendered = html(ImageryViewer); assert.match(rendered, /not georeferenced/); assert.doesNotMatch(rendered, /72\.88|19\.08|Mumbai|maplibre/); });
test("scene image URL encodes the controlled scene identifier", () => { assert.equal(getSceneImageUrl("scene id/unsafe"), "http://localhost:8000/api/scenes/scene%20id%2Funsafe/image"); });
test("evidence boxes map into the arbitrary frame with the image aspect", () => {
  assert.deepEqual(frameExtent(200, 100), [0.06, 0.03]); assert.deepEqual(frameExtent(null, null), [0.06, 0.06]);
  const [feature] = boxesToGeoJSON([{ index: 1, kind: "bounding_box", label: "ship", confidence: 0.9, x0: 0, y0: 0, x1: 0.5, y1: 1 }], [0.06, 0.03]).features;
  assert.deepEqual(feature.geometry.coordinates[0][0], [-0.06, 0.03]); assert.deepEqual(feature.geometry.coordinates[0][2], [0, -0.03]);
});
test("validation binds results to the requested scenes", () => {
  const live = fixture(); live.execution_mode = live.trace.params.execution_mode = "live"; live.results_artifact = null;
  live.trace.params.scene_id = "scene_" + "a".repeat(32); live.trace.params.scene_id_2 = "scene_" + "b".repeat(32);
  validateAnalysis(live, GOLDEN_SCENE.question, { sceneId: "scene_" + "a".repeat(32), sceneId2: "scene_" + "b".repeat(32) });
  assert.throws(() => validateAnalysis(live, GOLDEN_SCENE.question, { sceneId: "scene_" + "a".repeat(32) }), /provenance/);
  assert.throws(() => validateAnalysis(live, GOLDEN_SCENE.question), /provenance/);
  live.evidence = "boxes"; assert.throws(() => validateAnalysis(live, GOLDEN_SCENE.question, { sceneId: "scene_" + "a".repeat(32), sceneId2: "scene_" + "b".repeat(32) }), /evidence/);
});
test("golden analysis is explicitly live, uses Unknown sensor and sanitizes server errors", async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async (_url, init) => { const body = JSON.parse(init.body); assert.equal(body.sensor, "Unknown"); assert.equal(body.execution_mode, "live"); return new Response(JSON.stringify({ detail: "Traceback CUDA /private/secrets" }), { status: 500 }); };
    await assert.rejects(analyzeGolden(GOLDEN_SCENE.question), error => /Analysis service unavailable/.test(error.message) && !/Traceback|CUDA|private/.test(error.message));
  } finally { globalThis.fetch = original; }
});
