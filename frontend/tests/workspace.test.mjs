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
const { ImageryViewer, reconcileImageLoad } = await import("../components/imagery/ImageryViewer.tsx");
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
test("failed and unchecked integrity never render success", () => { assert.equal(integrityState("true"), "unknown"); for (const verified of [false, null, undefined]) { const rendered = html(IntegrityBadge, { verified }); assert.doesNotMatch(rendered, /text-success|Chain verified/); } assert.match(html(IntegrityBadge, { verified: false }), /Verification failed/); assert.match(html(IntegrityBadge, { verified: true }), /text-success/); });
test("evidence includes hashes and defaults to unchecked", () => { const rendered = html(EvidencePanel, { trace: fixture().trace }); assert.match(rendered, /Not checked/); assert.match(rendered, /Record hash/); assert.match(rendered, /Previous hash/); assert.match(rendered, /raw execution evidence/); });
test("metadata separates source and unknown sensor/location", () => { const rendered = html(SceneMetadata); assert.match(rendered, /Dataset \/ source/); assert.match(rendered, /Sensor<\/dt><dd[^>]*>Unknown/); assert.match(rendered, /Location<\/dt><dd[^>]*>Unknown/); });
test("image viewer has controls and no geographic position", () => { const rendered = html(ImageryViewer); assert.match(rendered, /Zoom in/); assert.match(rendered, /Zoom out/); assert.match(rendered, /not georeferenced/); assert.doesNotMatch(rendered, /72\.88|19\.08|Mumbai|maplibre/); });
test("scene image URL encodes the controlled scene identifier", () => { assert.equal(getSceneImageUrl("scene id/unsafe"), "http://localhost:8000/api/scenes/scene%20id%2Funsafe/image"); });
test("completed image requests leave loading state", () => { assert.equal(reconcileImageLoad("loading", true, 1024), "ready"); assert.equal(reconcileImageLoad("loading", true, 0), "failed"); assert.equal(reconcileImageLoad("loading", false, 0), "loading"); });
test("analysis request uses Unknown sensor and sanitizes server errors", async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async (_url, init) => { assert.equal(JSON.parse(init.body).sensor, "Unknown"); return new Response(JSON.stringify({ detail: "Traceback CUDA /private/secrets" }), { status: 500 }); };
    await assert.rejects(analyzeGolden(GOLDEN_SCENE.question), error => /Analysis service unavailable/.test(error.message) && !/Traceback|CUDA|private/.test(error.message));
  } finally { globalThis.fetch = original; }
});
