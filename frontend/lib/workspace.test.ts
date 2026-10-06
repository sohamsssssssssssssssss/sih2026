import { describe, expect, it } from "vitest";
import { evidenceRows, overlayBoxes } from "./evidence";
import { toSceneOptions } from "./scenes";
import { GOLDEN_SCENE, isGoldenRequest, parseFailureDetail } from "./workspace-contract";
import { measuredVillages, truncatedPolygons } from "@/test/__fixtures__/flood";

describe("scene options", () => {
  it("pins the golden scene, keeps visible curated scenes and treats missing uploads as empty", () => {
    const options = toSceneOptions({
      version: "1.0",
      scenes: [
        { id: "loveda-golden-vqa", title: "LoveDA golden", capability: "single_image_vqa", result_state: "cached_real", catalog_visible: false, available: true, unavailable_reason: null, question: GOLDEN_SCENE.question, source: { dataset: "LoveDA", source_id: GOLDEN_SCENE.id } },
        { id: "dior-rsvg-07272", title: "DIOR-RSVG yellow ship", capability: "grounding", result_state: "real_live", catalog_visible: true, available: false, unavailable_reason: "asset missing", question: "Locate a yellow ship.", source: { dataset: "DIOR-RSVG", source_id: "JPEGImages/07272.jpg" } },
      ],
      uploads: [],
    });
    expect(options.map(option => [option.id, option.kind, option.available])).toEqual([
      [GOLDEN_SCENE.id, "golden", true],
      ["dior-rsvg-07272", "curated", false],
    ]);
    expect(options[1].suggestedQuestion).toBe("Locate a yellow ship.");
  });
});

describe("golden replay eligibility", () => {
  it("matches only the pinned scene and question, single-scene", () => {
    expect(isGoldenRequest(GOLDEN_SCENE.id, `  ${GOLDEN_SCENE.question} `)).toBe(true);
    expect(isGoldenRequest(GOLDEN_SCENE.id, "Is there a road?")).toBe(false);
    expect(isGoldenRequest("loveda-golden-vqa", GOLDEN_SCENE.question)).toBe(false);
    expect(isGoldenRequest(GOLDEN_SCENE.id, GOLDEN_SCENE.question, "scene_2")).toBe(false);
  });
});

describe("failure details", () => {
  it("parses provider and pair objects and sanitizes 5xx strings", () => {
    expect(parseFailureDetail(503, { detail: { capability: "change_vqa", provider: "change-baseline", reason_code: "RASTERIO_MISSING", detail: "rasterio is not installed." } }))
      .toEqual({ kind: "provider", capability: "change_vqa", provider: "change-baseline", reasonCode: "RASTERIO_MISSING", detail: "rasterio is not installed." });
    expect(parseFailureDetail(422, { detail: { requested_workflow: "optical_sar", failed_checks: [{ code: "modality_mismatch", message: "Exactly one optical and one SAR scene are required." }] } }))
      .toMatchObject({ kind: "pair", workflow: "optical_sar", failedChecks: [{ code: "modality_mismatch", message: "Exactly one optical and one SAR scene are required." }] });
    expect(parseFailureDetail(404, { detail: "Local scene pixels are unavailable." })).toEqual({ kind: "message", message: "Local scene pixels are unavailable." });
    expect(parseFailureDetail(500, { detail: "Traceback" })).toEqual({ kind: "message", message: expect.not.stringContaining("Traceback") });
  });
});

describe("evidence helpers", () => {
  it("keeps only well-formed normalized boxes", () => {
    const boxes = overlayBoxes([
      { type: "bounding_box", label: "ship", coordinates: [0.1, 0.2, 0.3, 0.4], coordinate_space: "normalized_xyxy", confidence: 0.9 },
      { type: "change_extent", coordinates: [0, 0, 1, 1], coordinate_space: "normalized_xyxy" },
      { type: "bounding_box", coordinates: [0.1, 0.2, 1.3, 0.4], coordinate_space: "normalized_xyxy" },
      { type: "bounding_box", coordinates: [10, 20, 30, 40], coordinate_space: "pixel_xyxy" },
      { type: "optical_statistics", valid_pixels: 10 },
    ]);
    expect(boxes.map(box => [box.index, box.label])).toEqual([[1, "ship"], [2, "Change extent"]]);
  });

  it("flattens nested evidence into readable rows", () => {
    expect(evidenceRows({ type: "sar_statistics", valid_fraction: 0.123456, vv_db: { mean: -12.5 }, bands: ["VV", "VH"] })).toEqual([
      ["Valid fraction", "0.1235"],
      ["Vv db › Mean", "-12.5"],
      ["Bands", "VV, VH"],
    ]);
  });

  it("summarizes polygon GeoJSON and village lists by count instead of dumping them", () => {
    const polygonRows = evidenceRows(truncatedPolygons);
    expect(polygonRows).toContainEqual(["Geojson", "2 features (drawn on the flood map)"]);
    expect(polygonRows).toContainEqual(["Feature count", "3"]);
    expect(JSON.stringify(polygonRows)).not.toMatch(/FeatureCollection|coordinates|new_water/);

    const villageRows = evidenceRows(measuredVillages);
    expect(villageRows).toContainEqual(["Flooded villages", "2 villages (listed in the village table)"]);
    expect(villageRows).toContainEqual(["Boundary vintage", "Census 2001"]);
    expect(JSON.stringify(villageRows)).not.toMatch(/Test village/);
  });
});
