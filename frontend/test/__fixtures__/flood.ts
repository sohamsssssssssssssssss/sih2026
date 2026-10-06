import type { EvidenceItem } from "@/lib/types";

// Synthetic evidence in the shape POST /api/analyze returns for a Sentinel-1 flood pair.
// Coordinates are small made-up squares, not a real scene; names are placeholders.
const square = (x: number, y: number, size: number) => [[[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]];

export const measuredStats: EvidenceItem = {
  type: "water_change_statistics", status: "measured", reason_code: null, method: "split_based_otsu_vv_db",
  threshold_db: -13.2, bimodal_tiles: 192, water_t1_ha: 967.2, water_t2_ha: 2354.6, new_water_ha: 1404.9, receded_water_ha: 55.0,
  new_water_ha_sensitivity: { threshold_minus_1db: 1242.7, threshold_plus_1db: 1583.5 }, confidence: null,
};

export const abstainedStats: EvidenceItem = {
  type: "water_change_statistics", status: "abstained", reason_code: "NO_OPEN_WATER_MODE", method: "split_based_otsu_vv_db",
  threshold_db: null, bimodal_tiles: 0, water_t1_ha: null, water_t2_ha: null, new_water_ha: null, receded_water_ha: null,
  new_water_ha_sensitivity: null, confidence: null,
};

export const truncatedPolygons: EvidenceItem = {
  type: "water_change_polygons", format: "geojson", crs: "EPSG:4326", feature_count: 3, returned_feature_count: 2, truncated: true, ordering: "largest first",
  geojson: {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { change: "new_water", pixels: 200 }, geometry: { type: "Polygon", coordinates: square(1, 2, 0.5) } },
      { type: "Feature", properties: { change: "receded_water", pixels: 40 }, geometry: { type: "Polygon", coordinates: square(3, 1, 0.25) } },
    ],
  },
};
/** West-south and east-north corners of `truncatedPolygons`. */
export const truncatedPolygonBounds = [[1, 1], [3.25, 2.5]];

export const emptyPolygons: EvidenceItem = {
  type: "water_change_polygons", format: "geojson", crs: "EPSG:4326", feature_count: 0, returned_feature_count: 0, truncated: false, ordering: "largest first",
  geojson: { type: "FeatureCollection", features: [] },
};

const village = (name: string, flooded: number, area: number, observed: number | null, outside: boolean) => ({
  name, type: "Village", sub_district: `${name} sub-district`, district: "Test district", state: "Test state", census_2001_code: `code-${name}`,
  flooded_ha: flooded, area_in_scene_ha: area, flooded_fraction: flooded / area, observed_fraction: observed, partially_outside_scene: outside,
});

export const measuredVillages: EvidenceItem = {
  type: "village_flooding", boundary_source: "DataMeet Indian Village Boundaries, ODbL-1.0", boundary_vintage: "Census 2001",
  status: "measured", villages_in_scene: 5, flooded_village_count: 2, truncated: false,
  flooded_villages: [village("Test village A", 252.9, 777.8, 1, false), village("Test village B", 10, 400, null, true)],
};

export const noBoundaryVillages: EvidenceItem = {
  type: "village_flooding", boundary_source: "DataMeet Indian Village Boundaries, ODbL-1.0", boundary_vintage: "Census 2001",
  status: "no_boundaries", villages_in_scene: null, flooded_village_count: 0, truncated: false, flooded_villages: [],
};

export const noFloodedVillages: EvidenceItem = { ...measuredVillages, villages_in_scene: 4, flooded_village_count: 0, flooded_villages: [] };
