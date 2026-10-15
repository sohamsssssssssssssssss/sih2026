import type { KosiSarChange } from "@/lib/types";
export const kosiChange: KosiSarChange = {
  artifact_id: "kosi-2024-sar-change", scientific_result_type: "sar_backscatter_change",
  scientific_label: "Measured SAR backscatter change; flood interpretation unvalidated.", flood_interpretation_validated: false,
  statistics: { VV_change_dB: { count: 1235101, mean: -.890123, std: 3.73305, min: -24.14515, max: 24.01599 }, VH_change_dB: { count: 1235101, mean: -2.168876, std: 4.27999, min: -34.51265, max: 17.58076 } },
  formula: { vv_change_db: "10 * (log10(VV_after) - log10(VV_before))", vh_change_db: "10 * (log10(VH_after) - log10(VH_before))", units: "dB", intermediate_precision: "float64", output_dtype: "float32" },
  grid: { width: 749, height: 1649, epsg: 32645, crs: "EPSG:32645", resolution_m: 10, bounds: [432970, 2867120, 440460, 2883610] },
  input_archives: {
    before: { product_id: "S1A_IW_GRDH_1SDV_20240924T122146_20240924T122211_055807_06D1A1_686F_COG.SAFE", acquisition_date: "2024-09-24T12:21:46Z", sha256: "973b026d6f19eaa469647eaa23647ab89195512769c1fe23796e9ede59a76af5" },
    after: { product_id: "S1A_IW_GRDH_1SDV_20241006T122147_20241006T122212_055982_06D887_1753_COG.SAFE", acquisition_date: "2024-10-06T12:21:47Z", sha256: "643092f78815e0bd735cf76e067b5075bc7fceca97d34043ce1443d378084a8c" },
  },
  output_artifact_hashes: { "sar_change.tif": "7d63525d8851c695c9f1e5e72719b80aebd3e6d18581e875bc60ba5d28f6d3a4", "sar_change.json": "792492a47b18111d0aa92a6d613ec3ec0f60a6de557e1ddfec4c597a507d2747" },
  code_fingerprints: { sentinel1: "a".repeat(64), sar_water: "b".repeat(64), s1_offline_change: "c".repeat(64) },
  coverage: { valid_pixels: 1235101, total_pixels: 1235101, valid_fraction: 1 }, raster_download_url: "/api/sar/change/kosi-2024/raster",
  provenance_caveats: ["Processing settings are reconstructed, not independently verified historical request logs."],
};
