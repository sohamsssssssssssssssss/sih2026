import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { kosiChange } from "@/test/kosi-change-fixture";
vi.mock("@/lib/api", async importOriginal => ({ ...await importOriginal<typeof import("@/lib/api")>(), getKosiSarChange: vi.fn() }));
import { getKosiSarChange } from "@/lib/api";
import { KosiChangePanel } from "./KosiChangePanel";
beforeEach(() => { vi.mocked(getKosiSarChange).mockReset(); });
describe("Kosi preserved result", () => {
  it("shows loading and preserved-analysis labeling", () => {
    vi.mocked(getKosiSarChange).mockReturnValue(new Promise(() => {}));
    render(<KosiChangePanel />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading preserved Kosi result");
    expect(screen.getByText(/No live inference or satellite acquisition runs/)).toBeInTheDocument();
  });
  it("renders scientific labeling, metadata and download", async () => {
    vi.mocked(getKosiSarChange).mockResolvedValue(kosiChange);
    render(<KosiChangePanel />);
    expect(await screen.findByRole("link", { name: "Download change GeoTIFF" })).toHaveAttribute("href", "http://localhost:8000/api/sar/change/kosi-2024/raster");
    expect(screen.getByText(kosiChange.scientific_label)).toBeInTheDocument();
    expect(screen.getByText("September 24 represents pre-breach conditions, not a confirmed dry baseline.")).toBeInTheDocument();
    expect(screen.getByText("BEFORE: September 24, 2024")).toBeInTheDocument();
    expect(screen.getByText("AFTER: October 6, 2024")).toBeInTheDocument();
    expect(screen.getByText(/749 × 1649 · EPSG:32645/)).toBeInTheDocument();
    expect(screen.getByText(/1,235,101 \/ 1,235,101/)).toBeInTheDocument();
    expect(screen.getByRole("table")).toHaveTextContent("-0.8901");
    expect(screen.getByRole("table")).toHaveTextContent("-2.169");
    expect(screen.getByText(kosiChange.input_archives.before.product_id)).toBeInTheDocument();
    expect(screen.getByText(kosiChange.output_artifact_hashes["sar_change.tif"])).toBeInTheDocument();
    expect(screen.getByText(kosiChange.provenance_caveats[0])).toBeInTheDocument();
  });
  it.each(["Verified SAR change artifacts are unavailable.", "SAR change artifact integrity verification failed.", "SAR change artifact metadata is invalid."])("handles %s", async message => {
    vi.mocked(getKosiSarChange).mockRejectedValue(new Error(message));
    render(<KosiChangePanel />);
    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    expect(screen.queryByRole("link")).toBeNull();
  });
  it("labels every mobile statistic and preserves desktop values", async () => {
    vi.mocked(getKosiSarChange).mockResolvedValue(kosiChange);
    render(<KosiChangePanel />);
    const vv = await screen.findByRole("region", { name: "VV change statistics" });
    const vh = screen.getByRole("region", { name: "VH change statistics" });
    for (const [region, values] of [[vv, ["-0.8901", "3.733", "-24.15", "24.02"]], [vh, ["-2.169", "4.28", "-34.51", "17.58"]]] as const) {
      expect(region).toHaveTextContent("dB");
      for (const label of ["Mean", "Std. deviation", "Minimum", "Maximum"]) expect(within(region).getByText(label)).toBeInTheDocument();
      for (const value of values) {
        expect(within(region).getByText(value)).toBeInTheDocument();
        expect(screen.getByRole("table")).toHaveTextContent(value);
      }
    }
    expect(screen.getByRole("table")).toHaveAccessibleName("Backscatter change in dB · AFTER minus BEFORE in logarithmic power");
  });
});
