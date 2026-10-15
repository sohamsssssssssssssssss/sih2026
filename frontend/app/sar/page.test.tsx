import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import type { SarReport } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  API_URL: "http://localhost:8000",
  getSar: vi.fn(),
  getKosiSarChange: vi.fn().mockRejectedValue(new Error("Preserved artifacts unavailable")),
  getKosiSarRasterUrl: vi.fn(),
}));

import { getSar } from "@/lib/api";
import SarPage from "./page";

const mockedGetSar = vi.mocked(getSar);

const baseReport: SarReport = {
  scene: "mumbai-coastal",
  title: "Mumbai coastal interpretation",
  human_validation: true,
  render_available: false,
  summaries: {
    water: "ANALYST_WATER_NOTE",
    built_up: "ANALYST_BUILTUP_NOTE",
    vegetation: "ANALYST_VEGETATION_NOTE",
    terrain: "ANALYST_TERRAIN_NOTE",
  },
  annotation: "FULL_ANALYST_ANNOTATION_TEXT",
};

beforeEach(() => {
  mockedGetSar.mockReset();
});

describe("SAR page", () => {
  it("shows the reference visualization and preserves analyst interpretation when no render is available", async () => {
    mockedGetSar.mockResolvedValue({ ...baseReport, render_available: false });
    render(<SarPage />);

    // Reference visualization + explicit truthfulness disclosure.
    expect(await screen.findByText(/Sentinel-1 · SAR Interpretation Guide/i)).toBeInTheDocument();
    expect(screen.getByText(/Reference visualization — not satellite imagery/i)).toBeInTheDocument();

    // Human-validation / not-AI-output messaging remains intact.
    expect(screen.getByText(/Human SAR validation — not AI model output/i)).toBeInTheDocument();

    // Committed analyst interpretation is still present (summaries + full annotation).
    expect(screen.getByText("ANALYST_WATER_NOTE")).toBeInTheDocument();
    expect(screen.getByText("ANALYST_BUILTUP_NOTE")).toBeInTheDocument();
    expect(screen.getByText("FULL_ANALYST_ANNOTATION_TEXT")).toBeInTheDocument();

    // No genuine-image request on the fallback path.
    expect(document.querySelector("img")).toBeNull();
  });

  it("renders the genuine Sentinel-1 image when a render is available", async () => {
    mockedGetSar.mockResolvedValue({ ...baseReport, render_available: true });
    render(<SarPage />);

    const img = await screen.findByRole("img");
    expect(img.getAttribute("src")).toContain("/api/sar/mumbai-coastal/image");
    // Human-validation messaging is still present alongside the real render.
    expect(screen.getByText(/Human SAR validation — not AI model output/i)).toBeInTheDocument();
    // The reference guide must not appear when the genuine render exists.
    expect(screen.queryByText(/SAR Interpretation Guide/i)).toBeNull();
  });
});
