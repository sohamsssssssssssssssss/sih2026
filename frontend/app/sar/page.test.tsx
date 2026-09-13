import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { fireEvent } from "@testing-library/react";
import SarPage from "./page";
import * as api from "@/lib/api";
import type { SarReport } from "@/lib/types";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const report = (scene: string, title: string): SarReport => ({
  scene,
  title,
  human_validation: true,
  data_source: "real_sar_grd_rtc",
  sensor: "Sentinel-1 C-band SAR, IW GRD, dual-polarization VV/VH",
  location: "UNKNOWN",
  latitude: null,
  longitude: null,
  acquisition_date: null,
  processing_job_id: null,
  processing_chain: "Sentinel-1 IW GRD (VV+VH) -> ASF HyP3 RTC gamma-0",
  render_available: false,
  fusion_capability: "prototype_analyst_validation_not_ai_model_output",
  summaries: { water: "w", built_up: "b", vegetation: "v", terrain: "t" },
  annotation: "verbatim analyst text",
});

describe("/sar Optical-SAR prototype page", () => {
  it("shows the human-validation banner and provenance after a successful fetch", async () => {
    vi.spyOn(api, "getSar").mockResolvedValue(report("mumbai-coastal", "Mumbai coastal"));
    vi.spyOn(api, "getCapabilities").mockRejectedValue(new api.ApiError(0, "down"));
    render(<SarPage />);
    expect(screen.getByText("HUMAN SAR VALIDATION — NOT AI MODEL OUTPUT")).toBeTruthy();
    await waitFor(() => expect(screen.getByText("Mumbai coastal")).toBeTruthy());
    expect(screen.getByText("Scene provenance")).toBeTruthy();
    expect(screen.getByText("What is real here")).toBeTruthy();
  });

  it("fetches each annotated scene from the real endpoint when selected", async () => {
    const getSar = vi.spyOn(api, "getSar").mockImplementation((scene) =>
      Promise.resolve(report(scene ?? "mumbai-coastal", scene ?? "Mumbai coastal")),
    );
    vi.spyOn(api, "getCapabilities").mockRejectedValue(new api.ApiError(0, "down"));
    render(<SarPage />);
    await waitFor(() => expect(getSar).toHaveBeenCalledWith("mumbai-coastal"));
    fireEvent.click(screen.getByRole("button", { name: "Konkan coast" }));
    await waitFor(() => expect(getSar).toHaveBeenCalledWith("konkan-coast"));
  });

  it("fails closed with no substitute annotation when the backend cannot serve a scene", async () => {
    vi.spyOn(api, "getSar").mockRejectedValue(new api.ApiError(404, "No analyst annotation exists for SAR scene 'x'."));
    vi.spyOn(api, "getCapabilities").mockRejectedValue(new api.ApiError(0, "down"));
    render(<SarPage />);
    await waitFor(() => {
      expect(screen.getByRole("alert").textContent).toContain("fails closed");
    });
    expect(screen.queryByText("Scene provenance")).toBeNull();
    expect(screen.queryByText("verbatim analyst text")).toBeNull();
  });
});
