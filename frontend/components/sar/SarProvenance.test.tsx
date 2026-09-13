import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { SarProvenance } from "./SarProvenance";
import type { SarReport } from "@/lib/types";

afterEach(cleanup);

const report = (overrides: Partial<SarReport> = {}): SarReport => ({
  scene: "mumbai-coastal",
  title: "Mumbai coastal",
  human_validation: true,
  data_source: "real_sar_grd_rtc",
  sensor: "Sentinel-1 C-band SAR, IW GRD, dual-polarization VV/VH",
  location: "19.05, 72.85",
  latitude: 19.05,
  longitude: 72.85,
  acquisition_date: null,
  processing_job_id: "71cf874e-4303-4e1f-9ab7-74037b1956c9",
  processing_chain: "Sentinel-1 IW GRD (VV+VH) -> ASF HyP3 RTC gamma-0",
  render_available: false,
  fusion_capability: "prototype_analyst_validation_not_ai_model_output",
  summaries: { water: "w", built_up: "b", vegetation: "v", terrain: "t" },
  annotation: "annotation",
  ...overrides,
});

describe("SarProvenance truthful display", () => {
  it("shows recorded provenance facts without embellishment", () => {
    render(<SarProvenance report={report()} />);
    expect(screen.getByText("19.05, 72.85")).toBeTruthy();
    expect(screen.getByText("71cf874e-4303-4e1f-9ab7-74037b1956c9")).toBeTruthy();
    expect(screen.getByText("Sentinel-1 C-band SAR, IW GRD, dual-polarization VV/VH")).toBeTruthy();
    expect(screen.getByText("Sentinel-1 IW GRD (VV+VH) -> ASF HyP3 RTC gamma-0")).toBeTruthy();
  });

  it("labels missing acquisition date as unknown instead of inventing values", () => {
    render(<SarProvenance report={report()} />);
    expect(screen.getByText("UNKNOWN")).toBeTruthy();
    expect(screen.getByText("Not committed — viewer fails closed")).toBeTruthy();
  });

  it("marks an unrecorded processing job as not recorded", () => {
    render(<SarProvenance report={report({ processing_job_id: null })} />);
    expect(screen.getByText("Not recorded in this repository")).toBeTruthy();
  });

  it("marks unrecorded locations as UNKNOWN for scenes without provenance", () => {
    render(
      <SarProvenance
        report={report({
          scene: "flat-inland-plain",
          location: "UNKNOWN",
          latitude: null,
          longitude: null,
          processing_job_id: null,
        })}
      />,
    );
    const unknowns = screen.getAllByText("UNKNOWN");
    expect(unknowns.length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByText("19.05, 72.85")).toBeNull();
  });
});
