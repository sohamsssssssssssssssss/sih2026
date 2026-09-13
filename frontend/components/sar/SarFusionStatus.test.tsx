import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { SarFusionStatus } from "./SarFusionStatus";
import * as api from "@/lib/api";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("SarFusionStatus real-vs-prototype panel", () => {
  it("labels the human interpretation as CACHED REAL and fusion as PROTOTYPE", () => {
    vi.spyOn(api, "getCapabilities").mockRejectedValue(new api.ApiError(0, "down"));
    render(<SarFusionStatus />);
    expect(screen.getByText("CACHED REAL")).toBeTruthy();
    expect(screen.getByText("PROTOTYPE")).toBeTruthy();
    expect(screen.getByText("Human analyst interpretation")).toBeTruthy();
    expect(screen.getByText("Optical–SAR fusion")).toBeTruthy();
  });

  it("verifies the optical_sar prototype claim against /api/capabilities", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({
      capabilities: [
        { name: "single_image_vqa", available: true, provider: "qwen2.5vl-3b" },
        { name: "grounding", available: true, provider: "grounding-dino-swint" },
        { name: "change_vqa", available: false, provider: null },
        { name: "optical_sar", available: false, provider: null },
      ],
    });
    render(<SarFusionStatus />);
    await waitFor(() => {
      expect(screen.getByTestId("optical-sar-status").textContent).toContain("no registered provider");
    });
  });

  it("does not claim unavailability when the backend status cannot be reached", async () => {
    vi.spyOn(api, "getCapabilities").mockRejectedValue(new api.ApiError(0, "down"));
    render(<SarFusionStatus />);
    await waitFor(() => {
      expect(screen.getByTestId("optical-sar-status").textContent).toContain("could not be verified");
    });
  });
});
