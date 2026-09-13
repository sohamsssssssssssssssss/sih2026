import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { SarViewer } from "./SarViewer";

afterEach(cleanup);

const SCENE = "mumbai-coastal";

describe("SarViewer fail-closed behavior", () => {
  it("shows no image and no substitute when the render asset is not committed", () => {
    render(<SarViewer scene={SCENE} available={false} />);
    expect(screen.queryByRole("img")).toBeNull();
    const status = screen.getByRole("status");
    expect(status.textContent).toContain("not committed to this checkout");
    expect(status.textContent).toContain("no substitute imagery is displayed");
    expect(status.textContent).toContain("process_scenes.py");
  });

  it("requests the real backend render endpoint when the asset is committed", () => {
    render(<SarViewer scene={SCENE} available={true} />);
    expect((screen.getByRole("img") as HTMLImageElement).getAttribute("src")).toContain(
      `/api/sar/${encodeURIComponent(SCENE)}/image`,
    );
  });

  it("reports loading until the render reports pixels", () => {
    render(<SarViewer scene={SCENE} available={true} />);
    expect(screen.getByRole("status").textContent).toContain("Loading processed SAR render");
    fireEvent.load(screen.getByRole("img"));
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("fails closed with no substitute when the render endpoint errors", () => {
    render(<SarViewer scene={SCENE} available={true} />);
    fireEvent.error(screen.getByRole("img"));
    const status = screen.getByRole("status");
    expect(status.textContent).toContain("unavailable");
    expect(status.textContent).toContain("No substitute imagery is shown");
    expect((screen.getByRole("img") as HTMLImageElement).className).toContain("invisible");
  });
});
