import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { abstainedStats, measuredStats } from "@/test/__fixtures__/flood";
import { WaterChangeSummary } from "./WaterChangeSummary";

describe("WaterChangeSummary", () => {
  it("shows new and receded hectares, the ±1 dB range, threshold and tile count when measured", () => {
    render(<WaterChangeSummary item={measuredStats} />);
    expect(screen.getByText("1,404.9 ha")).toBeInTheDocument();
    expect(screen.getByText("1,242.7–1,583.5 ha")).toBeInTheDocument();
    expect(screen.getByText("55 ha")).toBeInTheDocument();
    expect(screen.getByText("-13.2 dB")).toBeInTheDocument();
    expect(screen.getByText("192")).toBeInTheDocument();
    expect(screen.queryByText(/abstained/i)).toBeNull();
    expect(screen.queryByText(/confidence/i)).toBeNull();
  });

  it("shows an abstention with the reason code and no numbers", () => {
    const { container } = render(<WaterChangeSummary item={abstainedStats} />);
    expect(screen.getByText(/abstained/i)).toBeInTheDocument();
    expect(screen.getByText("NO_OPEN_WATER_MODE")).toBeInTheDocument();
    expect(container.textContent).not.toMatch(/\d\s*ha|dB/);
  });
});
