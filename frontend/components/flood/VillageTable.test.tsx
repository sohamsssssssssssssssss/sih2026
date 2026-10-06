import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { measuredVillages, noBoundaryVillages, noFloodedVillages } from "@/test/__fixtures__/flood";
import { VillageTable } from "./VillageTable";

const ATTRIBUTION = "Village boundaries: DataMeet (Census 2001), ODbL";

describe("VillageTable", () => {
  it("lists flooded villages in the returned order with headers, fractions and the scene-edge marker", () => {
    render(<VillageTable item={measuredVillages} />);
    expect(screen.getByText("2 of 5 villages in the scene gained open water")).toBeInTheDocument();
    const table = screen.getByRole("table");
    expect(within(table).getAllByRole("columnheader").map(header => header.textContent)).toEqual([
      "Village", "Sub-district", "District", "Flooded (ha)", "% of village flooded", "% observed", "Extends past scene edge",
    ]);
    const rows = within(table).getAllByRole("row").slice(1).map(row => within(row).getAllByRole("cell").map(cell => cell.textContent));
    expect(rows).toEqual([
      ["Test village A", "Test village A sub-district", "Test district", "252.9", "32.5%", "100%", "No"],
      ["Test village B", "Test village B sub-district", "Test district", "10", "2.5%", "unknown", "Yes"],
    ]);
    expect(screen.getByText(ATTRIBUTION)).toBeInTheDocument();
  });

  it("says villages are not named when no boundaries cover the scene", () => {
    render(<VillageTable item={noBoundaryVillages} />);
    expect(screen.getByText("No village boundaries cover this scene, so villages are not named.")).toBeInTheDocument();
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("reports zero flooded villages without an empty table", () => {
    render(<VillageTable item={noFloodedVillages} />);
    expect(screen.getByText("0 of 4 villages in the scene gained open water")).toBeInTheDocument();
    expect(screen.queryByRole("table")).toBeNull();
  });
});
