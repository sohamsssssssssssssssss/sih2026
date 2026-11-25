import { render, screen, within } from "@testing-library/react";
import { expect, test } from "vitest";
import { CapabilitiesPanel } from "./CapabilitiesPanel";

test("shows a human-readable status and commit-pinned evidence", () => {
  const commit = "a".repeat(40);
  render(<CapabilitiesPanel rows={[
    { name: "Grounding", status: "in-progress", why: "Baseline evaluation", evidence_path: "docs/gate.md", commit },
  ]} />);
  const row = within(screen.getByRole("listitem"));
  expect(row.getByText("In progress")).toBeVisible();
  expect(row.getByText("Baseline evaluation")).toBeVisible();
  expect(row.getByRole("link")).toHaveAttribute("href", expect.stringContaining(`/blob/${commit}/docs/gate.md`));
});
