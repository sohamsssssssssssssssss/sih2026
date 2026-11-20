import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { AskWorkspace } from "./AskWorkspace";

afterEach(() => vi.unstubAllGlobals());

it("shows the live fetch deadline and the cached acquisition date", async () => {
  const message = "Live fetch timed out after 1s — showing cached scene from 2026-10-28";
  vi.stubGlobal("fetch", vi.fn(async (input: string) => {
    const path = new URL(input).pathname;
    const responses: Record<string, unknown> = {
      "/scenes/fetch": {
        scene_ref: { pack_id: "fresh-backups", patch_id: "ludhiana-s2" },
        image: "data:image/png;base64,",
        preview_kind: "RGB quicklook",
        scene: { sensor: "S2", source: "Earth Search", acquired: "2026-10-28", cloud_pct: null },
        provenance: {
          source: "Earth Search", collection: "sentinel-2-l2a", item_id: "ludhiana",
          acquisition_datetime: "2026-10-28T00:00:00Z", mode: "cached", label: "Cached scene (offline)",
        },
        fallback_reason: message,
        backup_place: "Ludhiana",
      },
      "/ask": { status: "low_confidence", text: "Land cover", evidence: [], trace_id: "demo" },
      "/trace/demo": { events: [], verified: true },
    };
    return new Response(JSON.stringify(responses[path] ?? {}), { status: path in responses ? 200 : 503 });
  }));
  render(<AskWorkspace />);
  fireEvent.change(screen.getByLabelText("Place or latitude, longitude"), { target: { value: "Ludhiana" } });
  fireEvent.change(screen.getByLabelText("Start date (UTC)"), { target: { value: "2026-10-28" } });
  fireEvent.click(screen.getByRole("button", { name: /^Ask$/ }));
  expect(await screen.findByText(message, { exact: true })).toBeInTheDocument();
});
