import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { GOLDEN_SCENE } from "@/lib/workspace-contract";

// MapLibre needs WebGL, which jsdom lacks; the viewer is covered by its own tests.
vi.mock("@/components/imagery/ImageryViewer", () => ({
  ImageryViewer: ({ sceneId }: { sceneId: string }) => <div data-testid="viewer">{sceneId}</div>,
}));

const { Workspace } = await import("./Workspace");

const NEW_ID = "scene_0123456789abcdef0123456789abcdef";
const curated = {
  id: "dior-rsvg-07272", title: "DIOR-RSVG yellow ship", capability: "grounding", result_state: "real_live", catalog_visible: true,
  available: true, unavailable_reason: null, question: "Locate a yellow ship.", source: { dataset: "DIOR-RSVG", source_id: "JPEGImages/07272.jpg" },
};

function json(status: number, body: unknown) { return new Response(JSON.stringify(body), { status }); }

afterEach(() => { vi.unstubAllGlobals(); });

describe("Workspace", () => {
  it("lists curated scenes when the backend omits uploads, and switches scene", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(200, { version: "1.0", scenes: [curated] })));
    render(<Workspace />);
    const picker = screen.getByLabelText("Scene") as HTMLSelectElement;
    await within(screen.getByLabelText("Scene")).findByRole("option", { name: "DIOR-RSVG yellow ship" });
    expect(picker.value).toBe(GOLDEN_SCENE.id);
    expect(screen.getByLabelText("Question")).toHaveValue(GOLDEN_SCENE.question);

    fireEvent.change(picker, { target: { value: curated.id } });
    expect(screen.getByTestId("viewer")).toHaveTextContent(curated.id);
    expect(screen.getByLabelText("Question")).toHaveValue("Locate a yellow ship.");
  });

  it("selects a scene right after it is uploaded", async () => {
    const created = { scene_id: NEW_ID, filename: "t1.tif", format: "TIFF", width: 64, height: 32, sensor: null, gsd: null, location: null, acquisition_date: null };
    const upload = { scene_id: NEW_ID, filename: "t1.tif", format: "TIFF", width: 64, height: 32, modality: "optical", sensor: null, acquisition_time: null, has_native_raster: true, georeferenced: true, uploaded_at: "2026-10-05T00:00:00Z" };
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === "POST") return json(201, created);
      const listed = fetchMock.mock.calls.filter(([, options]) => options?.method === "POST").length > 0;
      return json(200, { version: "1.0", scenes: [curated], uploads: listed ? [upload] : [] });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<Workspace />);
    await within(screen.getByLabelText("Scene")).findByRole("option", { name: "DIOR-RSVG yellow ship" });

    const form = screen.getByRole("form", { name: "Upload a scene" });
    fireEvent.change(within(form).getByTestId("upload-input"), { target: { files: [new File([new Uint8Array(8)], "t1.tif", { type: "image/tiff" })] } });
    fireEvent.click(within(form).getByRole("button", { name: /upload scene/i }));

    await waitFor(() => expect((screen.getByLabelText("Scene") as HTMLSelectElement).value).toBe(NEW_ID));
    expect(screen.getByTestId("viewer")).toHaveTextContent(NEW_ID);
    expect(screen.getByText("UPLOADED SCENE")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Yes · georeferenced")).toBeInTheDocument());
  });
});
