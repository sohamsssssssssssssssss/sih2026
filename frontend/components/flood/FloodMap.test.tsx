import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { emptyPolygons, truncatedPolygonBounds, truncatedPolygons } from "@/test/__fixtures__/flood";

// jsdom has no WebGL, so MapLibre is replaced by a constructor spy.
const { MapSpy } = vi.hoisted(() => ({
  MapSpy: vi.fn(function (_options: Record<string, unknown>) { return { on: vi.fn(), remove: vi.fn() }; }),
}));
vi.mock("maplibre-gl", () => ({ Map: MapSpy, setWorkerUrl: vi.fn() }));

const { FloodMap } = await import("./FloodMap");

beforeEach(() => { MapSpy.mockClear(); });

describe("FloodMap", () => {
  it("draws the returned GeoJSON, fits its own bounds and says it is truncated", () => {
    render(<FloodMap item={truncatedPolygons} />);
    expect(MapSpy).toHaveBeenCalledTimes(1);
    const options = MapSpy.mock.calls[0][0] as { bounds: unknown; center?: unknown; style: { sources: { water: { data: unknown } } } };
    expect(options.style.sources.water.data).toBe(truncatedPolygons.geojson);
    expect(options.bounds).toEqual(truncatedPolygonBounds);
    expect(options.center).toBeUndefined();
    expect(screen.getByText("Showing 2 of 3 polygons (largest first)")).toBeInTheDocument();
    expect(screen.getByText(/New water/)).toBeInTheDocument();
    expect(screen.getByText(/Receded water/)).toBeInTheDocument();
    expect(screen.getByText("© OpenStreetMap contributors")).toBeInTheDocument();
  });

  it("draws no map when no polygons are returned", () => {
    render(<FloodMap item={emptyPolygons} />);
    expect(MapSpy).not.toHaveBeenCalled();
    expect(screen.getByText(/No water-change polygons were returned/)).toBeInTheDocument();
  });
});
