import { afterEach, expect, it, vi } from "vitest";
import { kosiChange } from "@/test/kosi-change-fixture";
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); vi.resetModules(); });
it.each(["http://localhost:8000", ""])("uses API base %s", async base => {
  vi.stubEnv("NEXT_PUBLIC_API_URL", base); vi.resetModules();
  const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(kosiChange), { status: 200 }));
  vi.stubGlobal("fetch", fetch);
  const api = await import("./api");
  expect(await api.getKosiSarChange()).toEqual(kosiChange);
  expect(fetch).toHaveBeenCalledWith(`${base}/api/sar/change/kosi-2024`, expect.objectContaining({ cache: "no-store" }));
  expect(api.getKosiSarRasterUrl(kosiChange.raster_download_url)).toBe(`${base}/api/sar/change/kosi-2024/raster`);
});
it.each([404, 409, 422, 503])("reports API errors %s", async status => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Artifact unavailable or invalid" }), { status })));
  const api = await import("./api");
  await expect(api.getKosiSarChange()).rejects.toThrow("Artifact unavailable or invalid");
});
it("rejects malformed metadata instead of displaying a partial result", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ ...kosiChange, statistics: {} }))));
  const api = await import("./api");
  await expect(api.getKosiSarChange()).rejects.toThrow("metadata is invalid");
});
