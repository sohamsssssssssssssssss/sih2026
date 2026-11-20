import { resolve } from "node:path";
import { expect, test, type Page } from "@playwright/test";

const screenshots = resolve("../docs/screens/p17");
const examples = [
  ["Punjab land cover", "Low confidence", [/strongest land-cover signals/, /Scene-level cloud metadata unavailable/, /uncalibrated score/]],
  ["India flood", "Low confidence", [/possible open water/, /Experimental/]],
  ["Two-scene change", "Refused", [/T18 found/]],
  ["Outside India", "Refused", [/outside the supported India region/]],
  ["Cloudy Mumbai", "Refused", [/Too cloudy to answer reliably/, /scene-level cloud cover 99\.98%, limit 50%/, /Sentinel-1 \(S1\)/]],
] as const;

for (const [name, outcome, phrases] of examples) {
  test(`demo example: ${name}`, async ({ page }) => {
    await page.goto("/ask");
    const capabilities = page.getByRole("complementary", { name: "Capabilities" });
    await expect(capabilities.getByText("Experimental", { exact: true }).first()).toBeVisible();
    await expect(capabilities.getByText("Change QA")).toHaveCount(0);
    await expect(capabilities.getByText("Requires CUDA")).toHaveCount(2);

    const upcomingAnswer = page.waitForResponse((response) =>
      response.request().method() === "POST" && new URL(response.url()).pathname === "/ask",
    );
    await page.getByRole("button", { name, exact: true }).click();
    const response = await upcomingAnswer;
    expect(response.ok()).toBe(true);
    const answer = await response.json();
    const result = page.getByRole("region", { name: "Result", exact: true });
    await expect(result.getByRole("heading", { name: outcome })).toBeVisible();
    for (const phrase of phrases) await expect(result.getByText(phrase).first()).toBeVisible();
    await expect(result.getByText("Chain verified ✓", { exact: true })).toBeVisible();
    await expect(result.getByText(answer.trace_id, { exact: true })).toBeHidden();
    await result.locator("summary").click();
    await expect(result.getByText(answer.trace_id, { exact: true })).toBeVisible();
    await expect(result.getByRole("list", { name: "Trace events" }).getByRole("listitem").first()).toBeVisible();
  });
}

test("upload a GeoTIFF and submit a typed question", async ({ page }) => {
  await page.goto("/ask");
  await page.getByLabel("Upload GeoTIFF").setInputFiles(resolve("../tests/fixtures/mumbai-s2-cloudy-v2.tif"));
  await expect(page.getByRole("combobox", { name: "Scene", exact: true })).toHaveValue(/^upload_/);
  await page.getByRole("textbox", { name: "Question" }).fill("Describe the land cover");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  const result = page.getByRole("region", { name: "Result", exact: true });
  await expect(result.getByRole("heading", { name: "Refused" })).toBeVisible();
  await expect(result.getByText(/Try imagery from Sentinel-1/)).toBeVisible();
  await expect(result.getByText("Chain verified ✓", { exact: true })).toBeVisible();
});

async function askAt(page: Page, { place, start, end, sensor = "S2" }: {
  place: string; start: string; end?: string; sensor?: "S1" | "S2";
}) {
  await page.goto("/ask");
  await page.getByLabel("Place or latitude, longitude").fill(place);
  await page.getByRole("combobox", { name: "Sensor", exact: true }).selectOption(sensor);
  await page.getByLabel("Start date (UTC)").fill(start);
  if (end) await page.getByLabel("End date (optional)").fill(end);
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  return page.getByRole("region", { name: "Result", exact: true });
}

test("acquire live Sentinel-2 and Sentinel-1 chips from STAC and answer through the gate", async ({ page, request }) => {
  test.skip(process.env.SATQUERY_OFFLINE === "1", "Backend is in offline mode");
  const catalogs = [
    "https://earth-search.aws.element84.com/v1",
    "https://planetarycomputer.microsoft.com/api/stac/v1",
  ];
  for (const catalog of catalogs) {
    try {
      const response = await request.get(catalog, { timeout: 8_000, maxRedirects: 2 });
      test.skip(response.status() >= 500, `STAC is unavailable: ${catalog}`);
      expect(response.ok()).toBe(true);
    } catch (error) {
      if (/ENOTFOUND|EAI_AGAIN|ENETUNREACH|ECONNREFUSED|ECONNRESET|Timeout/i.test(String(error))) {
        test.skip(true, `STAC cannot be reached: ${catalog}`);
      }
      throw error;
    }
  }

  let result = await askAt(page, { place: "30.90, 75.86", start: "2026-10-01", end: "2026-11-17" });
  await expect(result.getByText(/strongest land-cover signals/)).toBeVisible();
  await expect(result.getByText("uncalibrated score", { exact: true })).toHaveCount(5);
  await expect(result.getByRole("img", { name: "S2 RGB quicklook" })).toBeVisible();
  await expect(result.getByText("Live scene", { exact: true })).toBeVisible();
  await expect(result.getByText(/Earth Search · sentinel-2-l2a/)).toBeVisible();
  await expect(result.getByText("Chain verified ✓", { exact: true })).toBeVisible();
  await page.screenshot({ path: resolve(screenshots, "live-s2.png"), fullPage: true, animations: "disabled" });

  await result.locator("summary").click();
  await expect(result.getByRole("list", { name: "Trace events" })).toBeVisible();
  await page.screenshot({ path: resolve(screenshots, "expanded-trace.png"), fullPage: true, animations: "disabled" });

  result = await askAt(page, { place: "24.795, 92.795", start: "2026-11-17", sensor: "S1" });
  await expect(result.getByText(/strongest land-cover signals/)).toBeVisible();
  await expect(result.getByRole("img", { name: "S1 VV grayscale" })).toBeVisible();
  await expect(result.getByText("Live scene", { exact: true })).toBeVisible();
  await expect(result.getByText(/Microsoft Planetary Computer · sentinel-1-rtc/)).toBeVisible();
  await expect(result.getByText("Chain verified ✓", { exact: true })).toBeVisible();
  await page.screenshot({ path: resolve(screenshots, "live-s1.png"), fullPage: true, animations: "disabled" });

  result = await askAt(page, { place: "19.025, 72.835", start: "2025-08-29" });
  await expect(result.getByText(/Too cloudy to answer reliably \(scene-level cloud cover .*limit 50%\)/)).toBeVisible();
  await expect(result.getByText("Live scene", { exact: true })).toBeVisible();
  await expect(result.getByText("Chain verified ✓", { exact: true })).toBeVisible();
  await page.screenshot({ path: resolve(screenshots, "cloud-refusal.png"), fullPage: true, animations: "disabled" });
});

test("refuse an outside-India point before acquisition", async ({ page }) => {
  const result = await askAt(page, { place: "48.45, 16.15", start: "2026-10-28" });
  await expect(result.getByText("Cannot answer because the scene extends outside the supported India region.", { exact: true })).toBeVisible();
  await expect(result.getByRole("img")).toHaveCount(0);
  await expect(result.getByText("Chain verified ✓", { exact: true })).toBeVisible();
  await page.screenshot({ path: resolve(screenshots, "outside-india-refusal.png"), fullPage: true, animations: "disabled" });
  await result.locator("summary").click();
  await expect(result.getByText("not_answerable", { exact: true })).toBeVisible();
});

test("offline fallback uses a real checksum-verified backup scene", async ({ page }) => {
  test.skip(process.env.SATQUERY_OFFLINE !== "1", "Run the demo with SATQUERY_OFFLINE=1 for this case");
  const result = await askAt(page, { place: "Ludhiana", start: "2026-11-01" });
  await expect(result.getByText(/strongest land-cover signals/)).toBeVisible();
  await expect(result.getByRole("img", { name: "S2 RGB quicklook" })).toBeVisible();
  await expect(result.getByText("Cached scene (offline)", { exact: true })).toBeVisible();
  await expect(result.getByText(/Offline mode is enabled. Showing the Ludhiana backup/)).toBeVisible();
  await expect(result.getByText("Chain verified ✓", { exact: true })).toBeVisible();
  await page.screenshot({ path: resolve(screenshots, "offline-fallback.png"), fullPage: true, animations: "disabled" });
});
