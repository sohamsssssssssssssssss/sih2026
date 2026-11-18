import { resolve } from "node:path";
import { expect, test } from "@playwright/test";

const examples = [
  ["Punjab land cover", "01-punjab-land-cover.png", "Low confidence", [/Uncalibrated coarse land-cover presence scores/, /Cloud cover not checked/, /Confidence: Uncalibrated/, /Experimental/]],
  ["India flood", "02-india-flood.png", "Low confidence", [/Experimental radar open-water mask/, /Open-water mask coverage/, /Confidence: Uncalibrated/]],
  ["Two-scene change", "03-two-scene-change.png", "Refused", [/T18 found/, /not_answerable/]],
  ["Outside India", "04-outside-india.png", "Refused", [/outside the supported India region/, /not_answerable/]],
  ["Cloudy Mumbai", "05-cloudy-mumbai.png", "Refused", [/Optical cloud cover is 99\.980801% \(limit 50%\)/, /Sentinel-1 \(S1\)/]],
] as const;

for (const [name, screenshot, outcome, phrases] of examples) {
  test(`live example: ${name}`, async ({ page }) => {
    await page.goto("/ask");
    const capabilities = page.getByRole("complementary", { name: "Capabilities" });
    await expect(capabilities.getByText("Disabled by decision")).toBeVisible();
    await expect(capabilities.getByText(/T18 found/)).toBeVisible();
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
    await expect(result.getByText(answer.trace_id, { exact: true })).toBeVisible();
    await expect(result.getByText("Chain verified", { exact: true })).toBeVisible();
    await result.locator("summary").click();
    await expect(result.getByRole("list", { name: "Trace events" }).getByRole("listitem").first()).toBeVisible();
    await page.screenshot({ path: resolve("../docs/plan/ui-acceptance", screenshot), fullPage: true, animations: "disabled" });
  });
}

test("upload a GeoTIFF and submit a typed question", async ({ page }) => {
  await page.goto("/ask");
  await page.getByLabel("Upload GeoTIFF").setInputFiles(resolve("../tests/fixtures/mumbai-s2-cloudy.tif"));
  await expect(page.getByRole("combobox", { name: "Scene", exact: true })).toHaveValue(/^upload_/);
  await page.getByRole("textbox", { name: "Question" }).fill("Describe the land cover");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  const result = page.getByRole("region", { name: "Result", exact: true });
  await expect(result.getByRole("heading", { name: "Refused" })).toBeVisible();
  await expect(result.getByText(/Try imagery from Sentinel-1/)).toBeVisible();
  await expect(result.getByText("Chain verified", { exact: true })).toBeVisible();
});
