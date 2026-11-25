import { expect, test } from "@playwright/test";

test("capabilities renders every evidence row without external requests", async ({ page, request }) => {
  const external: string[] = [];
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (["127.0.0.1", "localhost"].includes(url.hostname)) return route.continue();
    external.push(url.href);
    return route.abort();
  });
  const response = await request.get("http://127.0.0.1:8000/capabilities");
  expect(response.ok()).toBe(true);
  const rows = (await response.json()).evidence;
  expect(rows).toHaveLength(10);
  await page.goto("/capabilities");
  await expect(page.getByRole("heading", { name: "Capabilities", exact: true })).toBeVisible();
  const list = page.getByRole("list", { name: "Demo capabilities" });
  await expect(list.getByRole("listitem")).toHaveCount(rows.length);
  for (const row of rows) {
    await expect(list.getByRole("heading", { name: row.name, exact: true })).toBeVisible();
    await expect(list.locator(`a[href$="/${row.commit}/${row.evidence_path}"]`)).toBeVisible();
  }
  expect(external).toEqual([]);
  await page.screenshot({ path: process.env.SATQUERY_CAPABILITIES_SCREENSHOT ?? test.info().outputPath("capabilities.png"), fullPage: true });
});
