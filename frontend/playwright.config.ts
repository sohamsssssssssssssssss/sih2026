import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  timeout: 90_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: "http://127.0.0.1:3000",
    channel: "chrome",
    viewport: { width: 1536, height: 1024 },
    colorScheme: "light",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "bash ../scripts/run-demo.sh",
    url: "http://127.0.0.1:3000/ask",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
