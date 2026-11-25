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
  webServer: process.env.SATQUERY_E2E_USE_RUNNING_DEMO === "1" ? undefined : {
    command: "bash ../scripts/run-demo.sh",
    env: { SATQUERY_VQA_MLX: process.env.SATQUERY_VQA_MLX ?? "0" },
    url: "http://127.0.0.1:3000/ask",
    reuseExistingServer: false,
    timeout: 180_000,
  },
});
