import type { Config } from "tailwindcss";

const colors = {
  background: "#0b101d",
  surface: "#121a2d",
  "surface-elevated": "#080c17",
  border: "rgba(56, 189, 248, 0.16)",
  primary: "#06b6d4",
  onyx: "#f1f5f9",
  subtitle: "#94a3b8",
  secondary: "#94a3b8",
  tertiary: "#64748b",
  muted: "#64748b",
  cyan: "#38bdf8",
  raised: "#18233c",
  accent: "#38bdf8",
  success: "#10b981",
  warning: "#f59e0b",
  error: "#ef4444",
};

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ...colors,
      },
      boxShadow: {
        none: "none",
        panel: "0 20px 50px rgba(0, 0, 0, 0.18)",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "IBM Plex Mono", "Geist Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
