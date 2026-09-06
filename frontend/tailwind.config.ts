import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        background: "#071019",
        surface: "#0c1823",
        raised: "#122433",
        border: "#20394b",
        accent: "#47d7dd",
        success: "#55d68a",
        warning: "#f0ad4e",
        error: "#ff6b75",
      },
      boxShadow: {
        panel: "0 18px 50px rgba(0, 0, 0, 0.22)",
        glow: "0 0 30px rgba(71, 215, 221, 0.12)",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
