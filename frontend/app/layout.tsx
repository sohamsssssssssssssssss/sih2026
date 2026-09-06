import type { Metadata } from "next";
import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";
import { AppShell } from "@/components/shell/AppShell";

export const metadata: Metadata = {
  title: { default: "SatQuery AI", template: "%s · SatQuery AI" },
  description: "Evidence-backed geospatial intelligence with explicit execution provenance.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased" suppressHydrationWarning><AppShell>{children}</AppShell></body>
    </html>
  );
}
