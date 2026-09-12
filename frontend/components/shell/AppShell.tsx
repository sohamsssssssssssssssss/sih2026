"use client";

import { usePathname } from "next/navigation";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  if (pathname === "/" || pathname === "/intro") return children;

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[236px_minmax(0,1fr)]">
      <a href="#main-content" className="sr-only fixed left-4 top-4 z-50 rounded border border-border bg-surface px-4 py-2 text-white focus:not-sr-only">Skip to main content</a>
      <Sidebar />
      <div className="min-w-0">
        <TopBar />
        <main id="main-content" tabIndex={-1} className="min-w-0 px-4 py-5 focus:outline-none lg:px-6">{children}</main>
      </div>
    </div>
  );
}
