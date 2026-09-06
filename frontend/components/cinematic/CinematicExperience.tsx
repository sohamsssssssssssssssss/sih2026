"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ArrowDown, ArrowUpRight, Satellite } from "lucide-react";
import { useStoryScrollProgress } from "@/animation/scrollProgress";
import { HeroCopy } from "./HeroCopy";
import { Telemetry } from "./Telemetry";

const SceneCanvas = dynamic(
  () => import("./SceneCanvas").then((module) => module.SceneCanvas),
  { ssr: false },
);

export function CinematicExperience() {
  const storyRef = useRef<HTMLElement>(null);
  const progressRef = useRef(0);
  const [preferences, setPreferences] = useState({ reducedMotion: false, compact: false });
  useStoryScrollProgress(storyRef, progressRef);

  useEffect(() => {
    const reducedQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const compactQuery = window.matchMedia("(max-width: 720px)");
    const update = () => setPreferences({ reducedMotion: reducedQuery.matches, compact: compactQuery.matches });
    update();
    reducedQuery.addEventListener("change", update);
    compactQuery.addEventListener("change", update);
    return () => {
      reducedQuery.removeEventListener("change", update);
      compactQuery.removeEventListener("change", update);
    };
  }, []);

  return (
    <section
      ref={storyRef}
      className="cinematic-story"
      data-reduced-motion={preferences.reducedMotion ? "true" : "false"}
    >
      <div className="cinematic-viewport">
        <div className="cinematic-canvas" aria-hidden="true">
          <SceneCanvas
            progress={progressRef}
            reducedMotion={preferences.reducedMotion}
            compact={preferences.compact}
          />
        </div>
        <div className="cinematic-vignette" aria-hidden="true" />

        <header className="cinematic-header">
          <Link href="/" className="cinematic-brand" aria-label="SatQuery AI home">
            <span><Satellite size={16} /></span>
            SATQUERY AI
          </Link>
          <Link href="/workspace" className="cinematic-enter-link">
            ENTER WORKSPACE <ArrowUpRight size={14} />
          </Link>
        </header>

        <HeroCopy reducedMotion={preferences.reducedMotion} />
        <Telemetry />

        <div className="cinematic-scroll-cue" aria-hidden="true">
          <span>SCROLL TO DESCEND</span>
          <ArrowDown size={15} />
        </div>

        <div className="cinematic-progress" aria-hidden="true">
          <span>ORBIT</span>
          <div><i /></div>
          <span>SURFACE</span>
        </div>
      </div>
    </section>
  );
}
