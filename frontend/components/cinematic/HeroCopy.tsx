"use client";

import { useEffect, useRef } from "react";
import { animate, stagger } from "animejs";

export function HeroCopy({ reducedMotion }: { reducedMotion: boolean }) {
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!root.current || reducedMotion) return;
    const items = root.current.querySelectorAll("[data-intro]");
    const animation = animate(items, {
      opacity: { from: 0 },
      y: { from: 18 },
      duration: 1100,
      delay: stagger(110, { start: 180 }),
      ease: "out(4)",
    });
    return () => {
      animation.cancel();
    };
  }, [reducedMotion]);

  return (
    <div ref={root} className="cinematic-hero-copy">
      <p data-intro className="cinematic-kicker">SATQUERY // GEOSPATIAL INTELLIGENCE SYSTEM</p>
      <h1 data-intro>ASK EARTH<br />A QUESTION.</h1>
      <p data-intro className="cinematic-subtitle">Vision-Language Intelligence for Satellite Imagery</p>
    </div>
  );
}
