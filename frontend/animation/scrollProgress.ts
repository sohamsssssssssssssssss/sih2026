"use client";

import { useEffect, type RefObject } from "react";
import { clamp } from "./interpolators";

export function useStoryScrollProgress(
  storyRef: RefObject<HTMLElement | null>,
  progressRef: RefObject<number>,
) {
  useEffect(() => {
    const story = storyRef.current;
    if (!story) return;

    let frame = 0;
    let storyTop = 0;
    let scrollableDistance = 1;

    const measure = () => {
      storyTop = window.scrollY + story.getBoundingClientRect().top;
      scrollableDistance = Math.max(1, story.offsetHeight - window.innerHeight);
    };

    const update = () => {
      frame = 0;
      const progress = clamp((window.scrollY - storyTop) / scrollableDistance);
      progressRef.current = progress;
      story.style.setProperty("--story-progress", progress.toFixed(4));
    };

    const requestUpdate = () => {
      if (frame) return;
      frame = window.requestAnimationFrame(update);
    };

    const handleResize = () => {
      measure();
      requestUpdate();
    };

    measure();
    update();
    window.addEventListener("scroll", requestUpdate, { passive: true });
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("scroll", requestUpdate);
      window.removeEventListener("resize", handleResize);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [progressRef, storyRef]);
}
