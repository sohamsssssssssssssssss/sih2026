import { clamp } from "./interpolators";

export const chapters = {
  hero: [0, 0.12],
  satelliteArrival: [0.12, 0.28],
  satelliteLock: [0.28, 0.35],
  descent: [0.35, 0.52],
  terrainReveal: [0.52, 0.64],
  growth: [0.64, 0.76],
  intelligence: [0.76, 0.91],
  productReveal: [0.91, 1],
} as const satisfies Record<string, readonly [number, number]>;

export type ChapterName = keyof typeof chapters;

export function chapterProgress(
  masterProgress: number,
  range: readonly [number, number],
) {
  const [start, end] = range;
  return clamp((masterProgress - start) / (end - start));
}
