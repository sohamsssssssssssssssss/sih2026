export function clamp(value: number, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

export function lerp(start: number, end: number, progress: number) {
  return start + (end - start) * clamp(progress);
}

export function smoothstep(progress: number) {
  const value = clamp(progress);
  return value * value * (3 - 2 * value);
}

export function easeInOutCubic(progress: number) {
  const value = clamp(progress);
  return value < 0.5
    ? 4 * value * value * value
    : 1 - Math.pow(-2 * value + 2, 3) / 2;
}
