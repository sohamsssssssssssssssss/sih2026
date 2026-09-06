"use client";

import { useFrame, useThree } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import { chapterProgress, chapters } from "@/animation/chapters";
import { lerp, smoothstep } from "@/animation/interpolators";

export function SceneCamera({ progress, reducedMotion }: { progress: React.RefObject<number>; reducedMotion: boolean }) {
  const { camera } = useThree();
  const target = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    const descent = reducedMotion ? 0 : smoothstep(chapterProgress(progress.current ?? 0, chapters.descent));
    camera.position.set(lerp(0, 0.08, descent), lerp(0.05, -0.28, descent), lerp(8.35, 5.2, descent));
    target.set(0, lerp(-1.5, -2.25, descent), 0);
    camera.lookAt(target);
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.fov = lerp(42, 48, descent);
      camera.updateProjectionMatrix();
    }
  });

  return null;
}
