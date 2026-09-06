"use client";

import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

function seededRandom(index: number) {
  const value = Math.sin(index * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
}

export function Stars({ progress, compact }: { progress: React.RefObject<number>; compact: boolean }) {
  const points = useRef<THREE.Points>(null);
  const positions = useMemo(() => {
    const count = compact ? 600 : 1100;
    const values = new Float32Array(count * 3);
    for (let index = 0; index < count; index += 1) {
      values[index * 3] = (seededRandom(index * 3) - 0.5) * 32;
      values[index * 3 + 1] = (seededRandom(index * 3 + 1) - 0.5) * 20;
      values[index * 3 + 2] = -3 - seededRandom(index * 3 + 2) * 14;
    }
    return values;
  }, [compact]);

  useFrame(() => {
    if (!points.current) return;
    const value = progress.current ?? 0;
    points.current.position.x = value * -0.34;
    points.current.position.y = value * 0.18;
  });

  return (
    <points ref={points}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial color="#9fb5bf" size={compact ? 0.018 : 0.022} sizeAttenuation transparent opacity={0.52} depthWrite={false} />
    </points>
  );
}
