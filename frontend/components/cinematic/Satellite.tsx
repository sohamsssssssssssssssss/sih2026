"use client";

import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { chapterProgress, chapters } from "@/animation/chapters";
import { easeInOutCubic } from "@/animation/interpolators";

const forward = new THREE.Vector3(1, 0, 0);

export function Satellite({ progress, reducedMotion }: { progress: React.RefObject<number>; reducedMotion: boolean }) {
  const satellite = useRef<THREE.Group>(null);
  const path = useMemo(
    () =>
      new THREE.CatmullRomCurve3(
        [
          new THREE.Vector3(-8.4, -0.4, 0.1),
          new THREE.Vector3(-6.1, 0.1, -0.15),
          new THREE.Vector3(-3.8, 1.4, 0.05),
          new THREE.Vector3(-1.6, 2.45, 0.18),
          new THREE.Vector3(0, 2.82, 0.28),
        ],
        false,
        "catmullrom",
        0.52,
      ),
    [],
  );

  useFrame(() => {
    if (!satellite.current) return;
    const arrival = reducedMotion
      ? 1
      : easeInOutCubic(chapterProgress(progress.current ?? 0, chapters.satelliteArrival));
    const position = path.getPointAt(arrival);
    const tangent = path.getTangentAt(Math.min(arrival, 0.995)).normalize();
    satellite.current.position.copy(position);
    satellite.current.quaternion.setFromUnitVectors(forward, tangent);
    satellite.current.rotateX(Math.PI * 0.06);
    satellite.current.visible = reducedMotion || (progress.current ?? 0) >= chapters.satelliteArrival[0] - 0.015;
  });

  return (
    <group ref={satellite} scale={0.42}>
      <mesh>
        <boxGeometry args={[1.35, 0.44, 0.54]} />
        <meshStandardMaterial color="#b8c2c5" metalness={0.75} roughness={0.28} />
      </mesh>
      <mesh position={[0.58, 0, 0]}>
        <cylinderGeometry args={[0.21, 0.32, 0.52, 12]} />
        <meshStandardMaterial color="#76848a" metalness={0.82} roughness={0.3} />
      </mesh>
      {[-1.35, 1.35].map((offset) => (
        <group key={offset} position={[0, offset, 0]}>
          <mesh>
            <boxGeometry args={[1.95, 1.65, 0.045]} />
            <meshStandardMaterial color="#102c4f" metalness={0.45} roughness={0.38} />
          </mesh>
          <lineSegments position={[0, 0, 0.026]}>
            <edgesGeometry args={[new THREE.BoxGeometry(1.95, 1.65, 0.045)]} />
            <lineBasicMaterial color="#4f7693" transparent opacity={0.7} />
          </lineSegments>
        </group>
      ))}
      <pointLight position={[0.55, 0, 0.4]} color="#d9f9ff" intensity={0.35} distance={2.5} />
    </group>
  );
}
