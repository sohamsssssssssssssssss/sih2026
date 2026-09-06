"use client";

import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import { Earth } from "./Earth";
import { Satellite } from "./Satellite";
import { SceneCamera } from "./SceneCamera";
import { Stars } from "./Stars";

export function SceneCanvas({
  progress,
  reducedMotion,
  compact,
}: {
  progress: React.RefObject<number>;
  reducedMotion: boolean;
  compact: boolean;
}) {
  return (
    <Canvas
      dpr={compact ? [1, 1.2] : [1, 1.5]}
      camera={{ position: [0, 0.05, 8.35], fov: 42, near: 0.1, far: 70 }}
      gl={{ antialias: !compact, alpha: false, powerPreference: "high-performance" }}
      onCreated={({ gl }) => {
        gl.setClearColor("#020507");
        gl.outputColorSpace = "srgb";
        gl.toneMapping = THREE.ACESFilmicToneMapping;
        gl.toneMappingExposure = 1.08;
      }}
    >
      <SceneCamera progress={progress} reducedMotion={reducedMotion} />
      <Stars progress={progress} compact={compact} />
      <ambientLight intensity={0.08} />
      <directionalLight position={[-4, 4, 6]} intensity={2.8} color="#e8f7ff" />
      <Earth progress={progress} reducedMotion={reducedMotion} />
      <Satellite progress={progress} reducedMotion={reducedMotion} />
    </Canvas>
  );
}
