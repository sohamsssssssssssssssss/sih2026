"use client";

import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { chapterProgress, chapters } from "@/animation/chapters";
import { lerp, smoothstep } from "@/animation/interpolators";

const earthVertexShader = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vWorldPosition;
  varying vec2 vUv;

  void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPosition.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPosition;
  }
`;

const earthFragmentShader = /* glsl */ `
  uniform vec3 uSunDirection;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;
  varying vec2 vUv;

  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0)), f.x), f.y);
  }

  float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 5; i++) {
      value += amplitude * noise(p);
      p = p * 2.03 + vec2(19.4, 7.7);
      amplitude *= 0.5;
    }
    return value;
  }

  void main() {
    vec2 mapUv = vec2(vUv.x * 3.7, vUv.y * 2.25);
    float continents = fbm(mapUv + vec2(1.7, -0.2));
    continents += 0.28 * fbm(mapUv * vec2(0.65, 1.25) + vec2(7.0, 2.0));
    float landMask = smoothstep(0.56, 0.64, continents);

    float terrain = fbm(mapUv * 7.0);
    vec3 deepOcean = vec3(0.006, 0.055, 0.105);
    vec3 shelfOcean = vec3(0.012, 0.16, 0.21);
    vec3 ocean = mix(deepOcean, shelfOcean, smoothstep(0.35, 0.72, continents));
    vec3 vegetation = mix(vec3(0.045, 0.16, 0.095), vec3(0.19, 0.27, 0.11), terrain);
    vec3 arid = vec3(0.37, 0.31, 0.17);
    vec3 land = mix(vegetation, arid, smoothstep(0.58, 0.83, terrain));
    vec3 surface = mix(ocean, land, landMask);

    float cloudNoise = fbm(mapUv * 3.1 + vec2(4.2, 9.1));
    float clouds = smoothstep(0.66, 0.82, cloudNoise) * 0.52;
    surface = mix(surface, vec3(0.72, 0.79, 0.78), clouds);

    vec3 normal = normalize(vNormal);
    float diffuse = max(dot(normal, normalize(uSunDirection)), 0.0);
    float twilight = smoothstep(-0.12, 0.22, dot(normal, normalize(uSunDirection)));
    float rim = pow(1.0 - max(dot(normal, normalize(cameraPosition - vWorldPosition)), 0.0), 3.5);
    vec3 lit = surface * (0.12 + diffuse * 1.05);
    lit *= mix(0.16, 1.0, twilight);
    lit += vec3(0.05, 0.38, 0.52) * rim * 0.24;
    gl_FragColor = vec4(lit, 1.0);
  }
`;

const atmosphereVertexShader = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vWorldPosition;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPosition.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPosition;
  }
`;

const atmosphereFragmentShader = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vWorldPosition;
  void main() {
    vec3 viewDirection = normalize(cameraPosition - vWorldPosition);
    float intensity = pow(1.0 - abs(dot(vNormal, viewDirection)), 3.2);
    gl_FragColor = vec4(0.10, 0.58, 0.82, intensity * 0.62);
  }
`;

interface EarthProps {
  progress: React.RefObject<number>;
  reducedMotion: boolean;
}

export function Earth({ progress, reducedMotion }: EarthProps) {
  const group = useRef<THREE.Group>(null);
  const earth = useRef<THREE.Mesh>(null);
  const surfaceMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: earthVertexShader,
        fragmentShader: earthFragmentShader,
        uniforms: { uSunDirection: { value: new THREE.Vector3(-1.8, 1.2, 2.8) } },
      }),
    [],
  );
  const atmosphereMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: atmosphereVertexShader,
        fragmentShader: atmosphereFragmentShader,
        transparent: true,
        blending: THREE.AdditiveBlending,
        side: THREE.BackSide,
        depthWrite: false,
      }),
    [],
  );

  useFrame((_, delta) => {
    const value = progress.current ?? 0;
    const descent = smoothstep(chapterProgress(value, chapters.descent));
    if (earth.current && !reducedMotion) earth.current.rotation.y += delta * 0.035;
    if (group.current) {
      const scale = lerp(1, 1.09, descent);
      group.current.scale.setScalar(scale);
      group.current.position.y = lerp(-2.82, -2.5, descent);
    }
  });

  return (
    <group ref={group} position={[0, -2.82, 0]} rotation={[0.08, 0, -0.22]}>
      <mesh ref={earth} material={surfaceMaterial} rotation={[0, -0.7, 0]}>
        <sphereGeometry args={[3.58, 96, 96]} />
      </mesh>
      <mesh material={atmosphereMaterial} scale={1.035}>
        <sphereGeometry args={[3.58, 72, 72]} />
      </mesh>
    </group>
  );
}
