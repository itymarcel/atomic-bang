import type { SimulationConfig } from "./config.js";

export interface PresetBH { x: number; y: number; z: number; mass: number; }

export interface Preset {
  value: string;
  label: string;
  config: SimulationConfig;
  collisionEnabled: boolean;
  blackHoles: PresetBH[];
}

export const PRESETS: Preset[] = [
  {
    value: "spiral",
    label: "Spiral formation [+1bh]",
    config: {
      particleCount: 1766000,
      gravity: 100,
      impactSpeed: 38,
      sizeVariation: 120,
      lifetime: 510,
      randomness: 95,
      angularMomentum: 90,
      timeScale: 1,
      collisionStrength: 4,
    },
    collisionEnabled: true,
    blackHoles: [{ x: 10, y: 6, z: 3, mass: 10 }],
  },
  {
    value: "sparse",
    label: "Sparse formation",
    config: {
      particleCount: 1300000,
      gravity: 100,
      impactSpeed: 38,
      sizeVariation: 120,
      lifetime: 510,
      randomness: 95,
      angularMomentum: 90,
      timeScale: 1,
      collisionStrength: 4,
    },
    collisionEnabled: true,
    blackHoles: [],
  },
];
