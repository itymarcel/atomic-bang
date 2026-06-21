export interface SimulationConfig {
  particleCount: number;
  gravity: number;
  impactSpeed: number;
  sizeVariation: number;
  lifetime: number;
  randomness: number;
  timeScale: number;
}

export const DEFAULT_CONFIG: SimulationConfig = {
  particleCount: 50000,
  gravity: 82,
  impactSpeed: 82,
  sizeVariation: 120,
  lifetime: 180,
  randomness: 64,
  timeScale: 1,
};

export interface ControlDefinition {
  key: keyof SimulationConfig;
  label: string;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
}

export const CONTROL_DEFINITIONS: ControlDefinition[] = [
  { key: "particleCount", label: "Particle amount", min: 1000, max: 2000000, step: 1000, format: v => v.toLocaleString() },
  { key: "gravity", label: "Gravity", min: 0, max: 100, step: 1, format: v => `${v}%` },
  { key: "impactSpeed", label: "Explosion velocity", min: 10, max: 180, step: 2, format: v => `${v} u/s` },
  { key: "sizeVariation", label: "Luminosity spectrum", min: 0, max: 200, step: 2, format: v => `${v}%` },
  { key: "lifetime", label: "Mean lifetime", min: 10, max: 600, step: 10, format: v => `${v}s` },
  { key: "randomness", label: "Entropy", min: 0, max: 100, step: 1, format: v => `${v}%` },
  { key: "timeScale", label: "Overall speed", min: 0.1, max: 4, step: 0.1, format: v => `${v.toFixed(1)}×` },
];
