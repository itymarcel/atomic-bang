import { ParticleStore } from "./ParticleStore.js";
import { GridGravity } from "./GridGravity.js";
import type { SimulationConfig } from "./config.js";

type Phase = "ready" | "approach" | "running";

export class Simulation {
  readonly particles = new ParticleStore(250000);
  phase: Phase = "ready";
  paused = false;
  age = 0;
  private approachTime = 0;
  private readonly gravityField = new GridGravity();
  private gravityReady = false;
  private representedParticleCount = 1;
  private config: SimulationConfig;

  constructor(private readonly canvas: HTMLCanvasElement, config: SimulationConfig) {
    this.config = config;
  }

  get timeScale(): number { return this.config.timeScale; }

  setConfig(config: SimulationConfig): void { this.config = config; this.gravityReady = false; }

  trigger(): void {
    this.particles.clear();
    this.age = 0;
    this.approachTime = 0;
    this.paused = false;
    this.phase = "approach";
    this.gravityReady = false;
  }

  togglePause(): void { this.paused = !this.paused; }

  update(dt: number): void {
    if (this.paused || this.phase === "ready") return;
    if (this.phase === "approach") {
      this.approachTime += dt;
      if (this.approachTime >= 1.45) this.explode();
      return;
    }
    this.age += dt;
    const p = this.particles;
    if (!p.count) return;
    const worldSize = Math.max(this.canvas.width, this.canvas.height) * 2;
    let removed = false;
    for (let i = p.count - 1; i >= 0; i--) {
      p.life[i] -= dt;
      if (p.life[i] <= 0 || p.x[i] < -worldSize || p.x[i] > this.canvas.width + worldSize
        || p.y[i] < -worldSize || p.y[i] > this.canvas.height + worldSize || Math.abs(p.z[i]) > worldSize) {
        p.remove(i); removed = true;
      }
    }
    if (!p.count) return;
    if (removed) this.gravityReady = false;
    if (!this.gravityReady) this.calculateAccelerations();

    // Kick-drift-kick leapfrog integration is time-reversible and keeps bound
    // orbits stable far longer than the previous Euler update.
    const halfStep = dt * .5;
    const cooling = 1 - Math.exp(-.42 * dt);
    for (let i = 0; i < p.count; i++) {
      p.vx[i] += p.ax[i] * halfStep; p.vy[i] += p.ay[i] * halfStep; p.vz[i] += p.az[i] * halfStep;
      if (p.kind[i] === 1) this.coolRadialMotion(i, cooling);
      p.x[i] += p.vx[i] * dt; p.y[i] += p.vy[i] * dt; p.z[i] += p.vz[i] * dt;
    }
    this.calculateAccelerations();
    for (let i = 0; i < p.count; i++) {
      p.vx[i] += p.ax[i] * halfStep; p.vy[i] += p.ay[i] * halfStep; p.vz[i] += p.az[i] * halfStep;
    }
  }

  getAtomPositions(): [[number, number, number], [number, number, number]] {
    const t = Math.min(1, this.approachTime / 1.45);
    const eased = t * t * (3 - 2 * t);
    const cx = this.canvas.width / 2, cy = this.canvas.height / 2;
    const span = Math.max(180, this.canvas.width * 0.32);
    return [[cx - span * (1 - eased), cy, 0], [cx + span * (1 - eased), cy, 0]];
  }

  private explode(): void {
    this.phase = "running";
    const p = this.particles, c = this.config;
    this.representedParticleCount = Math.min(c.particleCount, 2000000);
    const count = Math.min(this.representedParticleCount, p.capacity);
    const representedMass = this.representedParticleCount / count;
    const cx = this.canvas.width / 2, cy = this.canvas.height / 2;
    const entropy = c.randomness / 100;
    const spectrum = c.sizeVariation / 100;
    const volumeRadius = Math.max(45, Math.min(this.canvas.width, this.canvas.height) * .12);
    const initialScale = .014;
    const modeCount = count > 400000 ? 7 : 14;
    const modeX = new Float32Array(modeCount), modeY = new Float32Array(modeCount), modeZ = new Float32Array(modeCount);
    const modeK = new Float32Array(modeCount), modePhase = new Float32Array(modeCount);
    const axisZ = Math.random() * 1.2 - .6;
    const axisAngle = Math.random() * Math.PI * 2;
    const axisRing = Math.sqrt(1 - axisZ * axisZ);
    const axisX = Math.cos(axisAngle) * axisRing, axisY = Math.sin(axisAngle) * axisRing;
    for (let mode = 0; mode < modeCount; mode++) {
      const z = Math.random() * 2 - 1, angle = Math.random() * Math.PI * 2, ring = Math.sqrt(1 - z * z);
      modeX[mode] = Math.cos(angle) * ring; modeY[mode] = Math.sin(angle) * ring; modeZ[mode] = z;
      modeK[mode] = Math.PI * 2 * (1 + Math.pow(Math.random(), 1.8) * 5) / volumeRadius;
      modePhase[mode] = Math.random() * Math.PI * 2;
    }

    for (let i = 0; i < count; i++) {
      // A slow, dense inner population survives the launch while the outer
      // population carries most of the expansion energy.
      const coreParticle = Math.random() < .28;
      const radial = coreParticle
        ? Math.pow(Math.random(), 1.5) * volumeRadius * .46
        : Math.pow(Math.random(), .42) * volumeRadius;
      const zDirection = Math.random() * 2 - 1, angle = Math.random() * Math.PI * 2;
      const ring = Math.sqrt(1 - zDirection * zDirection);
      const bx = Math.cos(angle) * ring * radial;
      const by = Math.sin(angle) * ring * radial;
      const bz = zDirection * radial;

      let displacementX = 0, displacementY = 0, displacementZ = 0;
      const modeAmplitude = volumeRadius * (.012 + entropy * .045) / Math.sqrt(modeCount);
      for (let mode = 0; mode < modeCount; mode++) {
        const wave = Math.sin((bx * modeX[mode] + by * modeY[mode] + bz * modeZ[mode]) * modeK[mode] + modePhase[mode]);
        displacementX += modeX[mode] * wave * modeAmplitude;
        displacementY += modeY[mode] * wave * modeAmplitude;
        displacementZ += modeZ[mode] * wave * modeAmplitude;
      }
      const dx = bx + displacementX, dy = by + displacementY, dz = bz + displacementZ;
      const radius = this.sampleRadius(spectrum, false);
      const life = c.lifetime * (0.45 + Math.random() * 1.15);
      const hubble = c.impactSpeed / volumeRadius;
      const peculiarScale = hubble * .7;
      const jitter = c.impactSpeed * entropy * .008;
      let tangentX = axisY * bz - axisZ * by;
      let tangentY = axisZ * bx - axisX * bz;
      let tangentZ = axisX * by - axisY * bx;
      const tangentLength = Math.hypot(tangentX, tangentY, tangentZ) || 1;
      tangentX /= tangentLength; tangentY /= tangentLength; tangentZ /= tangentLength;
      const spinSpeed = c.impactSpeed * (.07 + .11 * Math.pow(radial / volumeRadius, .45));
      const expansionMultiplier = coreParticle ? .12 + Math.random() * .3 : .78 + Math.random() * .32;
      const gasLike = Math.random() < .38 ? 1 : 0;
      p.add(cx + dx * initialScale, cy + dy * initialScale, dz * initialScale,
        bx * hubble * expansionMultiplier + displacementX * peculiarScale + tangentX * spinSpeed + this.gaussian() * jitter,
        by * hubble * expansionMultiplier + displacementY * peculiarScale + tangentY * spinSpeed + this.gaussian() * jitter,
        bz * hubble * expansionMultiplier + displacementZ * peculiarScale + tangentZ * spinSpeed + this.gaussian() * jitter,
        radius, life, 24 + Math.random() * 30, representedMass, 0, gasLike);
    }
    this.gravityReady = false;
  }

  private calculateAccelerations(): void {
    const p = this.particles;
    this.gravityField.build(p);
    const massNormalization = 4500 / Math.max(1, this.representedParticleCount);
    const gravityRampTime = Math.min(1, this.age / 4);
    const gravityRamp = .04 + .96 * gravityRampTime * gravityRampTime * (3 - 2 * gravityRampTime);
    const gravity = this.config.gravity * this.config.gravity * .0022 * massNormalization * gravityRamp;
    for (let i = 0; i < p.count; i++) this.gravityField.writeAcceleration(i, p, gravity);
    this.gravityReady = true;
  }

  private coolRadialMotion(index: number, cooling: number): void {
    const p = this.particles;
    const ax = p.ax[index], ay = p.ay[index], az = p.az[index];
    const acceleration2 = ax * ax + ay * ay + az * az;
    if (acceleration2 < .0001) return;
    const radialVelocity = (p.vx[index] * ax + p.vy[index] * ay + p.vz[index] * az) / acceleration2;
    p.vx[index] -= ax * radialVelocity * cooling;
    p.vy[index] -= ay * radialVelocity * cooling;
    p.vz[index] -= az * radialVelocity * cooling;
  }

  private gaussian(): number {
    const u = Math.max(Number.EPSILON, Math.random());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * Math.random());
  }

  private sampleRadius(spectrum: number, isAnchor: boolean): number {
    if (isAnchor) return 7 + spectrum * (5 + Math.random() * 4);
    const roll = Math.random();
    const giantChance = .00015 + spectrum * .00045;
    const stellarChance = .003 + spectrum * .009;
    if (roll < giantChance) {
      // Rare bright objects affect appearance, but no longer receive privileged
      // gravitational mass.
      return 4 + Math.pow(Math.random(), .55) * 8 * spectrum;
    }
    if (roll < giantChance + stellarChance) {
      return 1.8 + Math.pow(Math.random(), 1.6) * 3.8 * spectrum;
    }

    // The bulk is log-normally distributed dust. This creates much more mass
    // contrast than a linear random range without filling the screen with dots.
    const dust = .9 * Math.exp(this.gaussian() * .34 * spectrum);
    return Math.max(.22, Math.min(2.2, dust));
  }
}
