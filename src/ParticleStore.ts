export class ParticleStore {
  readonly capacity: number;
  count = 0;
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly z: Float32Array;
  readonly vx: Float32Array;
  readonly vy: Float32Array;
  readonly vz: Float32Array;
  readonly ax: Float32Array;
  readonly ay: Float32Array;
  readonly az: Float32Array;
  readonly mass: Float32Array;
  readonly radius: Float32Array;
  readonly life: Float32Array;
  readonly maxLife: Float32Array;
  readonly hue: Float32Array;
  readonly birthDelay: Float32Array;
  readonly kind: Uint8Array;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.x = new Float32Array(capacity);
    this.y = new Float32Array(capacity);
    this.z = new Float32Array(capacity);
    this.vx = new Float32Array(capacity);
    this.vy = new Float32Array(capacity);
    this.vz = new Float32Array(capacity);
    this.ax = new Float32Array(capacity);
    this.ay = new Float32Array(capacity);
    this.az = new Float32Array(capacity);
    this.mass = new Float32Array(capacity);
    this.radius = new Float32Array(capacity);
    this.life = new Float32Array(capacity);
    this.maxLife = new Float32Array(capacity);
    this.hue = new Float32Array(capacity);
    this.birthDelay = new Float32Array(capacity);
    this.kind = new Uint8Array(capacity);
  }

  clear(): void { this.count = 0; }

  add(x: number, y: number, z: number, vx: number, vy: number, vz: number, radius: number, life: number, hue: number, mass = 1, birthDelay = 0, kind = 0): void {
    if (this.count >= this.capacity) return;
    const i = this.count++;
    this.x[i] = x; this.y[i] = y; this.z[i] = z;
    this.vx[i] = vx; this.vy[i] = vy; this.vz[i] = vz;
    this.ax[i] = this.ay[i] = this.az[i] = 0;
    this.radius[i] = radius;
    this.mass[i] = mass;
    this.life[i] = this.maxLife[i] = life;
    this.hue[i] = hue;
    this.birthDelay[i] = birthDelay;
    this.kind[i] = kind;
  }

  remove(i: number): void {
    const last = --this.count;
    if (i === last) return;
    this.x[i] = this.x[last]; this.y[i] = this.y[last]; this.z[i] = this.z[last];
    this.vx[i] = this.vx[last]; this.vy[i] = this.vy[last]; this.vz[i] = this.vz[last];
    this.ax[i] = this.ax[last]; this.ay[i] = this.ay[last]; this.az[i] = this.az[last];
    this.mass[i] = this.mass[last]; this.radius[i] = this.radius[last];
    this.life[i] = this.life[last]; this.maxLife[i] = this.maxLife[last];
    this.hue[i] = this.hue[last];
    this.birthDelay[i] = this.birthDelay[last]; this.kind[i] = this.kind[last];
  }
}
