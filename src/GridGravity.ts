import { ParticleStore } from "./ParticleStore.js";

/**
 * Adaptive particle-mesh gravity. Mass is deposited onto a 10³ lattice using
 * cloud-in-cell weights. Forces are solved at lattice points and interpolated
 * back to particles, avoiding the discontinuous cell-wide acceleration of the
 * previous low-resolution field.
 */
export class GridGravity {
  private static readonly SIDE = 10;
  private static readonly CELLS = 1000;
  private readonly mass = new Float32Array(GridGravity.CELLS);
  private readonly fieldX = new Float32Array(GridGravity.CELLS);
  private readonly fieldY = new Float32Array(GridGravity.CELLS);
  private readonly fieldZ = new Float32Array(GridGravity.CELLS);
  private readonly occupied = new Uint16Array(GridGravity.CELLS);
  private occupiedCount = 0;
  private originX = 0;
  private originY = 0;
  private originZ = 0;
  private cellSize = 1;
  private inverseCellSize = 1;

  build(p: ParticleStore): void {
    this.mass.fill(0);
    this.occupiedCount = 0;
    if (!p.count) return;

    let minX = p.x[0], maxX = p.x[0], minY = p.y[0], maxY = p.y[0], minZ = p.z[0], maxZ = p.z[0];
    for (let i = 1; i < p.count; i++) {
      const x = p.x[i], y = p.y[i], z = p.z[i];
      if (x < minX) minX = x; else if (x > maxX) maxX = x;
      if (y < minY) minY = y; else if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; else if (z > maxZ) maxZ = z;
    }

    const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 12) * 1.08;
    this.cellSize = span / (GridGravity.SIDE - 1);
    this.inverseCellSize = 1 / this.cellSize;
    this.originX = (minX + maxX - span) * .5;
    this.originY = (minY + maxY - span) * .5;
    this.originZ = (minZ + maxZ - span) * .5;

    for (let i = 0; i < p.count; i++) this.deposit(p.x[i], p.y[i], p.z[i], p.mass[i]);
    for (let i = 0; i < GridGravity.CELLS; i++) if (this.mass[i] > 0) this.occupied[this.occupiedCount++] = i;
    this.solveField();
  }

  writeAcceleration(index: number, p: ParticleStore, gravity: number): void {
    const gx = this.gridCoordinate(p.x[index], this.originX);
    const gy = this.gridCoordinate(p.y[index], this.originY);
    const gz = this.gridCoordinate(p.z[index], this.originZ);
    const ix = Math.floor(gx), iy = Math.floor(gy), iz = Math.floor(gz);
    const fx = gx - ix, fy = gy - iy, fz = gz - iz;
    let ax = 0, ay = 0, az = 0;
    for (let dz = 0; dz <= 1; dz++) {
      const wz = dz ? fz : 1 - fz;
      for (let dy = 0; dy <= 1; dy++) {
        const wyz = (dy ? fy : 1 - fy) * wz;
        for (let dx = 0; dx <= 1; dx++) {
          const weight = (dx ? fx : 1 - fx) * wyz;
          const cell = ix + dx + (iy + dy) * 10 + (iz + dz) * 100;
          ax += this.fieldX[cell] * weight; ay += this.fieldY[cell] * weight; az += this.fieldZ[cell] * weight;
        }
      }
    }
    p.ax[index] = ax * gravity; p.ay[index] = ay * gravity; p.az[index] = az * gravity;
  }

  private deposit(x: number, y: number, z: number, mass: number): void {
    const gx = this.gridCoordinate(x, this.originX), gy = this.gridCoordinate(y, this.originY), gz = this.gridCoordinate(z, this.originZ);
    const ix = Math.floor(gx), iy = Math.floor(gy), iz = Math.floor(gz);
    const fx = gx - ix, fy = gy - iy, fz = gz - iz;
    for (let dz = 0; dz <= 1; dz++) {
      const wz = dz ? fz : 1 - fz;
      for (let dy = 0; dy <= 1; dy++) {
        const wyz = (dy ? fy : 1 - fy) * wz;
        for (let dx = 0; dx <= 1; dx++) {
          const cell = ix + dx + (iy + dy) * 10 + (iz + dz) * 100;
          this.mass[cell] += mass * (dx ? fx : 1 - fx) * wyz;
        }
      }
    }
  }

  private solveField(): void {
    const side = GridGravity.SIDE;
    const softening2 = Math.max(9, this.cellSize * this.cellSize * .28);
    for (let target = 0; target < GridGravity.CELLS; target++) {
      const tx = target % side;
      const ty = Math.floor(target / side) % side;
      const tz = Math.floor(target / (side * side));
      let ax = 0, ay = 0, az = 0;
      for (let j = 0; j < this.occupiedCount; j++) {
        const source = this.occupied[j];
        if (source === target) continue;
        const sx = source % side;
        const sy = Math.floor(source / side) % side;
        const sz = Math.floor(source / (side * side));
        const dx = (sx - tx) * this.cellSize;
        const dy = (sy - ty) * this.cellSize;
        const dz = (sz - tz) * this.cellSize;
        const d2 = dx * dx + dy * dy + dz * dz + softening2;
        const force = this.mass[source] / (d2 * Math.sqrt(d2));
        ax += dx * force; ay += dy * force; az += dz * force;
      }
      this.fieldX[target] = ax; this.fieldY[target] = ay; this.fieldZ[target] = az;
    }
  }

  private gridCoordinate(value: number, origin: number): number {
    return Math.max(0, Math.min(GridGravity.SIDE - 1.000001, (value - origin) * this.inverseCellSize));
  }
}
