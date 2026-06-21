import { Simulation } from "./Simulation.js";
import { OrbitCamera } from "./OrbitCamera.js";

export class Renderer {
  private readonly ctx: CanvasRenderingContext2D;
  readonly camera: OrbitCamera;
  private readonly glowSprite: HTMLCanvasElement;
  private readonly screenX = new Float32Array(250000);
  private readonly screenY = new Float32Array(250000);
  private readonly screenRadius = new Float32Array(250000);
  private readonly visible = new Uint8Array(250000);
  private selectedParticle = -1;
  private renderedStride = 1;
  private renderedCount = 0;
  private lastSimulation: Simulation | null = null;
  private focusX = 0;
  private focusY = 0;
  private focusZ = 0;
  private lastTapTime = 0;
  private lastTapX = 0;
  private lastTapY = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d", { alpha: false })!;
    this.camera = new OrbitCamera(canvas);
    this.glowSprite = document.createElement("canvas");
    this.glowSprite.width = this.glowSprite.height = 64;
    const glowContext = this.glowSprite.getContext("2d")!;
    const glow = glowContext.createRadialGradient(32, 32, 0, 32, 32, 32);
    glow.addColorStop(0, "rgba(255,250,224,.9)");
    glow.addColorStop(.08, "rgba(255,205,126,.64)");
    glow.addColorStop(.32, "rgba(255,145,70,.2)");
    glow.addColorStop(1, "rgba(255,110,50,0)");
    glowContext.fillStyle = glow;
    glowContext.fillRect(0, 0, 64, 64);
    canvas.addEventListener("dblclick", event => this.selectNearest(event.clientX, event.clientY));
    canvas.addEventListener("pointerup", event => {
      if (event.pointerType !== "touch") return;
      const now = performance.now();
      if (now - this.lastTapTime < 320 && Math.hypot(event.clientX - this.lastTapX, event.clientY - this.lastTapY) < 24) {
        this.selectNearest(event.clientX, event.clientY);
        this.lastTapTime = 0;
      } else {
        this.lastTapTime = now; this.lastTapX = event.clientX; this.lastTapY = event.clientY;
      }
    });
  }

  resize(): void {
    const dpr = Math.min(devicePixelRatio, 1.5);
    this.canvas.width = innerWidth * dpr;
    this.canvas.height = innerHeight * dpr;
    this.canvas.style.width = `${innerWidth}px`;
    this.canvas.style.height = `${innerHeight}px`;
  }

  render(sim: Simulation): void {
    const { ctx, canvas } = this;
    this.lastSimulation = sim;
    if (sim.phase !== "running" || this.selectedParticle >= sim.particles.count) this.selectedParticle = -1;
    if (this.selectedParticle >= 0) {
      this.focusX = sim.particles.x[this.selectedParticle];
      this.focusY = sim.particles.y[this.selectedParticle];
      this.focusZ = sim.particles.z[this.selectedParticle];
    } else {
      this.focusX = canvas.width / 2; this.focusY = canvas.height / 2; this.focusZ = 0;
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.filter = "none";
    ctx.fillStyle = "#05070d";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    this.camera.prepare();
    if (sim.phase === "approach") this.drawAtoms(sim);
    if (sim.phase === "running") this.drawParticles(sim);
  }

  private drawAtoms(sim: Simulation): void {
    const ctx = this.ctx;
    ctx.globalCompositeOperation = "lighter";
    for (const [x, y, z] of sim.getAtomPositions()) {
      if (!this.camera.project(x, y, z, this.focusX, this.focusY, this.focusZ)) continue;
      const px = this.camera.projectedX, py = this.camera.projectedY;
      const radius = 44 * this.camera.projectedScale;
      const gradient = ctx.createRadialGradient(px, py, 0, px, py, radius);
      gradient.addColorStop(0, "rgba(255,245,218,1)");
      gradient.addColorStop(.12, "rgba(255,183,92,.9)");
      gradient.addColorStop(.45, "rgba(239,91,44,.18)");
      gradient.addColorStop(1, "rgba(239,91,44,0)");
      ctx.fillStyle = gradient; ctx.beginPath(); ctx.arc(px, py, radius, 0, Math.PI * 2); ctx.fill();
    }
  }

  private drawParticles(sim: Simulation): void {
    const ctx = this.ctx, p = sim.particles;
    const stride = Math.max(1, Math.ceil(p.count / 75000));
    this.renderedStride = stride;
    this.renderedCount = p.count;
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < p.count; i += stride) {
      const x3d = p.x[i];
      const y3d = p.y[i];
      const z3d = p.z[i];
      if (!this.camera.project(x3d, y3d, z3d, this.focusX, this.focusY, this.focusZ)) {
        this.visible[i] = 0; continue;
      }
      const x = this.camera.projectedX, y = this.camera.projectedY;
      this.visible[i] = x > -10 && x < this.canvas.width + 10 && y > -10 && y < this.canvas.height + 10 ? 1 : 0;
      this.screenX[i] = x; this.screenY[i] = y;
      this.screenRadius[i] = Math.max(.55, p.radius[i] * this.camera.projectedScale);
    }

    const colors = sim.age < 5
      ? ["rgba(255,104,54,.7)", "rgba(255,145,66,.74)", "rgba(255,194,111,.8)", "rgba(255,248,222,.9)"]
      : ["rgba(111,151,255,.58)", "rgba(255,145,82,.66)", "rgba(183,208,255,.7)", "rgba(255,236,190,.8)"];
    for (let bucket = 0; bucket < 4; bucket++) {
      ctx.fillStyle = colors[bucket];
      ctx.beginPath();
      for (let i = 0; i < p.count; i += stride) {
        if ((Math.floor(i / stride) & 3) !== bucket) continue;
        if (!this.visible[i]) continue;
        const r = this.screenRadius[i];
        ctx.rect(this.screenX[i] - r * .5, this.screenY[i] - r * .5, Math.max(1, r), Math.max(1, r));
      }
      ctx.fill();
    }

    // A cached radial-gradient texture gives stars a smooth aura without the
    // expense and muddy edge of a live Canvas blur.
    for (let i = 0; i < p.count; i += stride) {
      if (!this.visible[i] || p.radius[i] < 3) continue;
      const diameter = this.screenRadius[i] * 7;
      ctx.drawImage(this.glowSprite, this.screenX[i] - diameter * .5, this.screenY[i] - diameter * .5, diameter, diameter);
    }
  }

  private selectNearest(clientX: number, clientY: number): void {
    const sim = this.lastSimulation;
    if (!sim || sim.phase !== "running") return;
    const rect = this.canvas.getBoundingClientRect();
    const x = (clientX - rect.left) * this.canvas.width / rect.width;
    const y = (clientY - rect.top) * this.canvas.height / rect.height;
    const maxDistance2 = Math.pow(34 * this.canvas.width / rect.width, 2);
    let nearest = -1, nearestScore = maxDistance2;
    for (let i = 0; i < this.renderedCount; i += this.renderedStride) {
      if (!this.visible[i]) continue;
      const dx = this.screenX[i] - x, dy = this.screenY[i] - y;
      const score = (dx * dx + dy * dy) / Math.max(1, this.screenRadius[i]);
      if (score < nearestScore) { nearestScore = score; nearest = i; }
    }
    if (nearest >= 0) {
      this.selectedParticle = nearest;
      this.camera.panX = 0; this.camera.panY = 0;
    }
  }
}
