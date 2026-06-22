import type { SimulationConfig } from "./config.js";
import { OrbitCamera } from "./OrbitCamera.js";
import { GPU_SHADER, GRID_SHADER, POISSON_SHADER, POST_SHADER, RENDER_SHADER, BH_RENDER_SHADER } from "./gpuShaders.js";

type Phase = "ready" | "approach" | "running";

interface BlackHole { x: number; y: number; z: number; mass: number; displayMass: number; vx: number; vy: number; vz: number; }

export class WebGPUUniverse {
  static readonly MAX_PARTICLES = 2_000_000;
  static readonly GRID_SIDE = 32;
  static readonly GRID_CELLS = 32 * 32 * 32;
  private static readonly PARTICLE_BYTES = 32;
  private static readonly ACCELERATION_BYTES = 16;
  private static readonly UNIFORM_BYTES = 96;
  private static readonly BH_BUFFER_BYTES = 144; // 16 header + 8 * 16 holes
  private static readonly BH_ACCUM_BYTES = 32;   // 8 × u32 absorption counters

  readonly camera: OrbitCamera;
  phase: Phase = "ready";
  paused = false;
  age = 0;
  private approachTime = 0;
  private physicsAccumulator = 0;
  private config: SimulationConfig;
  private count = 0;
  private needsInitialization = false;
  private blackHoles: BlackHole[] = [];
  private bhAbsorptionInFlight = false;

  private constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly format: string,
    config: SimulationConfig,
  ) {
    this.config = config;
    this.camera = new OrbitCamera(canvas);
  }

  private paramsBuffer!: GPUBuffer;
  private particleBuffer!: GPUBuffer;
  private accelerationBuffer!: GPUBuffer;
  private massBuffer!: GPUBuffer;
  private potentialA!: GPUBuffer;
  private potentialB!: GPUBuffer;
  private bhBuffer!: GPUBuffer;
  private bhRenderBuffer!: GPUBuffer;
  private bhAccumBuffer!: GPUBuffer;
  private bhAccumReadback!: GPUBuffer;
  private initializePipeline!: GPUComputePipeline;
  private prePipeline!: GPUComputePipeline;
  private clearPipeline!: GPUComputePipeline;
  private depositPipeline!: GPUComputePipeline;
  private jacobiPipeline!: GPUComputePipeline;
  private postPipeline!: GPUComputePipeline;
  private particlePipeline!: GPURenderPipeline;
  private atomPipeline!: GPURenderPipeline;
  private bhPipeline!: GPURenderPipeline;
  private lensPipeline!: GPURenderPipeline;
  private initializeGroup!: GPUBindGroup;
  private preGroup!: GPUBindGroup;
  private clearGroup!: GPUBindGroup;
  private depositGroup!: GPUBindGroup;
  private jacobiAB!: GPUBindGroup;
  private jacobiBA!: GPUBindGroup;
  private postGroup!: GPUBindGroup;
  private particleRenderGroup!: GPUBindGroup;
  private atomRenderGroup!: GPUBindGroup;
  private bhRenderGroup!: GPUBindGroup;
  private lensRenderGroup!: GPUBindGroup;
  private sceneTexture!: GPUTexture;
  private sceneView!: GPUTextureView;
  private sceneSampler!: GPUSampler;

  static async create(canvas: HTMLCanvasElement, config: SimulationConfig): Promise<WebGPUUniverse> {
    const gpu = navigator.gpu;
    if (!gpu) throw new Error("WebGPU is unavailable");
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter was found");
    console.info("[atom] WebGPU adapter", adapter.info ?? "adapter info unavailable");
    const device = await adapter.requestDevice();
    const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!context) throw new Error("Could not create a WebGPU canvas context");
    const format = gpu.getPreferredCanvasFormat();
    const universe = new WebGPUUniverse(canvas, device, context, format, config);
    device.pushErrorScope("validation");
    universe.initializeResources();
    const validationError = await device.popErrorScope();
    if (validationError) throw new Error(`WebGPU validation failed: ${validationError.message}`);
    universe.resize();
    return universe;
  }

  get timeScale(): number { return this.config.timeScale; }

  setConfig(config: SimulationConfig): void { this.config = config; }

  trigger(): void {
    this.phase = "approach";
    this.paused = false;
    this.age = 0;
    this.approachTime = 0;
    this.physicsAccumulator = 0;
    this.count = 0;
    this.needsInitialization = false;
    this.blackHoles = [];
  }

  togglePause(): void { this.paused = !this.paused; }

  placeBlackHole(x: number, y: number, z: number, mass: number): void {
    if (this.blackHoles.length >= 8) this.blackHoles.shift();
    this.blackHoles.push({ x, y, z, mass, displayMass: 0, vx: 0, vy: 0, vz: 0 });
  }

  resize(): void {
    const dpr = Math.min(devicePixelRatio, 1.5);
    this.canvas.width = Math.max(1, Math.floor(innerWidth * dpr));
    this.canvas.height = Math.max(1, Math.floor(innerHeight * dpr));
    this.canvas.style.width = `${innerWidth}px`;
    this.canvas.style.height = `${innerHeight}px`;
    this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    if (this.lensPipeline) this.recreateSceneTexture();
  }

  frame(realDt: number): void {
    this.camera.tick(realDt);
    const encoder = this.device.createCommandEncoder();
    let steps = 0;
    if (!this.paused) {
      const scaledDt = realDt * this.config.timeScale;
      if (this.phase === "approach") {
        this.approachTime += scaledDt;
        if (this.approachTime >= 1.45) {
          this.phase = "running";
          this.count = Math.min(WebGPUUniverse.MAX_PARTICLES, Math.floor(this.config.particleCount));
          this.needsInitialization = true;
          this.age = 0;
        }
      } else if (this.phase === "running") {
        this.physicsAccumulator += scaledDt;
        const step = 1 / 60;
        while (this.physicsAccumulator >= step && steps < 6) {
          this.age += step;
          this.physicsAccumulator -= step;
          steps++;
        }
      }
    }

    this.writeParams(1 / 60);
    this.writeBhBuffer();
    // Reset absorption counters before physics runs this frame
    this.device.queue.writeBuffer(this.bhAccumBuffer, 0, new Uint32Array(8));
    if (this.needsInitialization) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.initializePipeline);
      pass.setBindGroup(0, this.initializeGroup);
      pass.dispatchWorkgroups(Math.ceil(this.count / 256));
      pass.end();
      this.needsInitialization = false;
      steps = Math.max(1, steps);
    }
    for (let i = 0; i < steps && this.phase === "running"; i++) this.encodePhysics(encoder);
    this.encodeRender(encoder);
    const absorb = this.copyBhAbsorption(encoder);
    this.device.queue.submit([encoder.finish()]);
    if (absorb) void this.resolveBhAbsorption();
    if (steps > 0) this.updateBlackHolePhysics(steps);
    for (const bh of this.blackHoles) {
      bh.displayMass += (bh.mass - bh.displayMass) * 0.1;
    }
  }

  private initializeResources(): void {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.paramsBuffer = this.device.createBuffer({ size: WebGPUUniverse.UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.particleBuffer = this.device.createBuffer({ size: WebGPUUniverse.MAX_PARTICLES * WebGPUUniverse.PARTICLE_BYTES, usage });
    this.accelerationBuffer = this.device.createBuffer({ size: WebGPUUniverse.MAX_PARTICLES * WebGPUUniverse.ACCELERATION_BYTES, usage });
    this.massBuffer = this.device.createBuffer({ size: WebGPUUniverse.GRID_CELLS * 4, usage });
    this.potentialA = this.device.createBuffer({ size: WebGPUUniverse.GRID_CELLS * 4, usage });
    this.potentialB = this.device.createBuffer({ size: WebGPUUniverse.GRID_CELLS * 4, usage });
    this.bhBuffer = this.device.createBuffer({ size: WebGPUUniverse.BH_BUFFER_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bhRenderBuffer = this.device.createBuffer({ size: WebGPUUniverse.BH_BUFFER_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bhAccumBuffer = this.device.createBuffer({ size: WebGPUUniverse.BH_ACCUM_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.bhAccumReadback = this.device.createBuffer({ size: WebGPUUniverse.BH_ACCUM_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.sceneSampler = this.device.createSampler({ minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });

    const particleModule = this.device.createShaderModule({ code: GPU_SHADER });
    const gridModule = this.device.createShaderModule({ code: GRID_SHADER });
    const poissonModule = this.device.createShaderModule({ code: POISSON_SHADER });
    const postModule = this.device.createShaderModule({ code: POST_SHADER });
    const renderModule = this.device.createShaderModule({ code: RENDER_SHADER });
    const bhModule = this.device.createShaderModule({ code: BH_RENDER_SHADER });
    this.initializePipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: particleModule, entryPoint: "initialize" } });
    this.prePipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: particleModule, entryPoint: "preIntegrate" } });
    this.clearPipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: gridModule, entryPoint: "clearMass" } });
    this.depositPipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: gridModule, entryPoint: "deposit" } });
    this.jacobiPipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: poissonModule, entryPoint: "jacobi" } });
    this.postPipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: postModule, entryPoint: "postIntegrate" } });

    const blend = { color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } };
    this.particlePipeline = this.device.createRenderPipeline({
      layout: "auto", vertex: { module: renderModule, entryPoint: "particleVertex" },
      fragment: { module: renderModule, entryPoint: "particleFragment", targets: [{ format: this.format, blend }] },
      primitive: { topology: "triangle-list" },
    });
    this.atomPipeline = this.device.createRenderPipeline({
      layout: "auto", vertex: { module: renderModule, entryPoint: "atomVertex" },
      fragment: { module: renderModule, entryPoint: "atomFragment", targets: [{ format: this.format, blend }] },
      primitive: { topology: "triangle-list" },
    });
    this.bhPipeline = this.device.createRenderPipeline({
      layout: "auto", vertex: { module: bhModule, entryPoint: "bhVertex" },
      fragment: { module: bhModule, entryPoint: "bhFragment", targets: [{ format: this.format, blend }] },
      primitive: { topology: "triangle-list" },
    });
    this.lensPipeline = this.device.createRenderPipeline({
      layout: "auto", vertex: { module: bhModule, entryPoint: "lensVertex" },
      fragment: { module: bhModule, entryPoint: "lensFragment", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });

    const params = { buffer: this.paramsBuffer }, particles = { buffer: this.particleBuffer }, accelerations = { buffer: this.accelerationBuffer };
    this.initializeGroup = this.device.createBindGroup({ layout: this.initializePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: params }, { binding: 1, resource: particles }, { binding: 2, resource: accelerations },
    ] });
    this.preGroup = this.device.createBindGroup({ layout: this.prePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: params }, { binding: 1, resource: particles }, { binding: 2, resource: accelerations },
    ] });
    this.clearGroup = this.device.createBindGroup({ layout: this.clearPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: params }, { binding: 3, resource: { buffer: this.massBuffer } },
    ] });
    this.depositGroup = this.device.createBindGroup({ layout: this.depositPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: params }, { binding: 1, resource: particles }, { binding: 2, resource: accelerations }, { binding: 3, resource: { buffer: this.massBuffer } },
    ] });
    this.jacobiAB = this.makeJacobiGroup(this.potentialA, this.potentialB);
    this.jacobiBA = this.makeJacobiGroup(this.potentialB, this.potentialA);
    this.postGroup = this.device.createBindGroup({ layout: this.postPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: params }, { binding: 1, resource: particles }, { binding: 2, resource: accelerations },
      { binding: 3, resource: { buffer: this.potentialA } }, { binding: 4, resource: { buffer: this.bhBuffer } },
      { binding: 5, resource: { buffer: this.bhAccumBuffer } },
    ] });
    this.particleRenderGroup = this.device.createBindGroup({ layout: this.particlePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: params }, { binding: 1, resource: particles }, { binding: 2, resource: accelerations },
    ] });
    this.atomRenderGroup = this.device.createBindGroup({ layout: this.atomPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: params },
    ] });
    this.bhRenderGroup = this.device.createBindGroup({ layout: this.bhPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: params }, { binding: 1, resource: { buffer: this.bhRenderBuffer } },
    ] });
    this.recreateSceneTexture();
  }

  private recreateSceneTexture(): void {
    if (this.sceneTexture) this.sceneTexture.destroy();
    this.sceneTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: this.format as GPUTextureFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.sceneView = this.sceneTexture.createView();
    this.lensRenderGroup = this.device.createBindGroup({ layout: this.lensPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.paramsBuffer } },
      { binding: 1, resource: { buffer: this.bhRenderBuffer } },
      { binding: 2, resource: this.sceneView },
      { binding: 3, resource: this.sceneSampler },
    ] });
  }

  private makeJacobiGroup(input: GPUBuffer, output: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({ layout: this.jacobiPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.paramsBuffer } },
      { binding: 1, resource: { buffer: this.massBuffer } },
      { binding: 2, resource: { buffer: input } },
      { binding: 3, resource: { buffer: output } },
    ] });
  }

  private encodePhysics(encoder: GPUCommandEncoder): void {
    const particleGroups = Math.ceil(this.count / 256);
    const gridGroups = Math.ceil(WebGPUUniverse.GRID_CELLS / 256);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.prePipeline); pass.setBindGroup(0, this.preGroup); pass.dispatchWorkgroups(particleGroups);
    pass.setPipeline(this.clearPipeline); pass.setBindGroup(0, this.clearGroup); pass.dispatchWorkgroups(gridGroups);
    pass.setPipeline(this.depositPipeline); pass.setBindGroup(0, this.depositGroup); pass.dispatchWorkgroups(particleGroups);
    pass.setPipeline(this.jacobiPipeline);
    for (let iteration = 0; iteration < 128; iteration++) {
      pass.setBindGroup(0, iteration % 2 === 0 ? this.jacobiAB : this.jacobiBA);
      pass.dispatchWorkgroups(gridGroups);
    }
    pass.setPipeline(this.postPipeline); pass.setBindGroup(0, this.postGroup); pass.dispatchWorkgroups(particleGroups);
    pass.end();
  }

  private encodeRender(encoder: GPUCommandEncoder): void {
    // Pass 1: render scene (particles + BH rings) into intermediate texture
    const scene = encoder.beginRenderPass({
      colorAttachments: [{ view: this.sceneView, clearValue: { r: .004, g: .006, b: .012, a: 1 }, loadOp: "clear", storeOp: "store" }],
    });
    if (this.phase === "approach") {
      scene.setPipeline(this.atomPipeline); scene.setBindGroup(0, this.atomRenderGroup); scene.draw(6, 2);
    } else if (this.phase === "running" && this.count > 0) {
      scene.setPipeline(this.particlePipeline); scene.setBindGroup(0, this.particleRenderGroup); scene.draw(6, Math.min(this.count, 350000));
    }
    if (this.blackHoles.length > 0) {
      scene.setPipeline(this.bhPipeline); scene.setBindGroup(0, this.bhRenderGroup); scene.draw(6, this.blackHoles.length);
    }
    scene.end();

    // Pass 2: gravitational lens post-process → swapchain
    const lensPass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
    });
    lensPass.setPipeline(this.lensPipeline);
    lensPass.setBindGroup(0, this.lensRenderGroup);
    lensPass.draw(6);
    lensPass.end();
  }

  private updateBlackHolePhysics(steps: number): void {
    if (this.blackHoles.length === 0) return;
    const dt = 1 / 60;
    const domain = Math.max(8, 8 + this.age * this.config.impactSpeed * 1.12);
    const bhG = domain * domain * 0.0008;
    const bhSoft2 = Math.max(4, domain * domain * 0.0002);

    for (let s = 0; s < steps; s++) {
      // Merge BHs that overlap (momentum-conserving)
      for (let i = 0; i < this.blackHoles.length - 1; i++) {
        for (let j = i + 1; j < this.blackHoles.length; j++) {
          const bi = this.blackHoles[i], bj = this.blackHoles[j];
          const dx = bj.x - bi.x, dy = bj.y - bi.y, dz = bj.z - bi.z;
          const mergeR = Math.max(2, bi.mass * 0.4) + Math.max(2, bj.mass * 0.4);
          if (dx*dx + dy*dy + dz*dz < mergeR * mergeR) {
            const m = bi.mass + bj.mass;
            bi.x = (bi.x*bi.mass + bj.x*bj.mass) / m;
            bi.y = (bi.y*bi.mass + bj.y*bj.mass) / m;
            bi.z = (bi.z*bi.mass + bj.z*bj.mass) / m;
            bi.vx = (bi.vx*bi.mass + bj.vx*bj.mass) / m;
            bi.vy = (bi.vy*bi.mass + bj.vy*bj.mass) / m;
            bi.vz = (bi.vz*bi.mass + bj.vz*bj.mass) / m;
            bi.mass = m;
            this.blackHoles.splice(j, 1);
            break;
          }
        }
      }

      // Mutual BH-BH gravity (symplectic Euler)
      for (let i = 0; i < this.blackHoles.length; i++) {
        let ax = 0, ay = 0, az = 0;
        for (let j = 0; j < this.blackHoles.length; j++) {
          if (i === j) continue;
          const bi = this.blackHoles[i], bj = this.blackHoles[j];
          const dx = bj.x - bi.x, dy = bj.y - bi.y, dz = bj.z - bi.z;
          const dist2 = Math.max(dx*dx + dy*dy + dz*dz, bhSoft2);
          const dist = Math.sqrt(dist2);
          const f = bj.mass * bhG / dist2;
          ax += (dx / dist) * f;
          ay += (dy / dist) * f;
          az += (dz / dist) * f;
        }
        const bh = this.blackHoles[i];
        bh.vx += ax * dt; bh.vy += ay * dt; bh.vz += az * dt;
      }
      for (const bh of this.blackHoles) {
        bh.x += bh.vx * dt; bh.y += bh.vy * dt; bh.z += bh.vz * dt;
      }
    }
  }

  private copyBhAbsorption(encoder: GPUCommandEncoder): boolean {
    if (this.bhAbsorptionInFlight || this.blackHoles.length === 0 || this.phase !== "running") return false;
    encoder.copyBufferToBuffer(this.bhAccumBuffer, 0, this.bhAccumReadback, 0, WebGPUUniverse.BH_ACCUM_BYTES);
    this.bhAbsorptionInFlight = true;
    return true;
  }

  private async resolveBhAbsorption(): Promise<void> {
    try {
      await this.bhAccumReadback.mapAsync(GPUMapMode.READ);
      const counts = new Uint32Array(this.bhAccumReadback.getMappedRange());
      for (let k = 0; k < this.blackHoles.length; k++) {
        this.blackHoles[k].mass += counts[k] * 0.001;
      }
      this.bhAccumReadback.unmap();
    } finally {
      this.bhAbsorptionInFlight = false;
    }
  }

  private writeBhBuffer(): void {
    const n = this.blackHoles.length;
    const physData = new ArrayBuffer(WebGPUUniverse.BH_BUFFER_BYTES);
    const rendData = new ArrayBuffer(WebGPUUniverse.BH_BUFFER_BYTES);
    const pv = new DataView(physData), rv = new DataView(rendData);
    pv.setUint32(0, n, true); rv.setUint32(0, n, true);
    for (let i = 0; i < n; i++) {
      const bh = this.blackHoles[i];
      const off = 16 + i * 16;
      for (const v of [pv, rv]) {
        v.setFloat32(off, bh.x, true);
        v.setFloat32(off + 4, bh.y, true);
        v.setFloat32(off + 8, bh.z, true);
      }
      pv.setFloat32(off + 12, bh.mass, true);        // physics: true mass
      rv.setFloat32(off + 12, bh.displayMass, true); // render: smooth visual mass
    }
    this.device.queue.writeBuffer(this.bhBuffer, 0, physData);
    this.device.queue.writeBuffer(this.bhRenderBuffer, 0, rendData);
  }

  private writeParams(dt: number): void {
    const buffer = new ArrayBuffer(WebGPUUniverse.UNIFORM_BYTES);
    const view = new DataView(buffer);
    view.setUint32(0, this.count, true);
    view.setUint32(4, this.phase === "running" ? 2 : this.phase === "approach" ? 1 : 0, true);
    view.setUint32(8, WebGPUUniverse.GRID_SIDE, true);
    view.setFloat32(16, dt, true); view.setFloat32(20, this.age, true);
    const rampT = Math.min(1, this.age / 3), ramp = .06 + .94 * rampT * rampT * rampT * (rampT * (6 * rampT - 15) + 10);
    view.setFloat32(24, this.config.gravity * this.config.gravity * .0022 * 4500 / Math.max(1, this.count) * ramp, true);
    view.setFloat32(28, this.config.impactSpeed, true);
    view.setFloat32(32, this.config.randomness / 100, true);
    view.setFloat32(36, this.config.lifetime, true);
    view.setFloat32(40, this.config.sizeVariation / 100, true);
    view.setFloat32(44, Math.max(8, 8 + this.age * this.config.impactSpeed * 1.12), true);
    view.setFloat32(48, this.canvas.width, true); view.setFloat32(52, this.canvas.height, true);
    view.setFloat32(56, this.camera.zoom, true); view.setFloat32(60, this.camera.yaw, true);
    view.setFloat32(64, this.camera.pitch, true);
    view.setFloat32(68, Math.min(1, this.approachTime / 1.45), true);
    view.setFloat32(72, this.camera.panX, true); view.setFloat32(76, this.camera.panY, true);
    view.setFloat32(80, this.config.angularMomentum / 100, true);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
  }
}
