import type { SimulationConfig } from "./config.js";
import { OrbitCamera } from "./OrbitCamera.js";
import { GPU_SHADER, GRID_SHADER, POISSON_SHADER, POST_SHADER, RENDER_SHADER, PICK_SHADER } from "./gpuShaders.js";

type Phase = "ready" | "approach" | "running";

export class WebGPUUniverse {
  static readonly MAX_PARTICLES = 2_000_000;
  static readonly GRID_SIDE = 32;
  static readonly GRID_CELLS = 32 * 32 * 32;
  private static readonly PARTICLE_BYTES = 32;
  private static readonly ACCELERATION_BYTES = 16;
  private static readonly UNIFORM_BYTES = 96;

  readonly camera: OrbitCamera;
  phase: Phase = "ready";
  paused = false;
  age = 0;
  private approachTime = 0;
  private physicsAccumulator = 0;
  private config: SimulationConfig;
  private count = 0;
  private needsInitialization = false;

  private constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly format: string,
    config: SimulationConfig,
  ) {
    this.config = config;
    this.camera = new OrbitCamera(canvas);
    canvas.addEventListener("dblclick", event => this.requestPick(event.clientX, event.clientY));
    canvas.addEventListener("pointerup", event => {
      if (event.pointerType !== "touch") return;
      const now = performance.now();
      if (now - this.lastTapTime < 320 && Math.hypot(event.clientX - this.lastTapX, event.clientY - this.lastTapY) < 24) {
        this.requestPick(event.clientX, event.clientY); this.lastTapTime = 0;
      } else {
        this.lastTapTime = now; this.lastTapX = event.clientX; this.lastTapY = event.clientY;
      }
    });
  }

  private paramsBuffer!: GPUBuffer;
  private particleBuffer!: GPUBuffer;
  private accelerationBuffer!: GPUBuffer;
  private massBuffer!: GPUBuffer;
  private potentialA!: GPUBuffer;
  private potentialB!: GPUBuffer;
  private initializePipeline!: GPUComputePipeline;
  private prePipeline!: GPUComputePipeline;
  private clearPipeline!: GPUComputePipeline;
  private depositPipeline!: GPUComputePipeline;
  private jacobiPipeline!: GPUComputePipeline;
  private postPipeline!: GPUComputePipeline;
  private particlePipeline!: GPURenderPipeline;
  private atomPipeline!: GPURenderPipeline;
  private pickPipeline!: GPUComputePipeline;
  private initializeGroup!: GPUBindGroup;
  private preGroup!: GPUBindGroup;
  private clearGroup!: GPUBindGroup;
  private depositGroup!: GPUBindGroup;
  private jacobiAB!: GPUBindGroup;
  private jacobiBA!: GPUBindGroup;
  private postGroup!: GPUBindGroup;
  private particleRenderGroup!: GPUBindGroup;
  private atomRenderGroup!: GPUBindGroup;
  private pickGroup!: GPUBindGroup;
  private pickParams!: GPUBuffer;
  private pickResult!: GPUBuffer;
  private pickReadback!: GPUBuffer;
  private pendingPick: [number, number] | null = null;
  private pickInFlight = false;
  private focusId = 0xffffffff;
  private lastTapTime = 0;
  private lastTapX = 0;
  private lastTapY = 0;

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
    this.focusId = 0xffffffff;
  }

  togglePause(): void { this.paused = !this.paused; }

  resize(): void {
    const dpr = Math.min(devicePixelRatio, 1.5);
    this.canvas.width = Math.max(1, Math.floor(innerWidth * dpr));
    this.canvas.height = Math.max(1, Math.floor(innerHeight * dpr));
    this.canvas.style.width = `${innerWidth}px`;
    this.canvas.style.height = `${innerHeight}px`;
    this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
  }

  frame(realDt: number): void {
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
        while (this.physicsAccumulator >= step && steps < 4) {
          this.age += step;
          this.physicsAccumulator -= step;
          steps++;
        }
      }
    }

    this.writeParams(1 / 60);
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
    const picking = this.encodePick(encoder);
    this.encodeRender(encoder);
    this.device.queue.submit([encoder.finish()]);
    if (picking) void this.resolvePick();
  }

  private initializeResources(): void {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.paramsBuffer = this.device.createBuffer({ size: WebGPUUniverse.UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.particleBuffer = this.device.createBuffer({ size: WebGPUUniverse.MAX_PARTICLES * WebGPUUniverse.PARTICLE_BYTES, usage });
    this.accelerationBuffer = this.device.createBuffer({ size: WebGPUUniverse.MAX_PARTICLES * WebGPUUniverse.ACCELERATION_BYTES, usage });
    this.massBuffer = this.device.createBuffer({ size: WebGPUUniverse.GRID_CELLS * 4, usage });
    this.potentialA = this.device.createBuffer({ size: WebGPUUniverse.GRID_CELLS * 4, usage });
    this.potentialB = this.device.createBuffer({ size: WebGPUUniverse.GRID_CELLS * 4, usage });
    this.pickParams = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pickResult = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.pickReadback = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const particleModule = this.device.createShaderModule({ code: GPU_SHADER });
    const gridModule = this.device.createShaderModule({ code: GRID_SHADER });
    const poissonModule = this.device.createShaderModule({ code: POISSON_SHADER });
    const postModule = this.device.createShaderModule({ code: POST_SHADER });
    const renderModule = this.device.createShaderModule({ code: RENDER_SHADER });
    const pickModule = this.device.createShaderModule({ code: PICK_SHADER });
    this.initializePipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: particleModule, entryPoint: "initialize" } });
    this.prePipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: particleModule, entryPoint: "preIntegrate" } });
    this.clearPipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: gridModule, entryPoint: "clearMass" } });
    this.depositPipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: gridModule, entryPoint: "deposit" } });
    this.jacobiPipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: poissonModule, entryPoint: "jacobi" } });
    this.postPipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: postModule, entryPoint: "postIntegrate" } });
    this.pickPipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: pickModule, entryPoint: "pickNearest" } });

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
      { binding: 0, resource: params }, { binding: 1, resource: particles }, { binding: 2, resource: accelerations }, { binding: 3, resource: { buffer: this.potentialA } },
    ] });
    this.particleRenderGroup = this.device.createBindGroup({ layout: this.particlePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: params }, { binding: 1, resource: particles }, { binding: 2, resource: accelerations },
    ] });
    this.atomRenderGroup = this.device.createBindGroup({ layout: this.atomPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: params },
    ] });
    this.pickGroup = this.device.createBindGroup({ layout: this.pickPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: params }, { binding: 1, resource: particles }, { binding: 2, resource: accelerations },
      { binding: 3, resource: { buffer: this.pickParams } }, { binding: 4, resource: { buffer: this.pickResult } },
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
    for (let iteration = 0; iteration < 12; iteration++) {
      pass.setBindGroup(0, iteration % 2 === 0 ? this.jacobiAB : this.jacobiBA);
      pass.dispatchWorkgroups(gridGroups);
    }
    pass.setPipeline(this.postPipeline); pass.setBindGroup(0, this.postGroup); pass.dispatchWorkgroups(particleGroups);
    pass.end();
  }

  private encodeRender(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: { r: .004, g: .006, b: .012, a: 1 }, loadOp: "clear", storeOp: "store" }],
    });
    if (this.phase === "approach") {
      pass.setPipeline(this.atomPipeline); pass.setBindGroup(0, this.atomRenderGroup); pass.draw(6, 2);
    } else if (this.phase === "running" && this.count > 0) {
      // Physics always evolves the complete population. Rendering more than
      // this produces little additional visible density but substantial vertex
      // and blending pressure on integrated GPUs.
      pass.setPipeline(this.particlePipeline); pass.setBindGroup(0, this.particleRenderGroup); pass.draw(6, Math.min(this.count, 350000));
    }
    pass.end();
  }

  private requestPick(clientX: number, clientY: number): void {
    if (this.phase !== "running") return;
    const rect = this.canvas.getBoundingClientRect();
    this.pendingPick = [
      (clientX - rect.left) * this.canvas.width / rect.width,
      (clientY - rect.top) * this.canvas.height / rect.height,
    ];
  }

  private encodePick(encoder: GPUCommandEncoder): boolean {
    if (!this.pendingPick || this.pickInFlight || this.count === 0) return false;
    const data = new Float32Array([this.pendingPick[0], this.pendingPick[1], 64 * Math.min(devicePixelRatio, 1.5), 0]);
    this.device.queue.writeBuffer(this.pickParams, 0, data);
    this.device.queue.writeBuffer(this.pickResult, 0, new Uint32Array([0xffffffff]));
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pickPipeline); pass.setBindGroup(0, this.pickGroup);
    pass.dispatchWorkgroups(Math.ceil(this.count / 256)); pass.end();
    encoder.copyBufferToBuffer(this.pickResult, 0, this.pickReadback, 0, 4);
    this.pendingPick = null;
    this.pickInFlight = true;
    return true;
  }

  private async resolvePick(): Promise<void> {
    try {
      await this.pickReadback.mapAsync(GPUMapMode.READ);
      const packed = new Uint32Array(this.pickReadback.getMappedRange())[0];
      if (packed !== 0xffffffff) {
        this.focusId = packed & 0x1fffff;
        this.camera.panX = 0; this.camera.panY = 0;
      } else {
        this.focusId = 0xffffffff;
      }
      this.pickReadback.unmap();
    } finally {
      this.pickInFlight = false;
    }
  }

  private writeParams(dt: number): void {
    const buffer = new ArrayBuffer(WebGPUUniverse.UNIFORM_BYTES);
    const view = new DataView(buffer);
    view.setUint32(0, this.count, true);
    view.setUint32(4, this.phase === "running" ? 2 : this.phase === "approach" ? 1 : 0, true);
    view.setUint32(8, WebGPUUniverse.GRID_SIDE, true);
    view.setFloat32(16, dt, true); view.setFloat32(20, this.age, true);
    const rampT = Math.min(1, this.age / 4), ramp = .04 + .96 * rampT * rampT * (3 - 2 * rampT);
    view.setFloat32(24, this.config.gravity * this.config.gravity * .0022 * 4500 / Math.max(1, this.count) * ramp, true);
    view.setFloat32(28, this.config.impactSpeed, true);
    view.setFloat32(32, this.config.randomness / 100, true);
    view.setFloat32(36, this.config.lifetime, true);
    view.setFloat32(40, this.config.sizeVariation / 100, true);
    // Keep the mesh tight around the young universe. A 96-unit initial domain
    // put every particle into the same few atomic cells and serialized deposit.
    view.setFloat32(44, Math.max(8, 8 + this.age * 92), true);
    view.setFloat32(48, this.canvas.width, true); view.setFloat32(52, this.canvas.height, true);
    view.setFloat32(56, this.camera.zoom, true); view.setFloat32(60, this.camera.yaw, true);
    view.setFloat32(64, this.camera.pitch, true);
    view.setFloat32(68, Math.min(1, this.approachTime / 1.45), true);
    view.setFloat32(72, this.camera.panX, true); view.setFloat32(76, this.camera.panY, true);
    view.setUint32(80, this.focusId, true);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, buffer);
  }
}
