import { DEFAULT_CONFIG } from "./config.js";
import { Simulation } from "./Simulation.js";
import { Renderer } from "./Renderer.js";
import { UI } from "./UI.js";
import { WebGPUUniverse } from "./WebGPUUniverse.js";

const canvas = document.querySelector<HTMLCanvasElement>("#universe")!;
const config = { ...DEFAULT_CONFIG };
let gpuUniverse: WebGPUUniverse | null = null;
let simulation: Simulation | null = null;
let renderer: Renderer | null = null;

if (navigator.gpu) {
  try {
    gpuUniverse = await WebGPUUniverse.create(canvas, config);
  } catch (error) {
    console.warn("WebGPU initialization failed; using CPU fallback.", error);
  }
}
if (!gpuUniverse) {
  console.info("[atom] Using CPU fallback engine");
  simulation = new Simulation(canvas, config);
  renderer = new Renderer(canvas);
  renderer.resize();
}

const ui = new UI(config, next => gpuUniverse ? gpuUniverse.setConfig(next) : simulation!.setConfig(next));

addEventListener("resize", () => gpuUniverse ? gpuUniverse.resize() : renderer!.resize());
document.querySelector("#trigger")!.addEventListener("click", () => gpuUniverse ? gpuUniverse.trigger() : simulation!.trigger());
document.querySelector("#pause")!.addEventListener("click", () => gpuUniverse ? gpuUniverse.togglePause() : simulation!.togglePause());
addEventListener("keydown", event => {
  if (event.code === "Space" && event.target === document.body) {
    event.preventDefault();
    if (gpuUniverse) gpuUniverse.trigger(); else simulation!.trigger();
  }
});

let previous = performance.now();
let physicsAccumulator = 0;
const physicsStep = 1 / 30;
function frame(now: number): void {
  const dt = Math.min((now - previous) / 1000, .033);
  previous = now;
  if (gpuUniverse) {
    gpuUniverse.frame(dt);
    ui.update(gpuUniverse.paused);
  } else {
    physicsAccumulator += dt * simulation!.timeScale;
    let physicsIterations = 0;
    while (physicsAccumulator >= physicsStep && physicsIterations < 4) {
      simulation!.update(physicsStep);
      physicsAccumulator -= physicsStep;
      physicsIterations++;
    }
    renderer!.render(simulation!);
    ui.update(simulation!.paused);
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
