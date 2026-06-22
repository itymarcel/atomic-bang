import { DEFAULT_CONFIG, CONTROL_DEFINITIONS } from "./config.js";
import { UI } from "./UI.js";
import { WebGPUUniverse } from "./WebGPUUniverse.js";

function readURL(): typeof DEFAULT_CONFIG {
  const params = new URLSearchParams(location.search);
  const config = { ...DEFAULT_CONFIG };
  for (const def of CONTROL_DEFINITIONS) {
    const raw = params.get(def.key);
    if (raw !== null) {
      const value = Number(raw);
      if (isFinite(value)) config[def.key] = Math.max(def.min, Math.min(def.max, value)) as never;
    }
  }
  return config;
}

function writeURL(config: typeof DEFAULT_CONFIG): void {
  const params = new URLSearchParams();
  for (const def of CONTROL_DEFINITIONS) params.set(def.key, String(config[def.key]));
  history.replaceState(null, "", `?${params}`);
}

const canvas = document.querySelector<HTMLCanvasElement>("#universe")!;
const config = readURL();

let universe: WebGPUUniverse;
try {
  universe = await WebGPUUniverse.create(canvas, config);
} catch (error) {
  console.error("WebGPU initialization failed:", error);
  document.body.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#edf0f3;font:500 16px system-ui;text-align:center;padding:24px"><div><p style="font-size:1.4em;margin:0 0 12px">WebGPU required</p><p style="color:#888;margin:0">Enable WebGPU in your browser or use Firefox Nightly / Chrome Canary.</p></div></div>`;
  throw error;
}

const ui = new UI(config, next => {
  Object.assign(config, next);
  universe.setConfig(next);
  writeURL(config);
});

writeURL(config);

addEventListener("resize", () => universe.resize());
document.querySelector("#trigger")!.addEventListener("click", () => universe.trigger());
document.querySelector("#pause")!.addEventListener("click", () => universe.togglePause());
addEventListener("keydown", event => {
  if (event.code === "Space" && event.target === document.body) {
    event.preventDefault();
    universe.trigger();
  }
});

let previous = performance.now();
function frame(now: number): void {
  const dt = Math.min((now - previous) / 1000, .033);
  previous = now;
  universe.frame(dt);
  ui.update(universe.paused);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
