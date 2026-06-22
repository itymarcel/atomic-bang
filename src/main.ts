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
const bhRing = document.querySelector<HTMLDivElement>("#bh-ring")!;
const config = readURL();

let universe: WebGPUUniverse;
try {
  universe = await WebGPUUniverse.create(canvas, config);
} catch (error) {
  console.error("WebGPU initialization failed:", error);
  document.body.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#edf0f3;font:500 16px system-ui;text-align:center;padding:24px"><div><p style="font-size:1.4em;margin:0 0 12px">WebGPU required</p><p style="color:#888;margin:0">Enable WebGPU in your browser or use Firefox Nightly / Chrome Canary.</p></div></div>`;
  throw error;
}

let writeURLTimer = 0;
const ui = new UI(config, next => {
  Object.assign(config, next);
  universe.setConfig(next);
  clearTimeout(writeURLTimer);
  writeURLTimer = setTimeout(() => writeURL(config), 300);
});

writeURL(config);

const rotateBtn = document.querySelector<HTMLButtonElement>("#rotate")!;

addEventListener("resize", () => universe.resize());
document.querySelector("#trigger")!.addEventListener("click", () => universe.trigger());
document.querySelector("#pause")!.addEventListener("click", () => universe.togglePause());
rotateBtn.addEventListener("click", () => {
  universe.camera.autoRotate = !universe.camera.autoRotate;
  rotateBtn.classList.toggle("active", universe.camera.autoRotate);
});
addEventListener("keydown", event => {
  if (event.code === "Space" && event.target === document.body) {
    event.preventDefault();
    universe.trigger();
  }
});

// Black hole placement: click-and-hold on canvas
// Hold duration → mass (50ms = tiny, 2000ms = max).
// Movement > 8px cancels placement (it's a camera drag instead).
let bhDown = false;
let bhDownTime = 0;
let bhStartX = 0;
let bhStartY = 0;
let bhMoved = false;

canvas.addEventListener("pointerdown", event => {
  bhDown = true;
  bhDownTime = event.timeStamp;
  bhStartX = event.clientX;
  bhStartY = event.clientY;
  bhMoved = false;
  bhRing.style.left = `${event.clientX}px`;
  bhRing.style.top = `${event.clientY}px`;
});

canvas.addEventListener("pointermove", event => {
  if (!bhDown) return;
  if (Math.hypot(event.clientX - bhStartX, event.clientY - bhStartY) > 8) {
    bhMoved = true;
    bhRing.style.display = "none";
  }
});

canvas.addEventListener("pointerup", event => {
  if (!bhDown) return;
  bhDown = false;
  bhRing.style.display = "none";

  if (!bhMoved && universe.phase === "running") {
    const duration = event.timeStamp - bhDownTime;
    if (duration >= 80) {
      const mass = duration / 100;
      const rect = canvas.getBoundingClientRect();
      const dpr = canvas.width / rect.width;
      const canvasX = (event.clientX - rect.left) * dpr;
      const canvasY = (event.clientY - rect.top) * dpr;
      const [wx, wy, wz] = universe.camera.unproject(canvasX, canvasY);
      universe.placeBlackHole(wx, wy, wz, mass);
    }
  }
});

canvas.addEventListener("pointercancel", () => {
  bhDown = false;
  bhRing.style.display = "none";
});

let previous = performance.now();
function frame(now: number): void {
  const dt = Math.min((now - previous) / 1000, .033);
  previous = now;
  universe.frame(dt);
  ui.update(universe.paused);

  // Animate BH sizing ring during hold
  if (bhDown && !bhMoved) {
    const duration = now - bhDownTime;
    const mass = duration / 100;
    // Mirror the GPU formula: worldRadius * zoom / dpr, scaled by 1.24 (= 2 * ringR=0.62)
    // so the CSS ring border sits exactly on the accretion ring in the shader.
    const worldRadius = 5 + mass * 2;
    const dpr = Math.min(devicePixelRatio, 1.5);
    const size = Math.max(20, 1.24 * worldRadius * universe.camera.zoom / dpr);
    const opacity = 0.3 + Math.min(mass / 30, 1) * 0.7;
    bhRing.style.width = `${size}px`;
    bhRing.style.height = `${size}px`;
    bhRing.style.opacity = String(opacity);
    bhRing.style.display = "block";
  }

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
