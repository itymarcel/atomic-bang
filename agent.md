# Atom Bang — WebGPU Particle Universe Simulator

## Purpose

An interactive browser simulation of an expanding particle universe. Two symbolic atoms approach and collide, then a massive particle field evolves under gravity, angular momentum, and density perturbations. The user can place black holes that gravitationally attract particles, absorb them, merge with each other, and grow over time.

The visual intent is early-universe expansion, gravitational collapse, star systems, and emergent large-scale structure. It is not a scientifically complete cosmological model.

No runtime dependencies. TypeScript + WebGPU + WGSL only. WebGPU is required — there is no CPU fallback.

## Running the project

```bash
npm install
npm run dev
```

Open `http://localhost:4173`. Production build writes to `dist/`.

WebGPU requires a secure context; `localhost` qualifies. Firefox Nightly and Chrome/Edge with WebGPU enabled are the supported browsers.

## Source layout

```text
index.html              Minimal markup, panel structure, BH ring overlay
src/main.ts             App lifecycle, event handlers, BH placement, frame loop
src/config.ts           SimulationConfig types, defaults, and slider definitions
src/UI.ts               Slider and button bindings, setValues() for programmatic updates
src/style.css           Layout, floating panel, stats, info box, BH ring
src/OrbitCamera.ts      Orbit / pan / pinch-zoom / auto-rotate / unproject
src/WebGPUUniverse.ts   GPU simulation engine, BH physics, render pipeline
src/gpuShaders.ts       All WGSL shaders (compute + render + lens)
src/webgpu.d.ts         Minimal local WebGPU TypeScript type stubs
```

## Controls

**Sliders (all persisted in URL query string):**
- Particle count — 1,000 to 2,000,000
- Gravity — strength of the particle-mesh gravitational field
- Explosion velocity — initial outward speed after impact (also sets velocity dispersion)
- Luminosity spectrum — distribution of visible particle sizes
- Mean lifetime — approximate particle lifetime before fading
- Entropy — initial velocity and density randomness
- Angular momentum — coherent spin speed around a shared axis
- Overall speed — simulation-time multiplier 0.1× to 4×
- Collision energy loss — fraction of relative velocity dissipated per step (0.00–1.00); only active when collision is toggled on

**Buttons:**
- Start / retrigger — clears and restarts the universe
- Pause (Ⅱ) — freezes simulation time
- Auto-rotate (↻) — smooth camera orbit around current view position, at 1/3 normal orbit speed
- Collision physics — toggles grid-based momentum-conserving particle collision; the energy-loss slider is disabled (greyed) when off

**Black hole placement:**
- Click and hold on the canvas — a growing orange ring preview appears
- Hold duration determines initial mass (100 ms per mass unit, no cap)
- Release places the black hole at the world-space position under the cursor
- Dragging more than 8 px cancels placement (treats the gesture as a camera drag)
- Up to 8 black holes; oldest is evicted when the limit is reached

**Camera:**
- Drag to orbit
- Two-finger scroll / shift+wheel to pan
- Touchpad pinch or ctrl+wheel to zoom (0.03× – 50×)
- Space bar retriggers

## WebGPU architecture

### GPU buffers

| Buffer | Contents | Size |
|---|---|---|
| particleBuffer | position (xyz + visual radius), velocity (xyz) | 32 B × N |
| accelerationBuffer | acceleration (xyz) + lifetime/gas flag (w) | 16 B × N |
| paramsBuffer | uniform params struct | 96 B |
| massBuffer | atomic u32 grid mass (32³) | 128 KB |
| momentumBuffer | atomic i32 grid momentum, 3 components per cell (32³) | 384 KB |
| potentialA/B | f32 Poisson potential ping-pong (32³) | 128 KB each |
| gradientBuffer | precomputed potential gradient (xyz) per mesh cell | 512 KB |
| bhBuffer | black hole physics state (count + 8 × vec4) | 144 B |
| bhRenderBuffer | black hole display state (smoothed visual mass) | 144 B |
| bhAccumBuffer | atomic u32 absorption counters (8 slots) | 32 B |
| bhAccumReadback | MAP_READ copy of bhAccumBuffer | 32 B |
| sceneTexture | intermediate RGBA render target for lens pass | viewport size |

### Physics sequence (per fixed step, 1/60 s)

1. **preIntegrate** — half velocity kick from previous acceleration
2. **clearMass** — zero the 32³ mass grid and all 3 × 32³ momentum components atomically
3. **deposit** — cloud-in-cell mass and momentum deposition; each particle distributes `w = cicWeight × starMass` to each of 8 neighbouring cells, accumulating `w` into `massBuffer` and `w × velocity` into `momentumBuffer` (fixed-point, ×256 scale for i32 atomics)
4. **jacobiPair × 64** — 128 Jacobi relaxation iterations solving ∇²φ = density on the 32³ grid; each dispatch computes two iterations from a workgroup-local tile with a two-cell halo
5. **calculateGradient** — computes the central-difference potential gradient once at each of the 32³ mesh cells
6. **postIntegrate** — trilinearly interpolates the cached mesh gradients → particle acceleration; applies BH gravity (1/r² with softening); checks absorption radius per BH; second half velocity kick; then if `params.collision > 0`, reads back the deposited mass and momentum from all 8 CIC-neighbour cells (same weights as deposit) to reconstruct a smooth interpolated mean velocity field, and blends the particle's velocity toward that mean by `params.collision` (the energy-loss fraction)

### Collision physics

Grid-based, momentum-conserving. At deposit time each particle writes mass-weighted momentum (`w × v`) to the same 8 cells it deposits mass into. At postIntegrate time, the particle reads back mass and momentum from all 8 neighbours using the same CIC weights, reconstructing a continuous interpolated mean-velocity field:

```
interpMean = Σ_k ( w_k × mom_k / mass_k )  /  Σ_k w_k
velocity  += (interpMean - velocity) × params.collision
```

Reading all 8 cells (not just the nearest cell) is critical — nearest-cell lookup creates sharp discontinuities at grid boundaries and causes visible oscillation artefacts. The interpolated field is smooth and consistent with how gravity is read back from `gradientBuffer`.

`params.collision = collisionEnergyLoss / 100`. Value 0 disables the block entirely. Value 1 is fully inelastic (all relative velocity dissipated in one step). Typical useful range: 0.02–0.15.

### Black hole physics (CPU, per frame)

After GPU physics, `updateBlackHolePhysics(steps)` runs on CPU for each physics step:
- Merge check: if two BHs overlap (sum of event horizon radii), momentum-conserving merge into one
- BH-BH mutual gravity: symplectic Euler integration, same `bhG = domain² × 0.0008` constant as particle shader, softening `max(4, domain² × 0.0002)`

Absorption readback: `bhAccumBuffer` is copied to `bhAccumReadback` and mapped asynchronously. Each absorbed particle adds `0.001` to that BH's mass. `bhAbsorptionInFlight` flag prevents overlapping readbacks.

### Visual mass smoothing

Each `BlackHole` has `mass` (true physics value) and `displayMass` (visual). Every frame:
```ts
bh.displayMass += (bh.mass - bh.displayMass) * 0.1;
```
`bhBuffer` always receives true mass. `bhRenderBuffer` receives `displayMass`. This means growth from particle absorption or BH merges animates smoothly rather than jumping.

### Render pipeline (one pass normally, two with lensing)

**Pass 1 → sceneTexture when lensing, otherwise directly to the swapchain:**
- Approach phase: two atom sprites (instanced quads)
- Running phase: up to 350,000 particle quads (additive blend), full physics count always simulated
- BH accretion rings (additive blend, reads `bhRenderBuffer`)

**Pass 2 → swapchain (gravitational lens):**
- Full-screen quad; fragment shader samples sceneTexture with per-pixel UV displacement
- Displacement formula: for each BH, `offset -= (toPixel/dist) × pixelRadius × 0.35 / max(normDist², 0.04)` in normalised BH-radius coordinates (1/r² falloff)
- Creates dark interior (extreme deflection samples empty space), Einstein-ring-like warping at the accretion ring, and gentle field warp beyond it
- Runs only when at least one BH exists; zero BHs bypass the intermediate texture and identity blit

Both passes use `layout: "auto"` bind groups. Each pipeline has its own bind group because auto layouts are not cross-compatible even for identical bindings.

### Particle-mesh gravity

32³ grid. Domain starts tight around the initial impact volume and expands as `8 + age × impactSpeed × 1.12` world units. Jacobi solves ∇²φ = density with 128 iterations per physics step (warm-started from previous frame's potential). Two iterations are temporally blocked into each dispatch; the second iteration uses a radius-two halo, so its result is identical to global ping-pong Jacobi rather than an approximation at workgroup boundaries.

Mass deposition weights each particle's CIC contribution by `pow(position.w, 1.6)` — the visual radius raised to the 1.6 power, floored at 0.4. This means gravitational mass is correlated with visual size but amplified superlinearly: a star 5× visually larger is ~14× heavier.

Mean density is removed before solving (periodic boundary condition; uniform background should not generate net force).

## Params uniform layout

The `Params` struct must stay byte-for-byte aligned with `writeParams()` in `WebGPUUniverse.ts`:

| Offset | Field | Notes |
|---|---|---|
| 0 | count (u32) | |
| 4 | phase (u32) | 0=ready, 1=approach, 2=running |
| 8 | gridSide (u32) | currently 32 |
| 12 | _pad0 | |
| 16 | dt (f32) | fixed 1/60 |
| 20 | age (f32) | simulation seconds |
| 24 | gravity (f32) | |
| 28 | explosion (f32) | impactSpeed |
| 32 | entropy (f32) | randomness/100 |
| 36 | lifetime (f32) | |
| 40 | spectrum (f32) | sizeVariation/100 |
| 44 | domain (f32) | world-space radius |
| 48 | viewport (vec2f) | canvas pixels |
| 56 | zoom (f32) | |
| 60 | yaw (f32) | |
| 64 | pitch (f32) | |
| 68 | approach (f32) | 0→1 ramp |
| 72 | pan (vec2f) | canvas pixels |
| 80 | spin (f32) | angularMomentum/100 |
| 84 | collision (f32) | collisionEnergyLoss/100 when enabled, else 0 |
| 88–95 | _pad3 (vec2f) | |

## Black hole buffer layout

`bhBuffer` and `bhRenderBuffer` share the same 144-byte layout:

| Offset | Contents |
|---|---|
| 0 | count (u32) |
| 4–15 | padding |
| 16 + k×16 | holes[k]: vec4f (x, y, z, mass) |

`bhBuffer` holds true physics mass; `bhRenderBuffer` holds smoothed `displayMass` for visual sizing and lens calculation.

## Initial conditions

Particles initialise as a uniform sphere:
- Outer population (~72%): higher radial expansion velocity, lighter spin boost
- Core population (~28%): slower radial velocity, 2× spin boost, more concentrated radially
- All particles share a coherent spin axis `normalize(0.31, 0.19, 0.931)` with a flat-ish rotation-curve tangential profile
- Gas fraction (~38%): negative lifetime flag, subject to radial cooling in postIntegrate
- Visual sizes: log-normal base distribution, with rare medium and large bright objects at configurable frequency (spectrum slider)
- Gravity ramps from near-zero to full over ~3 simulated seconds

## Development guidance

**Physics changes:**
- The same algorithm must apply at every particle count.
- `pow(radius, 1.6)` in the deposit shader is the intentional mass–size coupling. Changing the exponent changes physics balance.
- Domain scaling affects BH gravity constant (`bhG = domain² × 0.0008`) and PM normalization. Check both when touching domain.
- 128 Jacobi iterations is the current balance between gravity quality and GPU cost. The grid is 32³; 64³ was tried but hurt performance too much.

**WGSL changes:**
- Keep Params struct byte-aligned with `writeParams()`.
- `vec2f` needs 8-byte alignment, `vec3f`/`vec4f` need 16-byte alignment in uniforms.
- Each pipeline using `layout: "auto"` requires its own bind group, even if bindings are identical. Sharing bind groups across auto-layout pipelines causes GPU validation errors.
- The lens pass (pass 2) reads `bhRenderBuffer` at binding 1 and `sceneTexture` at binding 2. The BH ring pass reads `bhRenderBuffer` at binding 1. The postIntegrate compute reads `bhBuffer` (physics mass) at binding 4.
- WGSL compiles at runtime; TypeScript compilation does not validate it.
- POST_SHADER Params uses a single-line struct format; `replace_all` edits that match multi-line patterns will silently miss it. Edit the single-line form directly.

**Adding controls:**
1. Add property to `SimulationConfig` in `config.ts`
2. Add default to `DEFAULT_CONFIG`
3. Add entry to `CONTROL_DEFINITIONS` (key, label, min, max, step, format)
4. Write value in `writeParams()` at the correct byte offset
5. Retrigger if the parameter affects initial conditions
6. Add `row.dataset.key = definition.key` is already done in `UI` constructor — queries by `[data-key]` work automatically

**BH changes:**
- Max 8 black holes (hardcoded; matches `holes: array<vec4f, 8>` in WGSL and 144-byte buffer layout)
- Both `bhBuffer` and `bhRenderBuffer` must be updated in `writeBhBuffer()`
- `lensRenderGroup` holds a reference to `sceneView` — must be recreated in `recreateSceneTexture()` whenever canvas resizes

## Physics limitations

Not modelled:
- General relativity or spacetime curvature
- Cosmological scale factor or expansion history
- Proper power-spectrum initial conditions
- Full gas pressure, shocks, or hydrodynamics
- Star formation, stellar evolution, supernova feedback
- Adaptive gravitational softening
- Chemical or radiative evolution

The largest single numerical improvement would be replacing 128× Jacobi with a GPU FFT or multigrid Poisson solver, which would give spectral-quality long-range gravity at a fraction of the iteration cost.

## Relevant references

- FastPM: https://arxiv.org/abs/1603.00476
- COLA: https://arxiv.org/abs/1301.0322
- GADGET-2: https://arxiv.org/abs/astro-ph/0505010
- FlowPM: https://arxiv.org/abs/2010.11847
- WebGPU samples: https://github.com/webgpu/webgpu-samples
