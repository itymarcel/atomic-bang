# Atom Collision Universe Simulator

## Purpose

This project is an interactive browser simulation in which two particles collide and produce an expanding three-dimensional matter field. Gravity, angular momentum, density perturbations, and gas-like cooling then influence its evolution.

The intended result is visually inspired by early-universe expansion, gravitational collapse, star systems, and galaxy formation. It is not a scientifically complete Big Bang simulation.

The application has no runtime dependencies. It uses TypeScript, browser APIs, WebGPU, WGSL compute shaders, and a Canvas 2D CPU fallback.

## Running the project

Install the TypeScript compiler and build the static files:

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:4173
```

If port 4173 is occupied:

```bash
lsof -nP -iTCP:4173 -sTCP:LISTEN
kill <PID>
```

Production build:

```bash
npm run build
```

The output is written to `dist/` and can be served by any static HTTP server. WebGPU requires a secure context, but `localhost` is accepted by browsers.

## Browser engine selection

The application prefers WebGPU. At startup:

1. `main.ts` requests a high-performance WebGPU adapter.
2. `WebGPUUniverse` creates and validates every shader pipeline.
3. If WebGPU is unavailable or validation fails, the application creates the CPU `Simulation` and `Renderer` instead.

The browser console reports one of:

```text
[atom] WebGPU adapter ...
[atom] Using CPU fallback engine
```

Firefox Nightly currently performs well on Apple Silicon. Browser WebGPU implementations and GPU drivers can have materially different performance.

## Controls

- **Particle amount:** Requested matter-particle count, from 1,000 to 2,000,000.
- **Gravity:** Strength of gravitational acceleration.
- **Explosion velocity:** Initial outward velocity after impact.
- **Luminosity spectrum:** Changes rendered particle sizes and the frequency of bright objects. It does not grant those objects extra gravitational mass.
- **Mean lifetime:** Approximate particle lifetime.
- **Entropy:** Initial velocity and density randomness.
- **Overall speed:** Simulation-time multiplier from 0.1× to 4×.
- **Start / retrigger:** Clears the current universe and starts another collision.
- **Pause:** Pauses or resumes simulation time.

Camera interaction:

- Drag to orbit.
- Two-finger scroll to pan.
- Touchpad pinch to zoom.
- Shift + mouse wheel to pan horizontally.
- Double-click or double-tap a particle to focus the camera on it.
- Zoom limits are 0.03× to 50×.
- Space retriggers the collision.

## Source layout

```text
index.html                 Minimal application markup
src/main.ts                Engine selection and application lifecycle
src/config.ts              Configuration types, defaults, and slider definitions
src/UI.ts                  Slider and button bindings
src/style.css              Application layout and controls
src/OrbitCamera.ts         Orbit, pan, pinch zoom, and projection state

src/WebGPUUniverse.ts      Primary GPU simulation and rendering engine
src/gpuShaders.ts          WGSL compute, gravity, rendering, and picking shaders
src/webgpu.d.ts            Minimal local WebGPU TypeScript declarations

src/Simulation.ts          CPU fallback simulation
src/GridGravity.ts         CPU particle-mesh gravity solver
src/ParticleStore.ts       CPU structure-of-arrays particle storage
src/Renderer.ts            CPU Canvas 2D renderer
```

## WebGPU architecture

The WebGPU path simulates the requested count directly, up to two million particles.

### GPU memory

Particle state is split across two storage buffers:

- Particle buffer: position, visible radius, and velocity — 32 bytes per particle.
- Acceleration buffer: acceleration, lifetime, and gas-like state — 16 bytes per particle.

At two million particles these buffers consume approximately 96 MB. Additional memory is used by the gravity mesh, potential buffers, uniforms, and GPU implementation.

### Simulation sequence

Each fixed physics step executes these GPU passes:

1. Half velocity kick using the previous acceleration.
2. Position drift.
3. Clear the mass mesh.
4. Deposit particle mass using cloud-in-cell weights.
5. Solve the mesh potential with twelve Jacobi iterations.
6. Calculate potential gradients.
7. Trilinearly interpolate acceleration back to particles.
8. Apply the second half velocity kick.
9. Apply radial cooling to gas-like particles.

This is a kick–drift–kick leapfrog-style integrator. Gas cooling is intentionally dissipative, so the gas component is not symplectic.

### Particle-mesh gravity

The GPU uses a 32³ mesh.

Mass deposition uses cloud-in-cell interpolation. Each particle contributes fixed-point mass to eight neighboring nodes through atomic integer additions. The same deposition and interpolation algorithm is used at every particle count.

The solver approximates Poisson's equation:

```text
∇²φ = density
acceleration = -gravity × ∇φ
```

Density is divided by cell volume. This is important: omitting the volume term causes gravity to change incorrectly when the simulation domain changes size.

The mean density is removed because the mesh is periodic. A uniform periodic density should not create a preferred acceleration direction; only density contrast should produce structure.

The simulation domain starts tightly around the impact volume and expands over time. A tight initial domain improves spatial resolution and avoids sending hundreds of thousands of atomic writes into the same few mesh nodes.

### GPU rendering

Particles are rendered directly from GPU storage buffers. No particle positions are copied to JavaScript each frame.

Each rendered particle is an instanced camera-facing quad. The fragment shader creates a radial profile. Large luminous bodies receive a larger radial-gradient aura.

Physics always processes the complete requested population. Rendering is capped at 350,000 instances because drawing and blending millions of overlapping quads adds little visible density while placing substantial pressure on integrated GPUs.

### GPU focus picking

Double-click and double-tap selection use a compute pass:

1. Every live particle is projected using the current camera.
2. Particles inside the pick radius compete through an atomic minimum.
3. Distance and particle ID are packed into one integer.
4. Four bytes are copied back to JavaScript.
5. The selected particle ID becomes the camera's live focus.

Only picking requires GPU-to-CPU readback.

## Initial conditions

The collision is retained as the visual trigger, although a real Big Bang was not two objects exploding into pre-existing space.

After impact, particles begin in a very small three-dimensional volume and immediately receive outward velocity. There is no delayed birth animation or preconstructed sphere reveal.

The initial state includes:

- A slower central population that creates an overdense core.
- A faster outer population carrying most expansion energy.
- Coherent angular momentum around a common axis.
- Random velocity perturbations controlled by entropy.
- A gas-like fraction that can dissipate radial motion.
- A heavy-tailed visible luminosity distribution.

Gravity ramps from a small initial fraction to full strength over four simulated seconds. This approximates a high-energy initial state while allowing the combined central mass to decelerate expansion smoothly.

## Why rotation and cooling are included

Pure radial expansion of collisionless matter does not naturally create spiral galaxies. Gravity can create halos, shells, and collapse, but spiral discs require angular momentum and a component capable of dissipating random kinetic energy.

The simulator therefore includes:

- Coherent differential rotation.
- Non-uniform initial density.
- Collisionless matter.
- A gas-like matter fraction.
- Gas cooling that removes motion parallel to local acceleration while preserving more tangential motion.

This is a visual subgrid approximation of gas settling into rotating structures. It is not full hydrodynamics.

## CPU fallback

The CPU fallback uses typed arrays and a 10³ adaptive particle mesh.

Its sequence is:

1. Determine particle bounds.
2. Deposit mass through cloud-in-cell interpolation.
3. Calculate the gravitational field at mesh nodes.
4. Trilinearly interpolate acceleration to particles.
5. Integrate with kick–drift–kick stepping.
6. Render a sampled population using Canvas 2D.

The CPU engine stores at most 250,000 active trajectories. Larger requested counts are represented by weighted super-particles. Total represented mass remains consistent, but this fallback does not provide two million independent trajectories.

The CPU engine exists for compatibility, not maximum performance.

## Performance characteristics

The major WebGPU costs are:

- Atomic mass deposition, particularly when matter is highly concentrated.
- Repeated Poisson iterations.
- Reading and writing particle storage buffers.
- Additive quad rendering and overdraw.

Important performance rules:

- Do not introduce particle-count thresholds that change the physical algorithm.
- Optimize dispatch organization, workgroup aggregation, mesh solving, and rendering separately.
- Keep simulation buffers on the GPU.
- Avoid regular GPU readback.
- Keep the gravity mesh close enough to the matter distribution to retain useful spatial resolution.
- Treat rendering count and physics count as separate performance concerns.

Potential future GPU optimizations:

- Workgroup-local mass accumulation before global atomic writes.
- A multigrid or FFT Poisson solver instead of fixed Jacobi iterations.
- Adaptive force-mesh resolution.
- Indirect drawing and density-aware rendering.
- Separate dark-matter and luminous-particle render passes.
- GPU timestamp queries for per-pass profiling.

## Physics limitations

The simulator is physically inspired but not a predictive cosmological model.

Not currently modeled:

- General relativity.
- A cosmological scale factor or ΛCDM expansion history.
- Proper Gaussian power-spectrum initial conditions.
- Zel'dovich or second-order Lagrangian perturbation initial conditions.
- Full gas pressure and shock hydrodynamics.
- Radiative cooling curves.
- Star formation and stellar evolution.
- Supernova or black-hole feedback.
- Chemical evolution.
- Adaptive gravitational softening.
- Halo finding or merger trees.

The largest accuracy improvement would be replacing the hand-authored impact initial conditions with a cosmological density power spectrum and comoving expansion model. The largest numerical improvement would be a GPU FFT or multigrid Poisson solver.

## Relevant references and implementations

- FastPM paper: <https://arxiv.org/abs/1603.00476>
- FastPM source: <https://github.com/fastpm/fastpm>
- COLA paper: <https://arxiv.org/abs/1301.0322>
- GADGET-2 paper: <https://arxiv.org/abs/astro-ph/0505010>
- PKDGRAV3 paper: <https://arxiv.org/abs/1609.08621>
- FlowPM paper: <https://arxiv.org/abs/2010.11847>
- JaxPM source: <https://github.com/DifferentiableUniverseInitiative/JaxPM>
- SWIFT source: <https://github.com/SWIFTSIM/SWIFT>
- REBOUND source: <https://github.com/hannorein/rebound>
- WebGPU samples: <https://github.com/webgpu/webgpu-samples>

## Development guidance

When changing physics:

1. Make the same physical algorithm apply at every count.
2. Keep visible size separate from gravitational mass unless intentionally changing the model.
3. Check dimensional scaling when the mesh or domain size changes.
4. Preserve fixed simulation timesteps.
5. Validate both expansion and long-term bound motion.
6. Test highly concentrated initial states for atomic contention.

When changing WGSL:

1. Keep uniform structures byte-for-byte aligned with `writeParams()`.
2. Remember that `vec2f` requires 8-byte alignment and `vec3f` requires 16-byte alignment in uniform structures.
3. Keep storage-buffer bindings below device limits.
4. Wrap pipeline creation in a WebGPU validation error scope.
5. Validate on an actual WebGPU browser; TypeScript compilation does not validate WGSL.

When changing controls:

1. Add the property to `SimulationConfig`.
2. Add its default to `DEFAULT_CONFIG`.
3. Add a `CONTROL_DEFINITIONS` entry.
4. Update both WebGPU uniforms and CPU behavior.
5. Retrigger when a value changes initial conditions rather than expecting it to modify existing particles.

## Current validation

- The TypeScript browser build succeeds with Bun and the configured TypeScript build.
- WebGPU pipeline creation and rendering were exercised in Firefox Nightly on macOS.
- Pipeline validation failures cause automatic CPU fallback.
- The WebGPU and CPU implementations share the same user controls, but they are separate physics implementations and should both be checked after relevant changes.
