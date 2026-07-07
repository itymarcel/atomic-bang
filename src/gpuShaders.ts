export const GPU_SHADER = /* wgsl */ `
struct Params {
  count: u32,
  phase: u32,
  gridSide: u32,
  _pad0: u32,
  dt: f32,
  age: f32,
  gravity: f32,
  explosion: f32,
  entropy: f32,
  lifetime: f32,
  spectrum: f32,
  domain: f32,
  viewport: vec2f,
  zoom: f32,
  yaw: f32,
  pitch: f32,
  approach: f32,
  pan: vec2f,
  spin: f32,
  collision: f32,
  _pad3: vec2f,
}

struct Particle { position: vec4f, velocity: vec4f }

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> accelerations: array<vec4f>;

fn hash(value: u32) -> u32 {
  var x = value;
  x = ((x >> 16u) ^ x) * 0x45d9f3bu;
  x = ((x >> 16u) ^ x) * 0x45d9f3bu;
  return (x >> 16u) ^ x;
}

fn random(seed: u32) -> f32 { return f32(hash(seed)) / 4294967295.0; }

@compute @workgroup_size(256)
fn initialize(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let u = random(i * 11u + 1u);
  let v = random(i * 17u + 3u);
  let w = random(i * 23u + 5u);
  let core = random(i * 29u + 7u) < 0.28;
  let theta = 6.2831853 * v;
  let z = 2.0 * w - 1.0;
  let ring = sqrt(max(0.0, 1.0 - z * z));
  let direction = vec3f(cos(theta) * ring, sin(theta) * ring, z);
  let radialDistribution = select(pow(u, 0.42), pow(u, 1.5) * 0.46, core);
  let initialRadius = 1.2 * radialDistribution;
  let position = direction * initialRadius;

  let expansionMultiplier = select(0.78 + random(i * 31u + 9u) * 0.32, 0.12 + random(i * 31u + 9u) * 0.3, core);
  let spinAxis = normalize(vec3f(0.31, 0.19, 0.931));
  let tangentRaw = cross(spinAxis, direction);
  let tangent = tangentRaw / max(length(tangentRaw), 0.001);
  let coreSpinFactor = select(1.0, 2.0, core);
  let rotationProfile = 0.18 * pow(radialDistribution + 0.08, 0.5) / pow(0.08, 0.5);
  let spinSpeed = params.explosion * params.spin * rotationProfile * coreSpinFactor;
  let noise = vec3f(random(i * 37u + 11u), random(i * 41u + 13u), random(i * 43u + 17u)) * 2.0 - 1.0;
  let velocity = direction * params.explosion * expansionMultiplier + tangent * spinSpeed + noise * params.explosion * params.entropy * 0.012;

  let spectrum = params.spectrum;
  let sizeRoll = random(i * 47u + 19u);
  var radius = clamp(0.9 * exp((random(i * 53u + 23u) * 2.0 - 1.0) * 0.7 * spectrum), 0.22, 2.2);
  if (sizeRoll < 0.00015 + spectrum * 0.00045) {
    radius = 4.0 + pow(random(i * 59u + 29u), 0.55) * 8.0 * spectrum;
  } else if (sizeRoll < 0.00315 + spectrum * 0.00945) {
    radius = 1.8 + pow(random(i * 61u + 31u), 1.6) * 3.8 * spectrum;
  }
  let gas = random(i * 67u + 37u) < 0.38;
  let life = params.lifetime * (0.45 + random(i * 71u + 41u) * 1.15);
  particles[i].position = vec4f(position, radius);
  particles[i].velocity = vec4f(velocity, 0.0);
  accelerations[i] = vec4f(0.0, 0.0, 0.0, select(life, -life, gas));
}

@compute @workgroup_size(256)
fn preIntegrate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.count || abs(accelerations[i].w) <= 0.0) { return; }
  particles[i].velocity = vec4f(particles[i].velocity.xyz + accelerations[i].xyz * params.dt * 0.5, 0.0);
  particles[i].position = vec4f(particles[i].position.xyz + particles[i].velocity.xyz * params.dt, particles[i].position.w);
  let signLife = select(1.0, -1.0, accelerations[i].w < 0.0);
  let remaining = max(0.0, abs(accelerations[i].w) - params.dt);
  accelerations[i].w = remaining * signLife;
}
`;

export const GRID_SHADER = /* wgsl */ `
struct Params {
  count: u32, phase: u32, gridSide: u32, _pad0: u32,
  dt: f32, age: f32, gravity: f32, explosion: f32,
  entropy: f32, lifetime: f32, spectrum: f32, domain: f32,
  viewport: vec2f, zoom: f32, yaw: f32,
  pitch: f32, approach: f32, pan: vec2f,
  spin: f32, _pad2: f32, _pad3: vec2f,
}
struct Particle { position: vec4f, velocity: vec4f }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> accelerations: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> mass: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> mom: array<atomic<i32>>; // 3 × i32 per cell: px, py, pz

fn index3(p: vec3u) -> u32 { return p.x + p.y * params.gridSide + p.z * params.gridSide * params.gridSide; }

@compute @workgroup_size(256)
fn clearMass(@builtin(global_invocation_id) gid: vec3u) {
  let cells = params.gridSide * params.gridSide * params.gridSide;
  if (gid.x >= cells) { return; }
  atomicStore(&mass[gid.x], 0u);
  atomicStore(&mom[gid.x * 3u + 0u], 0);
  atomicStore(&mom[gid.x * 3u + 1u], 0);
  atomicStore(&mom[gid.x * 3u + 2u], 0);
}

@compute @workgroup_size(256)
fn deposit(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.count || abs(accelerations[i].w) <= 0.0) { return; }
  let sideMax = f32(params.gridSide - 1u) - 0.0001;
  let grid = clamp((particles[i].position.xyz / params.domain + 0.5) * sideMax, vec3f(0.0), vec3f(sideMax));
  let base = vec3u(floor(grid));
  let fraction = fract(grid);
  let starMass = max(0.4, pow(particles[i].position.w, 1.6)); // bigger = heavier, superlinear cheat
  let vel = particles[i].velocity.xyz;
  for (var z = 0u; z <= 1u; z++) {
    for (var y = 0u; y <= 1u; y++) {
      for (var x = 0u; x <= 1u; x++) {
        let offset = vec3u(x, y, z);
        let cell = min(base + offset, vec3u(params.gridSide - 1u));
        let weight3 = select(vec3f(1.0) - fraction, fraction, offset == vec3u(1u));
        let w = weight3.x * weight3.y * weight3.z * starMass;
        let ci = index3(cell);
        atomicAdd(&mass[ci], u32(max(0.0, w * 256.0)));
        // Deposit mass-weighted momentum as fixed-point (scale 256)
        atomicAdd(&mom[ci * 3u + 0u], i32(w * vel.x * 256.0));
        atomicAdd(&mom[ci * 3u + 1u], i32(w * vel.y * 256.0));
        atomicAdd(&mom[ci * 3u + 2u], i32(w * vel.z * 256.0));
      }
    }
  }
}
`;

export const POISSON_SHADER = /* wgsl */ `
struct Params {
  count: u32, phase: u32, gridSide: u32, _pad0: u32,
  dt: f32, age: f32, gravity: f32, explosion: f32,
  entropy: f32, lifetime: f32, spectrum: f32, domain: f32,
  viewport: vec2f, zoom: f32, yaw: f32,
  pitch: f32, approach: f32, pan: vec2f,
  spin: f32, _pad2: f32, _pad3: vec2f,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> mass: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> inputPotential: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputPotential: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputGradient: array<vec4f>;

fn index3(p: vec3u) -> u32 { return p.x + p.y * params.gridSide + p.z * params.gridSide * params.gridSide; }
fn wrap(value: i32) -> u32 {
  let side = i32(params.gridSide);
  return u32((value % side + side) % side);
}

const BLOCK_X = 8u;
const BLOCK_Y = 8u;
const BLOCK_Z = 4u;
const INPUT_X = BLOCK_X + 4u;
const INPUT_Y = BLOCK_Y + 4u;
const INPUT_Z = BLOCK_Z + 4u;
const INNER_X = BLOCK_X + 2u;
const INNER_Y = BLOCK_Y + 2u;
const INNER_Z = BLOCK_Z + 2u;

var<workgroup> inputTile: array<f32, 1152>; // 12 * 12 * 8, two-cell halo
var<workgroup> innerTile: array<f32, 600>;  // 10 * 10 * 6, one-cell halo

fn inputIndex(x: u32, y: u32, z: u32) -> u32 {
  return x + y * INPUT_X + z * INPUT_X * INPUT_Y;
}

fn innerIndex(x: u32, y: u32, z: u32) -> u32 {
  return x + y * INNER_X + z * INNER_X * INNER_Y;
}

fn densityAt(cell: vec3u, spacing: f32, cells: u32) -> f32 {
  let depositedMass = f32(atomicLoad(&mass[index3(cell)])) / 256.0;
  let meanMass = f32(params.count) / f32(cells);
  return (depositedMass - meanMass) / (spacing * spacing * spacing);
}

// Compute two mathematically identical Jacobi iterations per dispatch. The
// two-cell halo makes the second iteration independent across workgroups.
@compute @workgroup_size(8, 8, 4)
fn jacobiPair(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_id) localId: vec3u,
  @builtin(local_invocation_index) localIndex: u32,
) {
  let side = params.gridSide;
  let cells = side * side * side;
  let spacing = params.domain / f32(side - 1u);
  let origin = vec3i(workgroupId * vec3u(BLOCK_X, BLOCK_Y, BLOCK_Z));

  // Cooperatively stage the source potential, including the radius-two halo.
  for (var tileIndex = localIndex; tileIndex < INPUT_X * INPUT_Y * INPUT_Z; tileIndex += 256u) {
    let tx = tileIndex % INPUT_X;
    let ty = (tileIndex / INPUT_X) % INPUT_Y;
    let tz = tileIndex / (INPUT_X * INPUT_Y);
    let cell = vec3u(
      wrap(origin.x + i32(tx) - 2),
      wrap(origin.y + i32(ty) - 2),
      wrap(origin.z + i32(tz) - 2),
    );
    inputTile[tileIndex] = inputPotential[index3(cell)];
  }
  workgroupBarrier();

  // First Jacobi iteration for the output tile and its one-cell halo.
  for (var tileIndex = localIndex; tileIndex < INNER_X * INNER_Y * INNER_Z; tileIndex += 256u) {
    let tx = tileIndex % INNER_X;
    let ty = (tileIndex / INNER_X) % INNER_Y;
    let tz = tileIndex / (INNER_X * INNER_Y);
    let ix = tx + 1u;
    let iy = ty + 1u;
    let iz = tz + 1u;
    let cell = vec3u(
      wrap(origin.x + i32(tx) - 1),
      wrap(origin.y + i32(ty) - 1),
      wrap(origin.z + i32(tz) - 1),
    );
    let neighborSum =
      inputTile[inputIndex(ix - 1u, iy, iz)] + inputTile[inputIndex(ix + 1u, iy, iz)] +
      inputTile[inputIndex(ix, iy - 1u, iz)] + inputTile[inputIndex(ix, iy + 1u, iz)] +
      inputTile[inputIndex(ix, iy, iz - 1u)] + inputTile[inputIndex(ix, iy, iz + 1u)];
    innerTile[tileIndex] = (neighborSum - densityAt(cell, spacing, cells) * spacing * spacing) / 6.0;
  }
  workgroupBarrier();

  // Second iteration writes only this workgroup's non-overlapping output tile.
  let cell = workgroupId * vec3u(BLOCK_X, BLOCK_Y, BLOCK_Z) + localId;
  if (all(cell < vec3u(side))) {
    let ix = localId.x + 1u;
    let iy = localId.y + 1u;
    let iz = localId.z + 1u;
    let neighborSum =
      innerTile[innerIndex(ix - 1u, iy, iz)] + innerTile[innerIndex(ix + 1u, iy, iz)] +
      innerTile[innerIndex(ix, iy - 1u, iz)] + innerTile[innerIndex(ix, iy + 1u, iz)] +
      innerTile[innerIndex(ix, iy, iz - 1u)] + innerTile[innerIndex(ix, iy, iz + 1u)];
    outputPotential[index3(cell)] = (neighborSum - densityAt(cell, spacing, cells) * spacing * spacing) / 6.0;
  }
}

@compute @workgroup_size(256)
fn calculateGradient(@builtin(global_invocation_id) gid: vec3u) {
  let cells = params.gridSide * params.gridSide * params.gridSide;
  let i = gid.x;
  if (i >= cells) { return; }
  let side = params.gridSide;
  let cell = vec3u(i % side, (i / side) % side, i / (side * side));
  let spacing = params.domain / f32(side - 1u);
  let x0 = index3(vec3u(wrap(i32(cell.x) - 1), cell.y, cell.z));
  let x1 = index3(vec3u(wrap(i32(cell.x) + 1), cell.y, cell.z));
  let y0 = index3(vec3u(cell.x, wrap(i32(cell.y) - 1), cell.z));
  let y1 = index3(vec3u(cell.x, wrap(i32(cell.y) + 1), cell.z));
  let z0 = index3(vec3u(cell.x, cell.y, wrap(i32(cell.z) - 1)));
  let z1 = index3(vec3u(cell.x, cell.y, wrap(i32(cell.z) + 1)));
  let gradient = -vec3f(
    inputPotential[x1] - inputPotential[x0],
    inputPotential[y1] - inputPotential[y0],
    inputPotential[z1] - inputPotential[z0],
  ) / (2.0 * spacing);
  outputGradient[i] = vec4f(gradient, 0.0);
}
`;

export const POST_SHADER = /* wgsl */ `
struct Params {
  count: u32, phase: u32, gridSide: u32, _pad0: u32,
  dt: f32, age: f32, gravity: f32, explosion: f32,
  entropy: f32, lifetime: f32, spectrum: f32, domain: f32,
  viewport: vec2f, zoom: f32, yaw: f32,
  pitch: f32, approach: f32, pan: vec2f,
  spin: f32, collision: f32, _pad3: vec2f,
}
struct Particle { position: vec4f, velocity: vec4f }
struct BlackHoles {
  count: u32, _p0: u32, _p1: u32, _p2: u32,
  holes: array<vec4f, 8>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> accelerations: array<vec4f>;
@group(0) @binding(3) var<storage, read> gridGradients: array<vec4f>;
@group(0) @binding(4) var<uniform> bh: BlackHoles;
@group(0) @binding(5) var<storage, read_write> bhAccum: array<atomic<u32>, 8>;
@group(0) @binding(6) var<storage, read_write> gridMass: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> gridMom: array<atomic<i32>>;

fn index3(p: vec3u) -> u32 { return p.x + p.y * params.gridSide + p.z * params.gridSide * params.gridSide; }
fn wrap(value: i32) -> u32 {
  let side = i32(params.gridSide);
  return u32((value % side + side) % side);
}
@compute @workgroup_size(256)
fn postIntegrate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.count || abs(accelerations[i].w) <= 0.0) { return; }
  let sideMax = f32(params.gridSide - 1u) - 0.0001;
  let grid = clamp((particles[i].position.xyz / params.domain + 0.5) * sideMax, vec3f(0.0), vec3f(sideMax));
  let base = vec3u(floor(grid));
  let fraction = fract(grid);
  var acceleration = vec3f(0.0);
  for (var z = 0u; z <= 1u; z++) {
    for (var y = 0u; y <= 1u; y++) {
      for (var x = 0u; x <= 1u; x++) {
        let offset = vec3u(x, y, z);
        let cell = min(base + offset, vec3u(params.gridSide - 1u));
        let weight3 = select(vec3f(1.0) - fraction, fraction, offset == vec3u(1u));
        acceleration += gridGradients[index3(cell)].xyz * weight3.x * weight3.y * weight3.z;
      }
    }
  }
  acceleration *= params.gravity;

  // Black hole gravity (1/r²) + absorption. Strength is domain²-scaled.
  let bhDom2 = params.domain * params.domain;
  let bhG = bhDom2 * 0.0008;
  let bhSoft2 = max(4.0, bhDom2 * 0.0002);
  for (var k = 0u; k < bh.count; k++) {
    let hole = bh.holes[k];
    let delta = hole.xyz - particles[i].position.xyz;
    let actualDist2 = dot(delta, delta);
    // Absorb particle if inside the event horizon radius
    let absRadius = max(2.0, hole.w * 0.4);
    if (actualDist2 < absRadius * absRadius) {
      atomicAdd(&bhAccum[k], 1u);
      accelerations[i].w = 0.0;
      return;
    }
    let dist2 = max(actualDist2, bhSoft2);
    acceleration += normalize(delta) * hole.w * bhG / dist2;
  }

  var velocity = particles[i].velocity.xyz + acceleration * params.dt * 0.5;
  if (accelerations[i].w < 0.0) {
    let a2 = dot(acceleration, acceleration);
    if (a2 > 0.0001) {
      velocity -= acceleration * (dot(velocity, acceleration) / a2) * (1.0 - exp(-0.42 * params.dt));
    }
  }
  // Grid-based collision: CIC-interpolate the local mean velocity from all 8 neighbour
  // cells using the same weights used during deposit. Reading only the base cell causes
  // sharp discontinuities at cell boundaries; interpolation gives a smooth field.
  if (params.collision > 0.0) {
    var interpMean = vec3f(0.0);
    var totalW = 0.0;
    for (var cz = 0u; cz <= 1u; cz++) {
      for (var cy = 0u; cy <= 1u; cy++) {
        for (var cx = 0u; cx <= 1u; cx++) {
          let offset = vec3u(cx, cy, cz);
          let cell = min(base + offset, vec3u(params.gridSide - 1u));
          let weight3 = select(vec3f(1.0) - fraction, fraction, offset == vec3u(1u));
          let w = weight3.x * weight3.y * weight3.z;
          let ci = index3(cell);
          let cm = f32(atomicLoad(&gridMass[ci])) / 256.0;
          if (cm > 0.5) {
            let mv = vec3f(
              f32(atomicLoad(&gridMom[ci * 3u + 0u])),
              f32(atomicLoad(&gridMom[ci * 3u + 1u])),
              f32(atomicLoad(&gridMom[ci * 3u + 2u])),
            ) / (cm * 256.0);
            interpMean += mv * w;
            totalW += w;
          }
        }
      }
    }
    if (totalW > 0.0) {
      velocity += (interpMean / totalW - velocity) * params.collision;
    }

    // Collisional matter should behave more like a bound, pressure-supported gas cloud
    // than a one-way ballistic shell. Damp only outward radial drift, then apply a weak
    // halo-like restoring term so clumps remain near the orbit target.
    let radius = length(particles[i].position.xyz);
    if (radius > 0.001) {
      let radial = particles[i].position.xyz / radius;
      let outward = max(0.0, dot(velocity, radial));
      let outflowCooling = 1.0 - exp(-params.collision * params.dt * 4.5);
      velocity -= radial * outward * outflowCooling;
      velocity -= particles[i].position.xyz * params.collision * params.dt * 1.15;
    }
  }
  particles[i].velocity = vec4f(velocity, 0.0);
  accelerations[i] = vec4f(acceleration, accelerations[i].w);
}
`;

export const RENDER_SHADER = /* wgsl */ `
struct Params {
  count: u32, phase: u32, gridSide: u32, _pad0: u32,
  dt: f32, age: f32, gravity: f32, explosion: f32,
  entropy: f32, lifetime: f32, spectrum: f32, domain: f32,
  viewport: vec2f, zoom: f32, yaw: f32,
  pitch: f32, approach: f32, pan: vec2f,
  spin: f32, _pad2: f32, _pad3: vec2f,
}
struct Particle { position: vec4f, velocity: vec4f }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> accelerations: array<vec4f>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec4f,
  @location(2) aura: f32,
}

fn rotate(point: vec3f) -> vec3f {
  let cy = cos(params.yaw); let sy = sin(params.yaw);
  let cp = cos(params.pitch); let sp = sin(params.pitch);
  let x = cy * point.x - sy * point.z;
  let z = sy * point.x + cy * point.z;
  return vec3f(x, cp * point.y - sp * z, sp * point.y + cp * z);
}

@vertex
fn particleVertex(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) instanceId: u32) -> VertexOutput {
  let corners = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(-1,1), vec2f(1,-1), vec2f(1,1));
  let particle = particles[instanceId];
  let view = rotate(particle.position.xyz);
  let focal = max(params.viewport.x, params.viewport.y) * 1.15;
  let perspective = focal / max(20.0, focal + view.z * params.zoom) * params.zoom;
  let center = params.viewport * 0.5 + params.pan + view.xy * perspective;
  let aura = select(0.0, 1.0, particle.position.w > 3.0);
  let baseRadius = max(0.65, particle.position.w * perspective) * select(1.0, 3.5, aura > 0.5);
  let screen = center + corners[vertexId] * baseRadius;
  var output: VertexOutput;
  output.position = vec4f(screen.x / params.viewport.x * 2.0 - 1.0, 1.0 - screen.y / params.viewport.y * 2.0, 0.0, 1.0);
  output.uv = corners[vertexId];

  // Four color families cycling by instance ID; velocity drives dim→bright
  // within each family. Additive blending creates purple/cyan/white at overlaps.
  let speed = length(particle.velocity.xyz);
  let t = clamp(speed / max(1.0, params.explosion * 0.4), 0.0, 1.0);
  let isGas = accelerations[instanceId].w < 0.0;
  let family = instanceId % 4u;
  var rgb: vec3f;
  if (family == 0u) {
    rgb = mix(vec3f(1.0, 0.30, 0.10), vec3f(1.0, 0.80, 0.40), t);   // orange → gold
  } else if (family == 1u) {
    rgb = mix(vec3f(0.40, 0.20, 0.90), vec3f(0.60, 0.86, 1.0), t);  // violet → blue-white
  } else if (family == 2u) {
    rgb = mix(vec3f(0.10, 0.58, 0.44), vec3f(0.65, 1.0, 0.84), t);  // teal → mint-white
  } else {
    rgb = mix(vec3f(1.0, 0.78, 0.20), vec3f(1.0, 0.96, 0.82), t);   // gold → warm white
  }
  // Gas particles lean visibly green
  rgb = select(rgb, mix(rgb, vec3f(0.18, 1.0, 0.52), 0.38), isGas);
  // Hot young universe: brief orange wash that fades out by age 5s
  let earlyHeat = max(0.0, 1.0 - params.age / 5.0);
  rgb = mix(rgb, mix(rgb, vec3f(1.0, 0.52, 0.18), 0.55), earlyHeat * earlyHeat);
  // Fade in over first 1.5 sim-seconds so particles don't snap into view
  let fadeIn = smoothstep(0.0, 1.5, params.age);
  output.color = vec4f(rgb, 0.68 * fadeIn);

  output.aura = aura;
  if (abs(accelerations[instanceId].w) <= 0.0) { output.position = vec4f(2.0, 2.0, 0.0, 1.0); }
  return output;
}

@fragment
fn particleFragment(input: VertexOutput) -> @location(0) vec4f {
  let radius = length(input.uv);
  if (radius > 1.0) { discard; }
  let core = 1.0 - smoothstep(0.0, 1.0, radius);
  let alpha = select(smoothstep(1.0, 0.15, radius) * input.color.a, pow(core, 1.7) * .72, input.aura > .5);
  let auraColor = mix(vec3f(1.0, .38, .12), vec3f(1.0, .96, .72), pow(core, 2.0));
  let finalColor = select(input.color.rgb, auraColor, input.aura > .5);
  return vec4f(finalColor, alpha);
}

@vertex
fn atomVertex(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) instanceId: u32) -> VertexOutput {
  let corners = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(-1,1), vec2f(1,-1), vec2f(1,1));
  let t = params.approach * params.approach * (3.0 - 2.0 * params.approach);
  let direction = select(-1.0, 1.0, instanceId == 1u);
  let center = params.viewport * .5 + vec2f(direction * max(180.0, params.viewport.x * .32) * (1.0 - t), 0.0);
  let screen = center + corners[vertexId] * 44.0;
  var output: VertexOutput;
  output.position = vec4f(screen.x / params.viewport.x * 2.0 - 1.0, 1.0 - screen.y / params.viewport.y * 2.0, 0.0, 1.0);
  output.uv = corners[vertexId]; output.color = vec4f(1.0, .45, .16, 1.0); output.aura = 1.0;
  return output;
}

@fragment
fn atomFragment(input: VertexOutput) -> @location(0) vec4f {
  let r = length(input.uv);
  if (r > 1.0) { discard; }
  let glow = 1.0 - smoothstep(0.0, 1.0, r);
  return vec4f(mix(vec3f(1.0,.25,.05), vec3f(1.0,1.0,.85), pow(glow,3.0)), glow);
}
`;

// Separate module so its binding declarations don't pollute RENDER_SHADER's auto-layout
export const BH_RENDER_SHADER = /* wgsl */ `
struct Params {
  count: u32, phase: u32, gridSide: u32, _pad0: u32,
  dt: f32, age: f32, gravity: f32, explosion: f32,
  entropy: f32, lifetime: f32, spectrum: f32, domain: f32,
  viewport: vec2f, zoom: f32, yaw: f32,
  pitch: f32, approach: f32, pan: vec2f,
  spin: f32, _pad2: f32, _pad3: vec2f,
}
struct BlackHoles {
  count: u32, _p0: u32, _p1: u32, _p2: u32,
  holes: array<vec4f, 8>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> bh: BlackHoles;

struct BHOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

fn rotate(point: vec3f) -> vec3f {
  let cy = cos(params.yaw); let sy = sin(params.yaw);
  let cp = cos(params.pitch); let sp = sin(params.pitch);
  let x = cy * point.x - sy * point.z;
  let z = sy * point.x + cy * point.z;
  return vec3f(x, cp * point.y - sp * z, sp * point.y + cp * z);
}

@vertex
fn bhVertex(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> BHOutput {
  let corners = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(-1,1), vec2f(-1,1), vec2f(1,-1), vec2f(1,1));
  var out: BHOutput;
  out.position = vec4f(2.0, 2.0, 0.0, 1.0);
  out.uv = vec2f(0.0);
  if (ii >= bh.count) { return out; }
  let hole = bh.holes[ii];
  let view = rotate(hole.xyz);
  let focal = max(params.viewport.x, params.viewport.y) * 1.15;
  let perspective = focal / max(20.0, focal + view.z * params.zoom) * params.zoom;
  let center = params.viewport * 0.5 + params.pan + view.xy * perspective;
  // World-space radius projected into screen pixels — BH scales with zoom like a real 3D object
  let worldRadius = 5.0 + hole.w * 2.0;
  let pixelRadius = max(6.0, worldRadius * perspective);
  let screen = center + corners[vi] * pixelRadius;
  out.position = vec4f(screen.x / params.viewport.x * 2.0 - 1.0, 1.0 - screen.y / params.viewport.y * 2.0, 0.0, 1.0);
  out.uv = corners[vi];
  return out;
}

@fragment
fn bhFragment(input: BHOutput) -> @location(0) vec4f {
  let r = length(input.uv);
  if (r > 1.0) { discard; }
  let ringDist = abs(r - 0.62);
  let ring = max(0.0, 1.0 - ringDist / 0.18);
  let ringColor = mix(vec3f(1.0, 0.45, 0.08), vec3f(1.0, 0.92, 0.65), ring * ring);
  let haze = smoothstep(1.0, 0.72, r) * 0.18;
  let alpha = ring * 0.92 + haze * (1.0 - ring * 0.7);
  if (alpha < 0.005) { discard; }
  return vec4f(ringColor * ring + vec3f(0.55, 0.18, 0.0) * haze, alpha);
}

// Gravitational lens post-process — full-screen blit that warps pixels toward each BH.
// 1/r² deflection in normalised BH-radius coordinates; naturally creates dark
// event horizon (over-deflected interior samples empty space) + warp field outside.
@group(0) @binding(2) var sceneTex: texture_2d<f32>;
@group(0) @binding(3) var sceneSamp: sampler;

struct LensOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex fn lensVertex(@builtin(vertex_index) vi: u32) -> LensOut {
  let c = array<vec2f,6>(vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1),vec2f(-1,1),vec2f(1,-1),vec2f(1,1))[vi];
  return LensOut(vec4f(c,0.0,1.0), c*vec2f(0.5,-0.5)+0.5);
}

@fragment fn lensFragment(in: LensOut) -> @location(0) vec4f {
  let pixelPos = in.uv * params.viewport;
  var offset = vec2f(0.0);
  let focal = max(params.viewport.x, params.viewport.y) * 1.15;
  for (var k = 0u; k < bh.count; k++) {
    let hole = bh.holes[k];
    let view = rotate(hole.xyz);
    let perspective = focal / max(20.0, focal + view.z * params.zoom) * params.zoom;
    let bhScreen = params.viewport * 0.5 + params.pan + view.xy * perspective;
    let worldRadius = 5.0 + hole.w * 2.0;
    let pixelRadius = max(6.0, worldRadius * perspective);
    let toPixel = pixelPos - bhScreen;
    let dist = max(length(toPixel), 0.5);
    let normDist = dist / pixelRadius;
    // 1/r² deflection capped at inner horizon to avoid singularity
    let lensForce = 1.0 / max(normDist * normDist, 0.04);
    offset -= (toPixel / dist) * lensForce * pixelRadius * 0.35;
  }
  let sampleUV = clamp((pixelPos + offset) / params.viewport, vec2f(0.0), vec2f(1.0));
  return textureSample(sceneTex, sceneSamp, sampleUV);
}
`;
