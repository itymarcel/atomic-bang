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
  focusId: u32,
  _pad1: u32,
  _pad2: vec2f,
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
  let spin = params.explosion * (0.07 + 0.11 * pow(radialDistribution, 0.45));
  let noise = vec3f(random(i * 37u + 11u), random(i * 41u + 13u), random(i * 43u + 17u)) * 2.0 - 1.0;
  let velocity = direction * params.explosion * expansionMultiplier + tangent * spin + noise * params.explosion * params.entropy * 0.012;

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
  focusId: u32, _pad1: u32, _pad2: vec2f,
}
struct Particle { position: vec4f, velocity: vec4f }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> accelerations: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> mass: array<atomic<u32>>;

fn index3(p: vec3u) -> u32 { return p.x + p.y * params.gridSide + p.z * params.gridSide * params.gridSide; }

@compute @workgroup_size(256)
fn clearMass(@builtin(global_invocation_id) gid: vec3u) {
  let cells = params.gridSide * params.gridSide * params.gridSide;
  if (gid.x < cells) { atomicStore(&mass[gid.x], 0u); }
}

@compute @workgroup_size(256)
fn deposit(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.count || abs(accelerations[i].w) <= 0.0) { return; }
  let sideMax = f32(params.gridSide - 1u) - 0.0001;
  let grid = clamp((particles[i].position.xyz / params.domain + 0.5) * sideMax, vec3f(0.0), vec3f(sideMax));
  let base = vec3u(floor(grid));
  let fraction = fract(grid);
  for (var z = 0u; z <= 1u; z++) {
    for (var y = 0u; y <= 1u; y++) {
      for (var x = 0u; x <= 1u; x++) {
        let offset = vec3u(x, y, z);
        let cell = min(base + offset, vec3u(params.gridSide - 1u));
        let weight3 = select(vec3f(1.0) - fraction, fraction, offset == vec3u(1u));
        let fixedWeight = u32(max(0.0, weight3.x * weight3.y * weight3.z * 256.0));
        atomicAdd(&mass[index3(cell)], fixedWeight);
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
  focusId: u32, _pad1: u32, _pad2: vec2f,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> mass: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> inputPotential: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputPotential: array<f32>;

fn index3(p: vec3u) -> u32 { return p.x + p.y * params.gridSide + p.z * params.gridSide * params.gridSide; }
fn wrap(value: i32) -> u32 {
  let side = i32(params.gridSide);
  return u32((value % side + side) % side);
}

@compute @workgroup_size(256)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let cells = params.gridSide * params.gridSide * params.gridSide;
  let i = gid.x;
  if (i >= cells) { return; }
  let side = params.gridSide;
  let x = i % side;
  let y = (i / side) % side;
  let z = i / (side * side);
  let left = index3(vec3u(wrap(i32(x) - 1), y, z));
  let right = index3(vec3u(wrap(i32(x) + 1), y, z));
  let down = index3(vec3u(x, wrap(i32(y) - 1), z));
  let up = index3(vec3u(x, wrap(i32(y) + 1), z));
  let back = index3(vec3u(x, y, wrap(i32(z) - 1)));
  let front = index3(vec3u(x, y, wrap(i32(z) + 1)));
  let spacing = params.domain / f32(side - 1u);
  let depositedMass = f32(atomicLoad(&mass[i])) / 256.0;
  let meanMass = f32(params.count) / f32(cells);
  let density = (depositedMass - meanMass) / (spacing * spacing * spacing);
  outputPotential[i] = (inputPotential[left] + inputPotential[right] + inputPotential[down] + inputPotential[up] + inputPotential[back] + inputPotential[front] - density * spacing * spacing) / 6.0;
}
`;

export const POST_SHADER = /* wgsl */ `
struct Params {
  count: u32, phase: u32, gridSide: u32, _pad0: u32,
  dt: f32, age: f32, gravity: f32, explosion: f32,
  entropy: f32, lifetime: f32, spectrum: f32, domain: f32,
  viewport: vec2f, zoom: f32, yaw: f32,
  pitch: f32, approach: f32, pan: vec2f,
  focusId: u32, _pad1: u32, _pad2: vec2f,
}
struct Particle { position: vec4f, velocity: vec4f }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> accelerations: array<vec4f>;
@group(0) @binding(3) var<storage, read> potential: array<f32>;

fn index3(p: vec3u) -> u32 { return p.x + p.y * params.gridSide + p.z * params.gridSide * params.gridSide; }
fn wrap(value: i32) -> u32 {
  let side = i32(params.gridSide);
  return u32((value % side + side) % side);
}
fn gradient(cell: vec3u) -> vec3f {
  let spacing = params.domain / f32(params.gridSide - 1u);
  let x0 = index3(vec3u(wrap(i32(cell.x) - 1), cell.y, cell.z));
  let x1 = index3(vec3u(wrap(i32(cell.x) + 1), cell.y, cell.z));
  let y0 = index3(vec3u(cell.x, wrap(i32(cell.y) - 1), cell.z));
  let y1 = index3(vec3u(cell.x, wrap(i32(cell.y) + 1), cell.z));
  let z0 = index3(vec3u(cell.x, cell.y, wrap(i32(cell.z) - 1)));
  let z1 = index3(vec3u(cell.x, cell.y, wrap(i32(cell.z) + 1)));
  return -vec3f(potential[x1] - potential[x0], potential[y1] - potential[y0], potential[z1] - potential[z0]) / (2.0 * spacing);
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
        acceleration += gradient(cell) * weight3.x * weight3.y * weight3.z;
      }
    }
  }
  acceleration *= params.gravity;
  var velocity = particles[i].velocity.xyz + acceleration * params.dt * 0.5;
  if (accelerations[i].w < 0.0) {
    let a2 = dot(acceleration, acceleration);
    if (a2 > 0.0001) {
      velocity -= acceleration * (dot(velocity, acceleration) / a2) * (1.0 - exp(-0.42 * params.dt));
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
  focusId: u32, _pad1: u32, _pad2: vec2f,
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
  @location(3) focused: f32,
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
  var focus = vec3f(0.0);
  if (params.focusId < params.count) { focus = particles[params.focusId].position.xyz; }
  let view = rotate(particle.position.xyz - focus);
  let focal = max(params.viewport.x, params.viewport.y) * 1.15;
  let perspective = focal / max(20.0, focal + view.z * params.zoom) * params.zoom;
  let center = params.viewport * 0.5 + params.pan + view.xy * perspective;
  let aura = select(0.0, 1.0, particle.position.w > 3.0);
  let isFocused = params.focusId < params.count && instanceId == params.focusId;
  let baseRadius = max(0.65, particle.position.w * perspective) * select(1.0, 3.5, aura > 0.5);
  let pixelRadius = select(baseRadius, max(baseRadius, 20.0), isFocused);
  let screen = center + corners[vertexId] * pixelRadius;
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
  output.color = vec4f(rgb, 0.68);

  output.aura = aura;
  output.focused = select(0.0, 1.0, isFocused);
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
  var finalColor = select(input.color.rgb, auraColor, input.aura > .5);
  var finalAlpha = alpha;
  if (input.focused > 0.5) {
    let ring = smoothstep(0.1, 0.0, abs(radius - 0.78));
    finalColor = mix(finalColor, vec3f(1.0, 1.0, 1.0), ring * 0.9);
    finalAlpha = max(finalAlpha, ring * 0.75);
  }
  return vec4f(finalColor, finalAlpha);
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

export const PICK_SHADER = /* wgsl */ `
struct Params {
  count: u32, phase: u32, gridSide: u32, _pad0: u32,
  dt: f32, age: f32, gravity: f32, explosion: f32,
  entropy: f32, lifetime: f32, spectrum: f32, domain: f32,
  viewport: vec2f, zoom: f32, yaw: f32,
  pitch: f32, approach: f32, pan: vec2f,
  focusId: u32, _pad1: u32, _pad2: vec2f,
}
struct Particle { position: vec4f, velocity: vec4f }
struct Pick { point: vec2f, radius: f32, _pad: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> accelerations: array<vec4f>;
@group(0) @binding(3) var<uniform> pick: Pick;
@group(0) @binding(4) var<storage, read_write> result: atomic<u32>;

fn rotate(point: vec3f) -> vec3f {
  let cy = cos(params.yaw); let sy = sin(params.yaw);
  let cp = cos(params.pitch); let sp = sin(params.pitch);
  let x = cy * point.x - sy * point.z;
  let z = sy * point.x + cy * point.z;
  return vec3f(x, cp * point.y - sp * z, sp * point.y + cp * z);
}

@compute @workgroup_size(256)
fn pickNearest(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.count || abs(accelerations[i].w) <= 0.0) { return; }
  var focus = vec3f(0.0);
  if (params.focusId < params.count) { focus = particles[params.focusId].position.xyz; }
  let view = rotate(particles[i].position.xyz - focus);
  let focal = max(params.viewport.x, params.viewport.y) * 1.15;
  let perspective = focal / max(20.0, focal + view.z * params.zoom) * params.zoom;
  let screen = params.viewport * .5 + params.pan + view.xy * perspective;
  let distance = length(screen - pick.point);
  if (distance <= pick.radius) {
    let quantized = min(2047u, u32(distance * 16.0));
    atomicMin(&result, (quantized << 21u) | i);
  }
}
`;
