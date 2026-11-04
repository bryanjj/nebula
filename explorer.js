const canvas = document.getElementById('gfx');
const slider = document.getElementById('slider');
const sliderValue = document.getElementById('sliderValue');
const clipSlider = document.getElementById('clipSlider');
const clipValue = document.getElementById('clipValue');
const modeSelect = document.getElementById('modeSelect');
const statusEl = document.getElementById('status');
const qualitySelect = document.getElementById('qualitySelect');
const retryButton = document.getElementById('retryButton');
const sliderDimNames = ['Im(c)', 'Re(z₀)', 'Im(z₀)', 'Re(c)'];
// X/Y/Z order matches map_to_fractal in the shader. Labels follow the selected slice; orientation follows the camera.
const visibleAxes = [
  ['Re(z₀)', 'Im(z₀)', 'Re(c)'],
  ['Re(c)', 'Im(c)', 'Im(z₀)'],
  ['Re(z₀)', 'Re(c)', 'Im(c)'],
  ['Re(z₀)', 'Im(z₀)', 'Im(c)'],
];
function updateAxisGuide() {
  const names = visibleAxes[mode];
  for (const [i, axis] of ['X', 'Y', 'Z'].entries()) {
    document.getElementById(`axis${axis}Label`).textContent = names[i];
  }
  document.getElementById('axisGuide').ariaLabel = `Visible axes: X ${names[0]}, Y ${names[1]}, Z ${names[2]}`;
}

let guidePitch;
let guideYaw;
function updateAxisOrientation() {
  if (guidePitch === rotationX && guideYaw === rotationY) return;
  guidePitch = rotationX;
  guideYaw = rotationY;
  // Same right/up basis as the shader; SVG Y points down on screen.
  const sinP = Math.sin(rotationX), cosP = Math.cos(rotationX);
  const sinY = Math.sin(rotationY), cosY = Math.cos(rotationY);
  const right = [cosY, 0, -sinY];
  const up = [-sinP * sinY, cosP, -sinP * cosY];
  const towardCamera = [cosP * sinY, sinP, cosP * cosY];
  for (const [i, axis] of ['X', 'Y', 'Z'].entries()) {
    const dx = 36 * right[i], dy = -36 * up[i];
    const length = Math.hypot(dx, dy);
    const ux = length > 0.001 ? dx / length : 0;
    const uy = length > 0.001 ? dy / length : 1;
    const x = 80 + dx, y = 70 + dy;
    const path = document.getElementById(`axis${axis}Arrow`);
    let d = `M80 70 L${x} ${y}`;
    if (length > 6) {
      d += ` M${x - 5 * ux + 3 * uy} ${y - 5 * uy - 3 * ux} L${x} ${y} L${x - 5 * ux - 3 * uy} ${y - 5 * uy + 3 * ux}`;
    }
    path.setAttribute('d', d);
    const label = document.getElementById(`axis${axis}Label`);
    label.setAttribute('x', x + 22 * ux);
    label.setAttribute('y', y + 22 * uy + 4);
    const opacity = towardCamera[i] < 0 ? '0.55' : '1';
    path.setAttribute('opacity', opacity);
    label.setAttribute('opacity', opacity);
  }
}

let sliderVal = 0;
let clipVal = -3;
let mode = 0;
let rotationX = 0.3;
let rotationY = 0;
let clipRotationX = rotationX;
let clipRotationY = rotationY;
let clipAnimation = null;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let zoom = 1;
let panX = 0;
let panY = 0;
let drag = null;
let renderFrame = null;
let pendingFrame = 0;
let stopped = false;
let resizeCanvas = () => {};

// Render while aligning the plane, then return to rendering only on interaction.
function animateClip(now) {
  if (!clipAnimation) return;
  const a = clipAnimation;
  const progress = reducedMotion.matches ? 1 : Math.min(1, (now - a.start) / 450);
  const eased = progress * progress * (3 - 2 * progress);
  clipRotationX = a.fromX + (a.toX - a.fromX) * eased;
  clipRotationY = a.fromY + a.deltaY * eased;
  if (progress === 1) clipAnimation = null;
}

function invalidate() {
  updateAxisOrientation();
  if (!renderFrame || pendingFrame || stopped || document.hidden) return;
  pendingFrame = requestAnimationFrame((now) => {
    pendingFrame = 0;
    if (stopped || document.hidden) return;
    try {
      animateClip(now);
      renderFrame();
      if (clipAnimation) invalidate();
    } catch (error) { fail(error); }
  });
}

function fail(error) {
  stopped = true;
  clipAnimation = null;
  cancelAnimationFrame(pendingFrame);
  pendingFrame = 0;
  statusEl.dataset.error = 'true';
  statusEl.textContent = error.message || String(error);
  retryButton.hidden = false;
  console.error(error);
}
retryButton.addEventListener('click', () => location.reload());

slider.addEventListener('input', () => {
  sliderVal = Number(slider.value);
  sliderValue.textContent = sliderVal.toFixed(3);
  invalidate();
});
clipSlider.addEventListener('input', () => {
  clipVal = Number(clipSlider.value);
  clipValue.textContent = clipVal.toFixed(2);
  invalidate();
});
modeSelect.addEventListener('change', () => {
  mode = Number(modeSelect.value);
  slider.ariaLabel = sliderDimNames[mode];
  updateAxisGuide();
  invalidate();
});
document.getElementById('alignClipButton').addEventListener('click', () => {
  // Use the shorter yaw arc, including after orbiting across ±π or many turns.
  const delta = rotationY - clipRotationY;
  clipAnimation = {
    start: performance.now(), fromX: clipRotationX, fromY: clipRotationY,
    toX: rotationX, deltaY: Math.atan2(Math.sin(delta), Math.cos(delta)),
  };
  invalidate();
});
qualitySelect.addEventListener('change', () => { resizeCanvas(); invalidate(); });

function reset() {
  clipAnimation = null;
  rotationX = 0.3;
  rotationY = 0;
  clipRotationX = rotationX;
  clipRotationY = rotationY;
  zoom = 1;
  panX = panY = 0;
  mode = 0;
  sliderVal = 0;
  clipVal = -3;
  modeSelect.value = '0';
  slider.value = '0';
  clipSlider.value = '-3';
  sliderValue.textContent = '0.000';
  clipValue.textContent = '-3.00';
  slider.ariaLabel = sliderDimNames[0];
  updateAxisGuide();
  invalidate();
}
document.getElementById('resetButton').addEventListener('click', reset);

canvas.addEventListener('pointerdown', (e) => {
  if (drag || ![0, 2].includes(e.button)) return;
  canvas.focus({ preventScroll: true });
  canvas.setPointerCapture(e.pointerId);
  drag = { id: e.pointerId, x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
  canvas.style.cursor = drag.pan ? 'move' : 'grabbing';
});
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointermove', (e) => {
  if (!drag || drag.id !== e.pointerId) return;
  const dx = e.clientX - drag.x;
  const dy = e.clientY - drag.y;
  drag.x = e.clientX;
  drag.y = e.clientY;
  if (drag.pan) {
    panX -= dx * 0.003 / zoom;
    panY += dy * 0.003 / zoom;
  } else {
    rotationY -= dx * 0.005;
    rotationX = Math.max(-1.55, Math.min(1.55, rotationX + dy * 0.005));
  }
  invalidate();
});
function endDrag(e) {
  if (!drag || drag.id !== e.pointerId) return;
  drag = null;
  canvas.style.cursor = 'grab';
}
for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) {
  canvas.addEventListener(event, endDrag);
}
function changeZoom(factor) { zoom = Math.max(0.1, Math.min(10, zoom * factor)); }
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const pixels = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? canvas.clientHeight : 1);
  changeZoom(Math.exp(-pixels * 0.001));
  invalidate();
}, { passive: false });
canvas.addEventListener('keydown', (e) => {
  switch (e.key) {
    case 'ArrowLeft': rotationY -= 0.1; break;
    case 'ArrowRight': rotationY += 0.1; break;
    case 'ArrowUp': rotationX = Math.max(-1.55, rotationX - 0.1); break;
    case 'ArrowDown': rotationX = Math.min(1.55, rotationX + 0.1); break;
    case '+': case '=': changeZoom(1.1); break;
    case '-': case '_': changeZoom(1 / 1.1); break;
    case 'r': case 'R': reset(); break;
    default: return;
  }
  e.preventDefault();
  invalidate();
});
document.addEventListener('visibilitychange', invalidate);

async function init() {
  if (!window.isSecureContext) throw new Error('WebGPU needs a secure context. Open this demo on localhost or HTTPS.');
  if (!navigator.gpu) throw new Error('WebGPU is unavailable. Try a WebGPU-enabled browser with hardware acceleration.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter found. Check hardware acceleration and GPU drivers.');
  const device = await adapter.requestDevice();
  device.lost.then((info) => fail(new Error('GPU connection lost: ' + info.message + ' Reload to retry.')));
  device.addEventListener('uncapturederror', (event) => fail(event.error));
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('Unable to create a WebGPU canvas. Try another browser.');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  resizeCanvas = () => {
    const rect = canvas.getBoundingClientRect();
    // Cap both device pixels and total work: raymarching is expensive on Retina displays.
    let scale = Math.min(window.devicePixelRatio || 1, 2) * Number(qualitySelect.value);
    scale = Math.min(scale, Math.sqrt(2_000_000 / Math.max(1, rect.width * rect.height)));
    const limit = device.limits.maxTextureDimension2D;
    scale = Math.min(scale, limit / Math.max(1, rect.width), limit / Math.max(1, rect.height));
    canvas.width = Math.max(1, Math.floor(rect.width * scale));
    canvas.height = Math.max(1, Math.floor(rect.height * scale));
    invalidate();
  };
  const observer = new ResizeObserver(resizeCanvas);
  observer.observe(canvas);
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // Uniforms: 3 vec4<f32> = 12 floats
  // data1: resolution.x, resolution.y, sliderVal, zoom
  // data2: mode, rotationX, rotationY, clipDist
  // data3: panX, panY, clipRotationX, clipRotationY
  const uniformBufferSize = 12 * 4;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const shaderCode = /* wgsl */`
struct Uniforms {
  data1 : vec4<f32>, // res.x, res.y, slider, zoom
  data2 : vec4<f32>, // mode, rotationX, rotationY, clipDist
  data3 : vec4<f32>, // panX, panY, clipRotationX, clipRotationY
};
@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv      : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> VSOut {
  // Fullscreen triangle
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>( 3.0,  1.0),
    vec2<f32>(-1.0,  1.0)
  );
  var out : VSOut;
  let p = positions[vid];
  out.pos = vec4<f32>(p, 0.0, 1.0);
  out.uv = 0.5 * (p + vec2<f32>(1.0, 1.0)); // [0,1]
  return out;
}

fn getResolution() -> vec2<f32> {
  return uniforms.data1.xy;
}

fn getSlider() -> f32 {
  return uniforms.data1.z;
}

fn getZoom() -> f32 {
  return uniforms.data1.w;
}

fn getRotationX() -> f32 {
  return uniforms.data2.y;
}

fn getRotationY() -> f32 {
  return uniforms.data2.z;
}

fn getClipDist() -> f32 {
  return uniforms.data2.w;
}

fn getMode() -> i32 {
  return i32(uniforms.data2.x);
}

fn getPan() -> vec2<f32> {
  return uniforms.data3.xy;
}

// Complex helpers
fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

fn c_len2(z: vec2<f32>) -> f32 {
  return dot(z,z);
}

// Define the return type as a named struct
struct FractalMap {
    z0 : vec2<f32>,
    c  : vec2<f32>,
};

// Map 3D world point p → (z0, c) in C^2
fn map_to_fractal(p: vec3<f32>, mode: i32, slider: f32) -> FractalMap {
  var result: FractalMap;

  if (mode == 0) {
    // Mode 0:
    //   z0 = x + i·y
    //   c  = z + i·slider
    result.z0 = vec2<f32>(p.x, p.y);
    result.c  = vec2<f32>(p.z, slider);
  } else if (mode == 1) {
    // Mode 1:
    //   z0 = slider + i·z
    //   c  = x + i·y
    result.z0 = vec2<f32>(slider, p.z);
    result.c  = vec2<f32>(p.x, p.y);
  } else if (mode == 2) {
    // Mode 2:
    //   z0 = x + i·slider
    //   c  = y + i·z
    result.z0 = vec2<f32>(p.x, slider);
    result.c  = vec2<f32>(p.y, p.z);
  } else {
    // Mode 3 (fallback/default):
    //   z0 = x + i·y
    //   c  = slider + i·z
    result.z0 = vec2<f32>(p.x, p.y);
    result.c  = vec2<f32>(slider, p.z);
  }

  return result;
}

// Distance-estimation-style function for z_{n+1} = z_n^2 + c
// This is adapted from Mandelbrot DE (w.r.t c), used as a pseudo-distance
// in the chosen 3D slice. It's not mathematically perfect, but works well
// visually for raymarching.
fn fractal_distance(p: vec3<f32>, mode: i32, slider: f32) -> f32 {
  let mp = map_to_fractal(p, mode, slider);
  var z  = mp.z0;
  let c  = mp.c;

  let maxIter : i32 = 80;
  let bailout : f32 = 8.0;
  let bailout2 = bailout * bailout;

  // derivative dz/dc
  var dz = vec2<f32>(0.0, 0.0);
  var escaped = false;
  var iEsc : i32 = maxIter;

  for (var i: i32 = 0; i < maxIter; i = i + 1) {
    let oldz = z;
    let olddz = dz;

    // z = z^2 + c
    let zx2 = oldz.x*oldz.x - oldz.y*oldz.y + c.x;
    let zy2 = 2.0*oldz.x*oldz.y + c.y;
    z = vec2<f32>(zx2, zy2);

    // dz = 2*z_old*dz_old + 1
    let dx = 2.0*(oldz.x*olddz.x - oldz.y*olddz.y) + 1.0;
    let dy = 2.0*(oldz.x*olddz.y + oldz.y*olddz.x);
    dz = vec2<f32>(dx, dy);

    if (c_len2(z) > bailout2) {
      escaped = true;
      iEsc = i;
      break;
    }
  }

  if (!escaped) {
    // Deep interior -> treat as "solid" but return a small value
    return 0.0;
  }

  let r = length(z);
  let dr = max(length(dz), 1e-6);
  let de = 0.5 * log(r) * r / dr;

  // Clamp for stability
  return clamp(de, 0.0, 1.0);
}

// Get escape time for coloring (returns 0-1 normalized value)
fn get_escape_time(p: vec3<f32>, mode: i32, slider: f32) -> f32 {
  let mp = map_to_fractal(p, mode, slider);
  var z = mp.z0;
  let c = mp.c;

  let maxIter: i32 = 80;
  let bailout: f32 = 8.0;
  let bailout2 = bailout * bailout;

  for (var i: i32 = 0; i < maxIter; i = i + 1) {
    let zx2 = z.x * z.x - z.y * z.y + c.x;
    let zy2 = 2.0 * z.x * z.y + c.y;
    z = vec2<f32>(zx2, zy2);

    let r2 = c_len2(z);
    if (r2 > bailout2) {
      // Smooth escape time using fractional iteration count
      let log_zn = log(r2) * 0.5;
      let nu = log(log_zn / log(bailout)) / log(2.0);
      let smooth_i = f32(i) + 1.0 - nu;
      return smooth_i / f32(maxIter);
    }
  }

  // Interior point - return 1.0 (will be colored differently)
  return 1.0;
}

// Estimate normal from distance field via central differences
fn estimate_normal(p: vec3<f32>, mode: i32, slider: f32, zoom: f32) -> vec3<f32> {
  let eps = 0.003 / zoom;
  let dx = fractal_distance(p + vec3<f32>(eps, 0.0, 0.0), mode, slider)
         - fractal_distance(p - vec3<f32>(eps, 0.0, 0.0), mode, slider);
  let dy = fractal_distance(p + vec3<f32>(0.0, eps, 0.0), mode, slider)
         - fractal_distance(p - vec3<f32>(0.0, eps, 0.0), mode, slider);
  let dz = fractal_distance(p + vec3<f32>(0.0, 0.0, eps), mode, slider)
         - fractal_distance(p - vec3<f32>(0.0, 0.0, eps), mode, slider);
  let gradient = vec3<f32>(dx, dy, dz);
  return gradient / max(length(gradient), 1e-8);
}

fn getClipNormal() -> vec3<f32> {
  let pitch = uniforms.data3.z;
  let yaw = uniforms.data3.w;
  return -vec3<f32>(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw));
}

// Removed material must not cast shadows or occlude the exposed cut.
fn clipped_distance(p: vec3<f32>, mode: i32, slider: f32) -> f32 {
  let d = fractal_distance(p, mode, slider);
  if (getClipDist() <= -2.999) { return d; }
  return max(d, getClipDist() - dot(p, getClipNormal()));
}

// Soft shadow via secondary raymarch
fn soft_shadow(origin: vec3<f32>, lightDir: vec3<f32>, mode: i32, slider: f32) -> f32 {
  var t = 0.02;
  let maxT: f32 = 4.0;
  var res: f32 = 1.0;
  let k: f32 = 16.0;

  for (var i: i32 = 0; i < 40; i = i + 1) {
    let p = origin + lightDir * t;
    let d = clipped_distance(p, mode, slider);
    if (d < 0.001) {
      return 0.0;
    }
    res = min(res, k * d / t);
    t = t + clamp(d, 0.01, 0.2);
    if (t > maxT) {
      break;
    }
  }
  return clamp(res, 0.0, 1.0);
}

// Very simple AO-like factor sampling along normal
fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>, mode: i32, slider: f32) -> f32 {
  var occ: f32 = 0.0;
  let steps: i32 = 4;
  let stepSize: f32 = 0.1;
  for (var i: i32 = 1; i <= steps; i = i + 1) {
    let t = f32(i) * stepSize;
    let sampleP = p + n * t;
    let d = clipped_distance(sampleP, mode, slider);
    occ = occ + (t - d);
  }
  occ = occ / f32(steps);
  return clamp(1.0 - 1.5 * occ, 0.0, 1.0);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let res = getResolution();
  let slider = getSlider();
  let zoom = getZoom();
  let rotX = getRotationX();
  let rotY = getRotationY();
  let clipDist = getClipDist();
  let mode = getMode();
  let pan = getPan();

  // UV -> [-1,1]
  let uv = (in.uv * 2.0 - vec2<f32>(1.0, 1.0));
  let aspect = res.x / res.y;

  // Camera position from spherical coordinates (mouse controlled)
  // rotX = pitch (up/down), rotY = yaw (left/right)
  // Camera stays at fixed distance; zoom scales the fractal space
  let camRadius: f32 = 5.0;

  // Build camera basis vectors directly from rotation angles
  // This avoids the singularity when forward is parallel to worldUp
  let sinP = sin(rotX);
  let cosP = cos(rotX);
  let sinY = sin(rotY);
  let cosY = cos(rotY);

  // Camera position on sphere
  let camPos = vec3<f32>(
    camRadius * cosP * sinY,
    camRadius * sinP,
    camRadius * cosP * cosY
  );

  // Forward vector (pointing toward origin)
  let forward = normalize(-camPos);

  // Right vector (always horizontal, perpendicular to Y axis rotation)
  let right = vec3<f32>(cosY, 0.0, -sinY);

  // Up vector (perpendicular to both forward and right)
  let up = cross(right, forward);

  // Perspective
  let fov = 45.0 * 3.14159 / 180.0;
  let tanFov = tan(0.5 * fov);

  var rd = normalize(
    forward +
    uv.x * aspect * tanFov * right +
    uv.y * tanFov * up
  );

  // Raymarch - scale precision with zoom level
  let maxDist: f32 = 120.0;
  let maxSteps: i32 = i32(200.0 + 100.0 * min(zoom, 5.0));  // More steps when zoomed in
  let hitEps: f32 = 0.001 / zoom;  // Finer hit detection when zoomed
  let minStep: f32 = 0.001 / zoom;  // Finer minimum step when zoomed

  // Transform the fractal bounds along with zoom and pan.
  let boxSize: f32 = 3.0;

  // Ray-box intersection to find entry point
  let panOffset = right * pan.x + up * pan.y;
  let boxMin = (vec3<f32>(-boxSize) - panOffset) * zoom;
  let boxMax = (vec3<f32>(boxSize) - panOffset) * zoom;
  let invRd = 1.0 / rd;
  let t1 = (boxMin - camPos) * invRd;
  let t2 = (boxMax - camPos) * invRd;
  let tMin = min(t1, t2);
  let tMax = max(t1, t2);
  let tEnter = max(max(tMin.x, tMin.y), tMin.z);
  let tExit = min(min(tMax.x, tMax.y), tMax.z);

  // Start ray at box entry (or 0 if already inside)
  var t: f32 = max(0.0, tEnter - 0.01);

  var hit = false;
  var hitPos = vec3<f32>(0.0);

  // Check if ray misses box entirely
  if (tEnter > tExit || tExit < 0.0) {
    let bg = 0.1 + 0.3 * uv.y;
    return vec4<f32>(vec3<f32>(bg), 1.0);
  }

  // The plane lives in fractal coordinates, so orbit, zoom and pan preserve
  // the cut. Only Align to view changes its orientation.
  let clipNormal = getClipNormal();
  var rayEnd = min(tExit, maxDist);
  if (clipDist > -2.999) {
    let originDistance = dot(camPos / zoom + panOffset, clipNormal) - clipDist;
    let direction = dot(rd / zoom, clipNormal);
    if (abs(direction) < 0.000001) {
      // Parallel rays either stay inside the retained half-space or miss it.
      if (originDistance < 0.0) { rayEnd = -1.0; }
    } else {
      let clipT = -originDistance / direction;
      if (direction > 0.0) {
        t = max(t, clipT);
      } else {
        // From the back, rays leave the retained half-space at the plane.
        rayEnd = min(rayEnd, clipT);
      }
    }
  }

  for (var i: i32 = 0; i < maxSteps; i = i + 1) {
    if (t > rayEnd) { break; }
    let p = camPos + rd * t;

    // Scale point by zoom and apply pan offset in camera-relative space
    let pScaled = p / zoom + right * pan.x + up * pan.y;
    let de = fractal_distance(pScaled, mode, slider) * zoom;  // Scale distance back
    if (de < hitEps) {
      hit = true;
      hitPos = p;
      break;
    }

    let stepSize = max(de * 0.7, minStep);
    t = t + stepSize;

    if (t > rayEnd) {
      break;
    }
  }

  if (!hit) {
    // Simple background gradient
    let bg = 0.1 + 0.3 * uv.y;
    return vec4<f32>(vec3<f32>(bg), 1.0);
  }

  // Shading - use scaled coordinates with pan offset for fractal lookups
  let hitPosScaled = hitPos / zoom + right * pan.x + up * pan.y;
  var n = estimate_normal(hitPosScaled, mode, slider, zoom);
  if (clipDist > -2.999 && abs(dot(hitPosScaled, clipNormal) - clipDist) < 0.00001) {
    n = -clipNormal;
  }
  if (dot(n, n) < 0.01) { n = -forward; }

  // Light - positioned behind and above camera
  let viewDir = normalize(camPos - hitPos);
  let lightDir = normalize(viewDir + up * 0.3);  // Light from camera direction, slightly above
  let diff = max(dot(n, lightDir), 0.0);
  let halfDir = normalize(lightDir + viewDir);
  let spec = pow(max(dot(n, halfDir), 0.0), 40.0);

  // Shadows & AO (use scaled coordinates with pan)
  let sh = soft_shadow(hitPosScaled + n * 0.01 / zoom, lightDir, mode, slider);
  let ao = ambient_occlusion(hitPosScaled, n, mode, slider);

  // Get escape time for coloring
  let escapeTime = get_escape_time(hitPosScaled, mode, slider);

  // Colormap based on escape time (single cycle across surface)
  let t_color = escapeTime;  // Single color cycle
  let baseColor = vec3<f32>(
    0.5 + 0.5 * cos(6.28318 * (t_color + 0.0)),
    0.5 + 0.5 * cos(6.28318 * (t_color + 0.33)),
    0.5 + 0.5 * cos(6.28318 * (t_color + 0.67))
  );

  let ambient = 0.35 * ao;
  let diffuse = 1.0 * diff * sh;
  let specular = 0.6 * spec * sh;

  var color = baseColor * (ambient + diffuse) + vec3<f32>(1.0, 0.95, 0.9) * specular;
  color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
  // Subtle gamma
  color = pow(color, vec3<f32>(0.9));

  return vec4<f32>(color, 1.0);
}
  `;

  const module = device.createShaderModule({ code: shaderCode });

  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length) throw new Error('Shader error: ' + errors.map((e) => e.message).join('; '));

  const pipeline = await device.createRenderPipelineAsync({
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vs_main',
    },
    fragment: {
      module,
      entryPoint: 'fs_main',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{
      binding: 0,
      resource: { buffer: uniformBuffer },
    }],
  });

  const data = new Float32Array(12);
  renderFrame = () => {
    data[0] = canvas.width;
    data[1] = canvas.height;
    data[2] = sliderVal;
    data[3] = zoom;
    data[4] = mode;
    data[5] = rotationX;
    data[6] = rotationY;
    data[7] = clipVal;
    data[8] = panX;
    data[9] = panY;
    data[10] = clipRotationX;
    data[11] = clipRotationY;
    device.queue.writeBuffer(uniformBuffer, 0, data.buffer, data.byteOffset, data.byteLength);

    const encoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3, 1, 0, 0);
    pass.end();

    device.queue.submit([encoder.finish()]);
  };

  if (stopped) return;
  statusEl.textContent = 'WebGPU ready · renders on interaction';
  invalidate();
}

updateAxisOrientation();
init().catch(fail);
