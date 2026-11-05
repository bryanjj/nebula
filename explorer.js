const canvas = document.getElementById('gfx');
const slider = document.getElementById('slider');
const sliderValue = document.getElementById('sliderValue');
const clipSlider = document.getElementById('clipSlider');
const clipValue = document.getElementById('clipValue');
const modeSelect = document.getElementById('modeSelect');
const statusEl = document.getElementById('status');
const retryButton = document.getElementById('retryButton');
const orbitButton = document.getElementById('orbitButton');
const flyButton = document.getElementById('flyButton');
const cutawayButton = document.getElementById('cutawayButton');
const bubbleSlider = document.getElementById('bubbleSlider');
const bubbleValue = document.getElementById('bubbleValue');
const help = document.getElementById('help');
const orbitHelp = help.textContent;
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

// Open on the z₀ = 0 cut: Re(c) right, Im(c) up, Im(z₀) toward the camera.
let sliderVal = 0;
let clipVal = 0;
let mode = 1;
let rotationX = 0;
let rotationY = 0;
let clipRotationX = rotationX;
let clipRotationY = rotationY;
let clipAnimation = null;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let zoom = 1;
let panX = 0;
let panY = 0;
let flying = false;
let cutaway = false;
let bubbleRadius = 0.001;
let flyPosition = [0, 0, 5];
let orbitView = null;
let drag = null;
let renderFrame = null;
let pendingFrame = 0;
let stopped = false;
let gpuBusy = false;
let needsRender = false;
let refineRequested = false;
let refineTimer = 0;
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

// Never queue camera poses behind unfinished GPU work. While a frame is in
// flight, inputs update the state and the next frame uses only the latest pose.
function scheduleFrame() {
  if (!renderFrame || pendingFrame || gpuBusy || !needsRender || stopped || document.hidden) return;
  pendingFrame = requestAnimationFrame((now) => {
    pendingFrame = 0;
    if (stopped || document.hidden) return;
    needsRender = false;
    gpuBusy = true;
    try {
      animateClip(now);
      renderFrame(flying && !refineRequested).then(() => {
        gpuBusy = false;
        if (stopped) return;
        if (clipAnimation) invalidate();
        else scheduleFrame();
      }, fail);
    } catch (error) { fail(error); }
  });
}

function invalidate() {
  updateAxisOrientation();
  needsRender = true;
  refineRequested = false;
  clearTimeout(refineTimer);
  // Refine only after input has settled. New input cancels pending refinement.
  if (flying && !stopped && !document.hidden) {
    refineTimer = setTimeout(() => {
      refineTimer = 0;
      refineRequested = true;
      needsRender = true;
      scheduleFrame();
    }, 240);
  }
  scheduleFrame();
}

function fail(error) {
  stopped = true;
  clearTimeout(refineTimer);
  clipAnimation = null;
  cancelAnimationFrame(pendingFrame);
  pendingFrame = 0;
  statusEl.hidden = false;
  statusEl.dataset.error = 'true';
  statusEl.textContent = error.message || String(error);
  retryButton.hidden = false;
  console.error(error);
}
retryButton.addEventListener('click', () => location.reload());

// Snap pointer drags within 2% of the range to zero. Keyboard input stays
// precise so a single arrow press can always move away from zero.
let activeSlider = null;
for (const control of [slider, clipSlider]) {
  control.addEventListener('pointerdown', () => { activeSlider = control; });
  for (const event of ['keydown', 'blur', 'lostpointercapture']) {
    control.addEventListener(event, () => {
      if (activeSlider === control) activeSlider = null;
    });
  }
}
for (const event of ['pointerup', 'pointercancel', 'blur']) {
  window.addEventListener(event, () => { activeSlider = null; });
}
function readSliderValue(control) {
  let value = Number(control.value);
  const snapDistance = (Number(control.max) - Number(control.min)) * 0.02;
  if (activeSlider === control && Math.abs(value) <= snapDistance) {
    value = 0;
    control.value = '0';
  }
  return value;
}

slider.addEventListener('input', () => {
  sliderVal = readSliderValue(slider);
  sliderValue.textContent = sliderVal.toFixed(3);
  invalidate();
});
clipSlider.addEventListener('input', () => {
  clipVal = readSliderValue(clipSlider);
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

function cameraBasis(pitch = rotationX, yaw = rotationY) {
  const sp = Math.sin(pitch), cp = Math.cos(pitch), sy = Math.sin(yaw), cy = Math.cos(yaw);
  return { forward: [-cp * sy, -sp, -cp * cy], right: [cy, 0, -sy], up: [-sp * sy, cp, -sp * cy] };
}

// CPU counterpart of the shader's pseudo-distance, used only for navigation.
// Keep the same mapping, derivative weights, and clipping. Navigation uses
// the conservative 80-iteration field; visible Fly hits can refine further.
function navigationDistance(p) {
  const [x, y, z] = p;
  let [a, b, c, d] = [
    [x, y, z, sliderVal], [sliderVal, z, x, y],
    [x, sliderVal, y, z], [x, y, sliderVal, z],
  ][mode];
  let za = 1, zb = 0, ca = 0, cb = 0, distance = 0;
  for (let i = 0; i < 80; i++) {
    [za, zb] = [2 * (a * za - b * zb), 2 * (a * zb + b * za)];
    [ca, cb] = [2 * (a * ca - b * cb) + 1, 2 * (a * cb + b * ca)];
    [a, b] = [a * a - b * b + c, 2 * a * b + d];
    const r = Math.hypot(a, b);
    if (r > 8) {
      const zWeight = mode === 0 || mode === 3 ? 2 : 1;
      const derivative = Math.max(Math.sqrt(zWeight * (za * za + zb * zb) + (3 - zWeight) * (ca * ca + cb * cb)), 1e-6);
      distance = Math.min(1, Math.max(0, 0.5 * Math.log(r) * r / derivative));
      break;
    }
  }
  if (clipVal > -2.999) {
    const normal = cameraBasis(clipRotationX, clipRotationY).forward;
    distance = Math.max(distance, clipVal - p.reduce((dot, v, i) => dot + v * normal[i], 0));
  }
  return distance;
}

function activeBubbleRadius() { return flying && cutaway ? bubbleRadius : 0; }

function updateNavigationUI() {
  orbitButton.setAttribute('aria-pressed', String(!flying));
  flyButton.setAttribute('aria-pressed', String(flying));
  document.getElementById('flyReticle').hidden = !flying;
  document.getElementById('flyControls').hidden = !flying;
  cutawayButton.setAttribute('aria-pressed', String(cutaway));
  cutawayButton.title = cutaway
    ? 'Turn off the bubble; return outside if the camera is inside solid material'
    : 'Carve a bubble around the camera and fly through solid material';
  bubbleSlider.disabled = !cutaway;
  document.getElementById('resetButton').title = flying
    ? 'Return to the starting position outside the fractal'
    : 'Restore the starting camera orientation and zoom';
  help.textContent = flying
    ? 'Drag to look · Scroll or W/S to fly forward/back · A/D or Shift-drag to move sideways · Q/E down/up · R to return outside · Orbit restores your overview'
    : orbitHelp;
}
function setFlying(next) {
  if (flying === next) return;
  drag = null;
  canvas.style.cursor = 'grab';
  if (next) {
    orbitView = { rotationX, rotationY, zoom, panX, panY };
    const { forward, right, up } = cameraBasis();
    flyPosition = forward.map((v, i) => -v * 5 / zoom + right[i] * panX + up[i] * panY);
  } else {
    ({ rotationX, rotationY, zoom, panX, panY } = orbitView);
  }
  flying = next;
  updateNavigationUI();
  canvas.focus({ preventScroll: true });
  invalidate();
}
orbitButton.addEventListener('click', () => setFlying(false));
flyButton.addEventListener('click', () => setFlying(true));
cutawayButton.addEventListener('click', () => {
  cutaway = !cutaway;
  if (!cutaway && navigationDistance(flyPosition) <= 0) reset();
  updateNavigationUI();
  canvas.focus({ preventScroll: true });
  invalidate();
});
function updateBubbleSize() {
  // Equal slider travel changes radius by equal ratios, from 0.0001 to 2.
  bubbleRadius = Number((10 ** (Number(bubbleSlider.value) / 100 - 3)).toPrecision(3));
  const label = String(bubbleRadius);
  bubbleValue.textContent = label;
  bubbleSlider.setAttribute('aria-valuetext', label);
}
bubbleSlider.addEventListener('input', () => {
  updateBubbleSize();
  invalidate();
});

// Re-evaluate clearance in small steps to limit large wheel jumps.
// A small speed floor lets you back out of a solid
// region after changing the slice. Cutaway permits crossing solid material,
// with the bubble radius setting a comfortable minimum movement scale.
function moveFly(direction, amount) {
  if (!Number.isFinite(amount)) return;
  const bounded = Math.max(-4, Math.min(4, amount));
  const steps = Math.max(1, Math.ceil(Math.abs(bounded) / 0.1));
  for (let step = 0; step < steps; step++) {
    const bubble = activeBubbleRadius();
    const clearance = Math.max(navigationDistance(flyPosition), bubble);
    const speed = Math.max(bubble ? 0 : 0.002, Math.min(1.5, clearance)) * 0.6;
    const next = flyPosition.map((v, i) => v + direction[i] * speed * bounded / steps);
    if (!bubble && clearance > 0 && navigationDistance(next) <= 0) break;
    flyPosition = next;
  }
}
function travel(amount) { moveFly(cameraBasis().forward, amount); }

function reset() {
  rotationX = 0;
  rotationY = 0;
  zoom = 1;
  if (flying) flyPosition = [0, 0, 5];
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
    if (flying) {
      const { right, up } = cameraBasis();
      moveFly(right, -dx * 0.01);
      moveFly(up, dy * 0.01);
    } else {
      panX -= dx * 0.003 / zoom;
      panY += dy * 0.003 / zoom;
    }
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
  if (flying) travel(-pixels * 0.01);
  else changeZoom(Math.exp(-pixels * 0.001));
  invalidate();
}, { passive: false });
canvas.addEventListener('keydown', (e) => {
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  if (flying && ['w', 's', 'a', 'd', 'q', 'e'].includes(e.key.toLowerCase())) {
    const key = e.key.toLowerCase();
    const basis = cameraBasis();
    const direction = 'ws'.includes(key) ? basis.forward : 'ad'.includes(key) ? basis.right : basis.up;
    moveFly(direction, 'saq'.includes(key) ? -0.2 : 0.2);
    e.preventDefault();
    invalidate();
    return;
  }
  switch (e.key) {
    case 'ArrowLeft': rotationY += flying ? 0.1 : -0.1; break;
    case 'ArrowRight': rotationY += flying ? -0.1 : 0.1; break;
    case 'ArrowUp': rotationX = Math.max(-1.55, rotationX - 0.1); break;
    case 'ArrowDown': rotationX = Math.min(1.55, rotationX + 0.1); break;
    case '+': case '=': flying ? travel(0.2) : changeZoom(1.1); break;
    case '-': case '_': flying ? travel(-0.2) : changeZoom(1 / 1.1); break;
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

  let previewPixels = 600_000;
  let minimumPreviewPixels = 1;
  let detailHeight = 1;
  function sizeCanvas(preview = false) {
    const rect = canvas.getBoundingClientRect();
    // Keep High for the resting image; moving Fly views use a smaller image.
    let scale = Math.min(window.devicePixelRatio || 1, 2) * 2;
    scale = Math.min(scale, Math.sqrt(2_000_000 / Math.max(1, rect.width * rect.height)));
    const limit = device.limits.maxTextureDimension2D;
    scale = Math.min(scale, limit / Math.max(1, rect.width), limit / Math.max(1, rect.height));
    detailHeight = Math.max(1, Math.floor(rect.height * scale));
    // Never magnify a tiny preview across the screen. Preserve at least one
    // render pixel per CSS pixel, subject to the existing GPU/2M pixel cap.
    minimumPreviewPixels = Math.max(1, rect.width * rect.height * Math.min(1, scale) ** 2);
    if (preview) scale = Math.min(scale, Math.sqrt(Math.max(minimumPreviewPixels, previewPixels) / Math.max(1, rect.width * rect.height)));
    const width = Math.max(1, Math.floor(rect.width * scale));
    const height = Math.max(1, Math.floor(rect.height * scale));
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
  }
  resizeCanvas = () => { sizeCanvas(); invalidate(); };
  const observer = new ResizeObserver(resizeCanvas);
  observer.observe(canvas);
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // Uniforms: 5 vec4<f32> = 20 floats
  // data1: resolution.x, resolution.y, sliderVal, orbit zoom (1 in Fly)
  // data2: mode, rotationX, rotationY, clipDist
  // data3: panX, panY, clipRotationX, clipRotationY
  // data4: flyPosition.xyz, flying
  // data5: active bubble radius, full-detail height, reserved.xy
  const uniformBufferSize = 20 * 4;
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const shaderCode = /* wgsl */`
struct Uniforms {
  data1 : vec4<f32>, // res.x, res.y, slider, orbit zoom (1 in Fly)
  data2 : vec4<f32>, // mode, rotationX, rotationY, clipDist
  data3 : vec4<f32>, // panX, panY, clipRotationX, clipRotationY
  data4 : vec4<f32>, // flyPosition.xyz, flying
  data5 : vec4<f32>, // active bubble radius, full-detail height, reserved.xy
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
// Use both orbit derivatives because the visible slice varies initial z as
// well as c. This remains a pseudo-distance, not an exact signed distance.
fn fractal_distance_at(p: vec3<f32>, mode: i32, slider: f32, maxIter: i32) -> f32 {
  let mp = map_to_fractal(p, mode, slider);
  var z  = mp.z0;
  let c  = mp.c;

  let bailout : f32 = 8.0;
  let bailout2 = bailout * bailout;

  // Derivatives with respect to both initial z and parameter c.
  var dz0 = vec2<f32>(1.0, 0.0);
  var dz = vec2<f32>(0.0, 0.0);
  var escaped = false;
  var checkpoint = z;
  var checkpointAge = 0;
  var checkpointPeriod = 1;

  for (var i: i32 = 0; i < maxIter; i = i + 1) {
    let oldz = z;
    let olddz = dz;
    dz0 = 2.0 * c_mul(oldz, dz0);

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
      break;
    }
    if (maxIter > 80) {
      // An exactly repeating f32 orbit cannot subsequently escape. Brent's
      // checkpoint schedule catches stable interiors without a large history.
      if (all(z == checkpoint)) { return 0.0; }
      checkpointAge += 1;
      if (checkpointAge == checkpointPeriod) {
        checkpoint = z;
        checkpointAge = 0;
        checkpointPeriod *= 2;
      }
    }
  }

  if (!escaped) {
    // Deep interior -> treat as "solid" but return a small value
    return 0.0;
  }

  let r = length(z);
  // Frobenius norm of the orbit Jacobian for the three visible real axes.
  // Using only dz/dc can overestimate a step and skip the z0 surface.
  let zWeight = select(1.0, 2.0, mode == 0 || mode == 3);
  let cWeight = 3.0 - zWeight;
  let dr = max(sqrt(zWeight * dot(dz0, dz0) + cWeight * dot(dz, dz)), 1e-6);
  let de = 0.5 * log(r) * r / dr;

  // Clamp for stability
  return clamp(de, 0.0, 1.0);
}

// Coarse field for travel through empty space and the unchanged Orbit view.
fn fractal_distance(p: vec3<f32>, mode: i32, slider: f32) -> f32 {
  return fractal_distance_at(p, mode, slider, 80);
}

// Approximate a third of a pixel at this ray distance (45-degree vertical FOV).
// Keep the resting-image footprint during previews so geometry does not pop
// as the pixel count changes. The floor bounds useful f32 precision.
fn fly_epsilon(rayDistance: f32, point: vec3<f32>) -> f32 {
  let pixelWidth = 0.82842712 * max(rayDistance, 0.0) / max(max(getResolution().y, uniforms.data5.y), 1.0);
  // f32 spacing depends on coordinate magnitude. A fixed 2e-7 floor was
  // several screen pixels wide in tiny cavities near the Mandelbrot cusp.
  let magnitude = max(max(abs(point.x), abs(point.y)), abs(point.z));
  let coordinateFloor = max(0.000000001, magnitude * 0.00000011920929);
  return min(0.001, max(pixelWidth * 0.35, coordinateFloor));
}

fn fly_iterations(epsilon: f32) -> i32 {
  let base = clamp(80.0 + 40.0 * log2(0.001 / epsilon), 80.0, 512.0);
  // Spend the additional budget only below the old close-up accuracy floor.
  let extra = 192.0 * max(0.0, log2(0.0000002 / epsilon));
  return i32(min(1024.0, base + extra));
}

// Escape coloring preserves the original first cycle and extends late escapes.
fn get_escape_time(p: vec3<f32>, mode: i32, slider: f32, maxIter: i32) -> f32 {
  let mp = map_to_fractal(p, mode, slider);
  var z = mp.z0;
  let c = mp.c;
  var checkpoint = z;
  var checkpointAge = 0;
  var checkpointPeriod = 1;

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
      let originalColor = smooth_i / 80.0;
      // Previously all escapes after 80 were clamped to the solid pink color.
      if (maxIter > 80 && originalColor > 1.0) {
        return 1.0 + 0.25 * log2(originalColor);
      }
      return originalColor;
    }
    if (maxIter > 80) {
      if (all(z == checkpoint)) { return 1.0; }
      checkpointAge += 1;
      if (checkpointAge == checkpointPeriod) {
        checkpoint = z;
        checkpointAge = 0;
        checkpointPeriod *= 2;
      }
    }
  }

  // Interior point - return 1.0 (will be colored differently)
  return 1.0;
}

// Estimate normal from distance field via central differences
fn estimate_normal(p: vec3<f32>, mode: i32, slider: f32, zoom: f32, iterations: i32) -> vec3<f32> {
  let eps = 0.003 / zoom;
  let dx = fractal_distance_at(p + vec3<f32>(eps, 0.0, 0.0), mode, slider, iterations)
         - fractal_distance_at(p - vec3<f32>(eps, 0.0, 0.0), mode, slider, iterations);
  let dy = fractal_distance_at(p + vec3<f32>(0.0, eps, 0.0), mode, slider, iterations)
         - fractal_distance_at(p - vec3<f32>(0.0, eps, 0.0), mode, slider, iterations);
  let dz = fractal_distance_at(p + vec3<f32>(0.0, 0.0, eps), mode, slider, iterations)
         - fractal_distance_at(p - vec3<f32>(0.0, 0.0, eps), mode, slider, iterations);
  let gradient = vec3<f32>(dx, dy, dz);
  return gradient / max(length(gradient), 1e-8);
}

fn getClipNormal() -> vec3<f32> {
  let pitch = uniforms.data3.z;
  let yaw = uniforms.data3.w;
  return -vec3<f32>(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw));
}

fn getBubbleRadius() -> f32 {
  return select(0.0, uniforms.data5.x, uniforms.data4.w > 0.5);
}

// Removed material must not cast shadows or occlude either exposed cut.
fn clipped_distance_at(p: vec3<f32>, mode: i32, slider: f32, iterations: i32) -> f32 {
  var cutDistance = 0.0;
  if (getClipDist() > -2.999) {
    cutDistance = max(cutDistance, getClipDist() - dot(p, getClipNormal()));
  }
  let radius = getBubbleRadius();
  if (radius > 0.0) {
    cutDistance = max(cutDistance, radius - distance(p, uniforms.data4.xyz));
  }
  // Inside removed material, the cut alone gives a conservative empty-space
  // distance. No fractal iterations are needed for these lighting samples.
  if (cutDistance > 0.0 && uniforms.data4.w > 0.5) { return cutDistance; }
  return max(cutDistance, fractal_distance_at(p, mode, slider, iterations));
}

fn clipped_distance(p: vec3<f32>, mode: i32, slider: f32) -> f32 {
  return clipped_distance_at(p, mode, slider, 80);
}

// Soft shadow via secondary raymarch
fn soft_shadow(origin: vec3<f32>, lightDir: vec3<f32>, mode: i32, slider: f32, lightDistance: f32, detail: f32, iterations: i32) -> f32 {
  let radius = getBubbleRadius();
  if (radius > 0.0 && distance(origin, uniforms.data4.xyz) < radius &&
      distance(origin + lightDir * lightDistance, uniforms.data4.xyz) < radius) {
    // A segment whose endpoints lie in the spherical cavity is all empty.
    return 1.0;
  }
  var t = 0.02 / detail;
  let maxT = min(4.0 / detail, lightDistance);
  var res: f32 = 1.0;
  let k: f32 = 16.0;

  for (var i: i32 = 0; i < 40; i = i + 1) {
    if (t >= maxT) { break; }
    let p = origin + lightDir * t;
    let d = clipped_distance_at(p, mode, slider, iterations);
    if (d < 0.001 / detail) {
      return 0.0;
    }
    res = min(res, k * d / t);
    t = t + clamp(d, 0.01 / detail, 0.2 / detail);
    if (t > maxT) {
      break;
    }
  }
  return clamp(res, 0.0, 1.0);
}

// Very simple AO-like factor sampling along normal
fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>, mode: i32, slider: f32, detail: f32, iterations: i32) -> f32 {
  var occ: f32 = 0.0;
  let steps: i32 = 4;
  let stepSize: f32 = 0.1 / detail;
  for (var i: i32 = 1; i <= steps; i = i + 1) {
    let t = f32(i) * stepSize;
    let sampleP = p + n * t;
    let d = clipped_distance_at(sampleP, mode, slider, iterations);
    occ = occ + (t - d) * detail;
  }
  occ = occ / f32(steps);
  return clamp(1.0 - 1.5 * occ, 0.0, 1.0);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let res = getResolution();
  let slider = getSlider();
  let flying = uniforms.data4.w > 0.5;
  var detail = getZoom();
  let zoom = select(detail, 1.0, flying);
  let rotX = getRotationX();
  let rotY = getRotationY();
  let clipDist = getClipDist();
  let mode = getMode();
  let pan = select(getPan(), vec2<f32>(0.0), flying);

  // UV -> [-1,1]
  let uv = (in.uv * 2.0 - vec2<f32>(1.0, 1.0));
  let aspect = res.x / res.y;

  // Orbit uses spherical coordinates; Fly supplies a world-space position.
  // rotX = pitch (up/down), rotY = yaw (left/right).
  // Orbit zoom scales fractal space; Fly uses scale 1 and finer precision.
  let camRadius: f32 = 5.0;

  // Build camera basis vectors directly from rotation angles
  // This avoids the singularity when forward is parallel to worldUp
  let sinP = sin(rotX);
  let cosP = cos(rotX);
  let sinY = sin(rotY);
  let cosY = cos(rotY);

  // Camera position and orientation
  let outward = vec3<f32>(cosP * sinY, sinP, cosP * cosY);
  let camPos = select(camRadius * outward, uniforms.data4.xyz, flying);

  // Fly turns in place: viewing direction is independent of camera position.
  let forward = -outward;

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

  // Fly precision follows distance along each ray, not camera clearance.
  let maxDist: f32 = 120.0;
  let maxSteps: i32 = select(i32(200.0 + 100.0 * min(detail, 5.0)), 700, flying);
  var hitIterations: i32 = 80;

  // March from the camera without a fixed world-space cropping box.
  // The step budget and far distance bound the work, not the visible volume.
  let panOffset = right * pan.x + up * pan.y;
  // Every ray starts at the camera-centered bubble boundary. The plane
  // interval below can move the entry farther out or exclude it entirely.
  let bubble = getBubbleRadius();
  var t: f32 = bubble;
  var hit = false;
  var hitPos = vec3<f32>(0.0);

  // The plane lives in fractal coordinates, so orbit, zoom and pan preserve
  // the cut. Only Align to view changes its orientation.
  let clipNormal = getClipNormal();
  var rayEnd = maxDist;
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
    let hitEps = select(0.001 / detail, fly_epsilon(t, pScaled), flying);
    hitIterations = select(80, fly_iterations(hitEps), flying);
    // Early-escaping points still stop immediately. Continuing one orbit to
    // the required budget avoids repeating its first 80 steps at every hit.
    let de = fractal_distance_at(pScaled, mode, slider, hitIterations) * zoom;
    if (de < hitEps) {
      hit = true;
      hitPos = p;
      if (flying) { detail = 0.001 / hitEps; }
      break;
    }

    let stepSize = max(de * 0.7, select(hitEps, hitEps * 0.5, flying));
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
  var n = vec3<f32>(0.0);
  if (bubble > 0.0 && abs(t - bubble) < max(0.00000001, bubble * 0.00001)) {
    // The exposed cavity wall faces inward; its normal is analytic.
    n = normalize(uniforms.data4.xyz - hitPosScaled);
  } else if (clipDist > -2.999 && abs(dot(hitPosScaled, clipNormal) - clipDist) < 0.00001) {
    n = -clipNormal;
  } else {
    n = estimate_normal(hitPosScaled, mode, slider, detail, hitIterations);
  }
  if (dot(n, n) < 0.01) { n = -forward; }

  // Light - positioned behind and above camera
  let viewDir = normalize(camPos - hitPos);
  var lightDir = normalize(viewDir + up * 0.3);
  var lightDistance = 4.0;
  if (bubble > 0.0) {
    // Keep the lamp inside the cavity. Shadow rays stop at the lamp instead
    // of continuing through it and hitting the opposite cavity wall.
    let lightPosition = camPos + up * bubble * 0.3;
    lightDir = normalize(lightPosition - hitPos);
    lightDistance = length(lightPosition - (hitPos + n * 0.01 / detail));
  }
  let diff = max(dot(n, lightDir), 0.0);
  let halfDir = normalize(lightDir + viewDir);
  let spec = pow(max(dot(n, halfDir), 0.0), 40.0);

  // Shadows & AO (use scaled coordinates with pan)
  let sh = soft_shadow(hitPosScaled + n * 0.01 / detail, lightDir, mode, slider, lightDistance, select(1.0, detail, flying), hitIterations);
  let ao = ambient_occlusion(hitPosScaled, n, mode, slider, select(1.0, detail, flying), hitIterations);

  // Get escape time for coloring
  let escapeTime = get_escape_time(hitPosScaled, mode, slider, hitIterations);

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

  const data = new Float32Array(20);
  renderFrame = (preview) => {
    sizeCanvas(preview);
    const started = performance.now();
    data[0] = canvas.width;
    data[1] = canvas.height;
    data[2] = sliderVal;
    data[3] = flying ? 1 : zoom;
    data[4] = mode;
    data[5] = rotationX;
    data[6] = rotationY;
    data[7] = clipVal;
    data[8] = panX;
    data[9] = panY;
    data[10] = clipRotationX;
    data[11] = clipRotationY;
    data.set(flyPosition, 12);
    data[15] = flying ? 1 : 0;
    data[16] = activeBubbleRadius();
    data[17] = detailHeight;
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
    return device.queue.onSubmittedWorkDone().then(() => {
      if (preview) {
        // Adapt only the moving preview. Bound each adjustment and retain the
        // same fractal/ray budgets so movement doesn't switch field definitions.
        const elapsed = Math.max(1, performance.now() - started);
        previewPixels = Math.max(minimumPreviewPixels, Math.min(Math.max(600_000, minimumPreviewPixels),
          previewPixels * Math.max(0.5, Math.min(1.25, 50 / elapsed))));
      }
    });
  };

  if (stopped) return;
  statusEl.textContent = '';
  statusEl.hidden = true;
  invalidate();
}

// Browsers can restore form values on reload or history navigation without
// firing input events. Reset both the controls and renderer state together.
function resetOnLoad() {
  flying = false;
  cutaway = false;
  bubbleSlider.value = '0';
  updateBubbleSize();
  flyPosition = [0, 0, 5];
  orbitView = null;
  drag = null;
  updateNavigationUI();
  clipAnimation = null;
  rotationX = 0;
  rotationY = 0;
  clipRotationX = rotationX;
  clipRotationY = rotationY;
  zoom = 1;
  panX = panY = 0;
  mode = 1;
  sliderVal = 0;
  clipVal = 0;
  modeSelect.value = '1';
  slider.value = '0';
  clipSlider.value = '0';
  sliderValue.textContent = '0.000';
  clipValue.textContent = '0.00';
  slider.ariaLabel = sliderDimNames[mode];
  updateAxisGuide();
  resizeCanvas();
  invalidate();
}
window.addEventListener('pageshow', resetOnLoad);
resetOnLoad();
init().catch(fail);
