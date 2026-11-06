import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { referenceDistance } from './shader-probe.js';

// Exercise scheduling and GPU failure recovery without depending on a driver.
async function viewer({ failPipeline = false, reducedMotion = false, restoredControls = false, width = 360, height = 400, holdGPU = false } = {}) {
  class Element extends EventTarget {
    value = '0';
    textContent = '';
    style = {};
    dataset = {};
    hidden = true;
    width = 1;
    height = 1;
    attributes = {};
    setAttribute(name, value) { this.attributes[name] = String(value); }
    focus() {}
    setPointerCapture() {}
    getBoundingClientRect() { return { width, height }; }
  }
  const ids = ['flyControls', 'cutawayButton', 'bubbleSlider', 'bubbleValue', 'orbitButton', 'flyButton', 'flyReticle', 'help', 'gfx', 'slider', 'sliderValue', 'clipSlider', 'clipValue', 'modeSelect', 'status', 'retryButton', 'resetButton', 'alignClipButton', 'axisGuide', 'axisXLabel', 'axisYLabel', 'axisZLabel', 'axisXArrow', 'axisYArrow', 'axisZArrow'];
  const elements = Object.fromEntries(ids.map((id) => [id, new Element()]));
  elements.slider.min = '-1.5';
  elements.slider.max = '1.5';
  elements.clipSlider.min = '-3';
  elements.clipSlider.max = '3';
  if (restoredControls) {
    elements.slider.value = '0.7';
    elements.clipSlider.value = '1.2';
    elements.modeSelect.value = '2';
  }
  let submitted = 0;
  let uniforms;
  let lose;
  let completeGPU;
  const device = new EventTarget();
  Object.assign(device, {
    limits: { maxTextureDimension2D: 8192 },
    lost: new Promise((resolve) => { lose = resolve; }),
    queue: { writeBuffer(buffer, offset, data) { uniforms = Array.from(new Float32Array(data)); }, submit() { submitted++; }, onSubmittedWorkDone() {
      return holdGPU ? new Promise(resolve => { completeGPU = resolve; }) : Promise.resolve();
    } },
    createBuffer() { return {}; },
    createShaderModule() { return { getCompilationInfo: async () => ({ messages: [] }) }; },
    async createRenderPipelineAsync() {
      if (failPipeline) throw new Error('Invalid test pipeline');
      return { getBindGroupLayout() {} };
    },
    createBindGroup() {},
    createCommandEncoder() {
      return {
        finish() {},
        beginRenderPass() { return { setPipeline() {}, setBindGroup() {}, draw() {}, end() {} }; },
      };
    },
  });
  elements.gfx.getContext = () => ({ configure() {}, getCurrentTexture: () => ({ createView() {} }) });
  const document = new EventTarget();
  Object.assign(document, { hidden: false, getElementById: (id) => elements[id] });
  const frames = new Map();
  let frameId = 0;
  let now = 0;
  const timers = new Map();
  let timerId = 0;
  const window = new EventTarget();
  Object.assign(window, { isSecureContext: true, devicePixelRatio: 2, matchMedia: () => ({ matches: reducedMotion }) });
  const context = vm.createContext({
    document, window, console: { error() {} },
    navigator: { gpu: { requestAdapter: async () => ({ requestDevice: async () => device }), getPreferredCanvasFormat: () => 'bgra8unorm' } },
    GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2 },
    Float32Array, performance: { now: () => now },
    requestAnimationFrame(callback) { const id = ++frameId; frames.set(id, callback); return id; },
    cancelAnimationFrame(id) { frames.delete(id); },
    setTimeout(callback, delay) { const id = ++timerId; timers.set(id, {callback, at: now + delay}); return id; },
    clearTimeout(id) { timers.delete(id); },
    ResizeObserver: class { observe() {} },
  });
  vm.runInContext(await readFile(new URL('../explorer.js', import.meta.url), 'utf8'), context);
  await new Promise((resolve) => setImmediate(resolve));
  return {
    elements, document, window, lose,
    distance(point) { return vm.runInContext(`navigationDistance(${JSON.stringify(point)})`, context); },
    pose(position, pitch = 0, yaw = 0) {
      vm.runInContext(`flyPosition = ${JSON.stringify(position)}; rotationX = ${pitch}; rotationY = ${yaw}; invalidate();`, context);
    },
    get uniforms() { return uniforms; },
    async finishGPU(elapsed = 0) { now += elapsed; completeGPU(); await new Promise(resolve => setImmediate(resolve)); },
    timers(elapsed) { now += elapsed; for (const [id, timer] of timers) if (timer.at <= now) { timers.delete(id); timer.callback(); } },
    key(key, modifiers = {}) { const event = new Event('keydown'); Object.assign(event, { key, ...modifiers }); elements.gfx.dispatchEvent(event); },
    get pending() { return frames.size; },
    get submitted() { return submitted; },
    async frame(elapsed = 0) { now += elapsed; const batch = [...frames.values()]; frames.clear(); batch.forEach((callback) => callback(now)); await new Promise(resolve => setImmediate(resolve)); },
    input(id, value) { elements[id].value = value; elements[id].dispatchEvent(new Event('input')); },
  };
}

test('renders once, stays idle, and coalesces input into one frame', async () => {
  const app = await viewer();
  assert.equal(app.pending, 1);
  await app.frame();
  assert.equal(app.submitted, 1);
  assert.equal(app.pending, 0);
  app.input('slider', '0.3');
  app.input('slider', '0.4');
  app.input('clipSlider', '0.5');
  assert.equal(app.pending, 1);
  await app.frame();
  assert.equal(app.submitted, 2);
  assert.equal(app.pending, 0);
  assert.equal(app.elements.sliderValue.textContent, '0.400');
});

test('both sliders snap pointer input to zero and allow precise keyboard input', async () => {
  const app = await viewer();
  for (const [id, output, precision, uniform, edge, step] of [
    ['slider', 'sliderValue', 3, 2, 0.06, 0.001],
    ['clipSlider', 'clipValue', 2, 7, 0.12, 0.01],
  ]) {
    const control = app.elements[id];
    control.dispatchEvent(new Event('pointerdown'));
    // The previous slider blurs after pointerdown when focus changes.
    app.elements[id === 'slider' ? 'clipSlider' : 'slider'].dispatchEvent(new Event('blur'));
    for (const value of [-edge, -step, 0, step, edge]) {
      app.input(id, String(value));
      await app.frame();
      assert.equal(control.value, '0');
      assert.equal(app.elements[output].textContent, (0).toFixed(precision));
      assert.equal(app.uniforms[uniform], 0);
    }
    for (const value of [-edge - step, edge + step]) {
      app.input(id, String(value));
      await app.frame();
      assert.equal(Number(control.value), value);
      assert.ok(Math.abs(app.uniforms[uniform] - value) < 1e-7);
    }
    for (const endEvent of ['pointerup', 'pointercancel', 'blur']) {
      control.dispatchEvent(new Event('pointerdown'));
      app.window.dispatchEvent(new Event(endEvent));
      app.input(id, String(step));
      await app.frame();
      assert.equal(Number(control.value), step);
      assert.ok(Math.abs(app.uniforms[uniform] - step) < 1e-7);
    }
    control.dispatchEvent(new Event('pointerdown'));
    control.dispatchEvent(new Event('keydown'));
    app.input(id, String(-step));
    await app.frame();
    assert.equal(Number(control.value), -step);
    assert.ok(Math.abs(app.uniforms[uniform] + step) < 1e-7);
  }
});

test('uses High render scale and caps large canvases at two million pixels', async () => {
  const small = await viewer({ width: 200, height: 150 });
  await small.frame();
  assert.deepEqual(small.uniforms.slice(0, 2), [800, 600]);
  const large = await viewer({ width: 1200, height: 800 });
  await large.frame();
  const [width, height] = large.uniforms;
  assert.ok(width * height <= 2_000_000);
  assert.ok(width * height > 1_990_000);
});

test('hidden tabs defer rendering until visible', async () => {
  const app = await viewer();
  await app.frame();
  app.document.hidden = true;
  app.input('slider', '0.2');
  assert.equal(app.pending, 0);
  app.document.hidden = false;
  app.document.dispatchEvent(new Event('visibilitychange'));
  assert.equal(app.pending, 1);
  await app.frame();
  assert.equal(app.submitted, 2);
});

test('GPU loss cancels queued work and exposes recovery', async () => {
  const app = await viewer();
  app.lose({ message: 'Test device loss' });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(app.pending, 0);
  assert.match(app.elements.status.textContent, /Test device loss/);
  assert.equal(app.elements.retryButton.hidden, false);
  app.input('slider', '0.5');
  assert.equal(app.pending, 0);
});

test('pipeline rejection becomes a visible error with no scheduled work', async () => {
  const app = await viewer({ failPipeline: true });
  assert.equal(app.pending, 0);
  assert.equal(app.elements.status.dataset.error, 'true');
  assert.match(app.elements.status.textContent, /Invalid test pipeline/);
  assert.equal(app.elements.retryButton.hidden, false);
});


test('reset restores camera orientation and zoom while preserving slice, cut, and pan', async () => {
  const app = await viewer();
  app.input('clipSlider', '0.25');
  app.input('slider', '0.4');
  app.elements.modeSelect.value = '2';
  app.elements.modeSelect.dispatchEvent(new Event('change'));
  // Pan through the same pointer events used by Shift-drag.
  for (const [type, props] of [
    ['pointerdown', { button: 0, shiftKey: true, pointerId: 1, clientX: 0, clientY: 0 }],
    ['pointermove', { pointerId: 1, clientX: 20, clientY: 30 }],
    ['pointerup', { pointerId: 1 }],
  ]) {
    const event = new Event(type);
    Object.assign(event, props);
    app.elements.gfx.dispatchEvent(event);
  }
  await app.frame();
  const initial = app.uniforms.slice(10, 12);
  app.key('ArrowRight');
  app.key('ArrowDown');
  app.key('+');
  await app.frame();
  assert.deepEqual(app.uniforms.slice(10, 12), initial);
  assert.notDeepEqual(app.uniforms.slice(5, 7), initial);
  app.elements.alignClipButton.dispatchEvent(new Event('click'));
  await app.frame(450);
  assert.deepEqual(app.uniforms.slice(10, 12), app.uniforms.slice(5, 7));
  assert.equal(app.uniforms[7], 0.25);
  app.key('ArrowRight');
  await app.frame();
  assert.notDeepEqual(app.uniforms.slice(10, 12), app.uniforms.slice(5, 7));
  const before = [...app.uniforms];
  for (const trigger of [() => app.elements.resetButton.dispatchEvent(new Event('click')), () => app.key('r')]) {
    app.key('ArrowRight');
    app.key('+');
    await app.frame();
    assert.notEqual(app.uniforms[3], 1);
    trigger();
    await app.frame();
    assert.deepEqual(app.uniforms.slice(5, 7), initial);
    assert.equal(app.uniforms[3], 1);
    for (const index of [2, 4, 7, 8, 9, 10, 11]) assert.equal(app.uniforms[index], before[index]);
    assert.equal(app.elements.modeSelect.value, '2');
    assert.equal(app.elements.slider.value, '0.4');
    assert.equal(app.elements.clipSlider.value, '0.25');
  }
});


test('alignment eases between frames, keeps depth, and stops rendering on completion', async () => {
  const app = await viewer();
  app.input('clipSlider', '0.4');
  app.key('ArrowRight');
  app.key('ArrowRight');
  await app.frame();
  app.elements.alignClipButton.dispatchEvent(new Event('click'));
  await app.frame();
  assert.equal(app.uniforms[11], 0);
  assert.equal(app.pending, 1);
  await app.frame(225);
  assert.ok(Math.abs(app.uniforms[11] - 0.1) < 1e-6);
  assert.ok(Math.abs(app.uniforms[7] - 0.4) < 1e-6);
  await app.frame(225);
  assert.ok(Math.abs(app.uniforms[11] - 0.2) < 1e-6);
  assert.equal(app.pending, 0);
});

test('alignment takes the short arc, can be retargeted, and survives camera reset', async () => {
  const app = await viewer();
  for (let i = 0; i < 64; i++) app.key('ArrowRight');
  await app.frame();
  app.elements.alignClipButton.dispatchEvent(new Event('click'));
  await app.frame(225);
  assert.ok(app.uniforms[11] > 0 && app.uniforms[11] < 0.1);
  const midway = app.uniforms[11];
  app.key('ArrowRight');
  app.elements.alignClipButton.dispatchEvent(new Event('click'));
  await app.frame();
  assert.equal(app.uniforms[11], midway);
  await app.frame(450);
  assert.ok(Math.abs(app.uniforms[11] - (6.5 - 2 * Math.PI)) < 1e-6);
  assert.equal(app.pending, 0);
  app.key('ArrowRight');
  app.elements.alignClipButton.dispatchEvent(new Event('click'));
  await app.frame(100);
  app.elements.resetButton.dispatchEvent(new Event('click'));
  await app.frame();
  assert.equal(app.uniforms[6], 0);
  assert.equal(app.pending, 1);
  await app.frame(350);
  assert.ok(Math.abs(app.uniforms[11] - (6.6 - 2 * Math.PI)) < 1e-6);
  assert.equal(app.pending, 0);
});

test('reduced motion aligns in one frame', async () => {
  const app = await viewer({ reducedMotion: true });
  app.key('ArrowRight');
  app.elements.alignClipButton.dispatchEvent(new Event('click'));
  await app.frame();
  assert.equal(app.uniforms[11], app.uniforms[6]);
  assert.equal(app.pending, 0);
});


test('axis guide projects world axes through the camera basis and ignores zoom', async () => {
  const app = await viewer();
  const arrow = (axis) => app.elements[`axis${axis}Arrow`].attributes.d;
  const initialX = arrow('X');
  assert.match(initialX, /^M80 70 L116 70 /);
  assert.match(arrow('Y'), /^M80 70 L80 34 /);
  assert.equal(arrow('Z'), 'M80 70 L80 70');
  assert.deepEqual(['X', 'Y', 'Z'].map(axis => app.elements[`axis${axis}Label`].textContent), ['Re(c)', 'Im(c)', 'Im(z₀)']);
  for (let i = 0; i < 16; i++) app.key('ArrowRight');
  const zLabel = app.elements.axisZLabel.attributes;
  assert.ok(Number(zLabel.x) < 30, 'positive Z points left after a quarter orbit');
  assert.notEqual(arrow('X'), initialX);
  const rotated = arrow('Z');
  app.key('+');
  assert.equal(arrow('Z'), rotated);
  app.elements.resetButton.dispatchEvent(new Event('click'));
  assert.equal(arrow('X'), initialX);
});


test('page load discards restored controls and keeps the first slider input in sync', async () => {
  const app = await viewer({ restoredControls: true });
  await app.frame();
  assert.equal(app.elements.slider.value, '0');
  assert.equal(app.elements.clipSlider.value, '0');
  assert.equal(app.elements.modeSelect.value, '1');
  assert.equal(app.uniforms[2], 0);
  assert.equal(app.uniforms[4], 1);
  assert.deepEqual(app.uniforms.slice(5, 7), [0, 0]);
  assert.deepEqual(app.uniforms.slice(10, 12), [0, 0]);
  assert.equal(app.uniforms[7], 0);
  app.input('slider', '0.001');
  await app.frame();
  assert.ok(Math.abs(app.uniforms[2] - 0.001) < 1e-8);
  app.input('clipSlider', '1');
  app.key('ArrowRight');
  app.window.dispatchEvent(new Event('pageshow'));
  await app.frame();
  assert.equal(app.uniforms[2], 0);
  assert.equal(app.uniforms[6], 0);
  assert.equal(app.uniforms[7], 0);
});


test('navigation distance matches independent orbit derivatives and honors clipping', async () => {
  const app = await viewer();
  app.input('slider', '0.35');
  app.input('clipSlider', '-3');
  for (let mode = 0; mode < 4; mode++) {
    app.elements.modeSelect.value = String(mode);
    app.elements.modeSelect.dispatchEvent(new Event('change'));
    assert.ok(Math.abs(app.distance([1.7, 0.2, -0.6]) - referenceDistance(mode)) < 1e-6);
  }
  app.input('clipSlider', '0');
  assert.equal(app.distance([0, 0, 5]), 5, 'removed half-space is empty for navigation');
});

test('Fly enters from the orbit pose, turns in place, strafes, and restores the overview', async () => {
  const app = await viewer();
  app.key('ArrowRight');
  app.key('ArrowUp');
  app.key('+');
  for (const [type, props] of [
    ['pointerdown', { button: 2, pointerId: 1, clientX: 0, clientY: 0 }],
    ['pointermove', { pointerId: 1, clientX: 60, clientY: 40 }],
    ['pointerup', { pointerId: 1 }],
  ]) {
    const event = new Event(type);
    Object.assign(event, props);
    app.elements.gfx.dispatchEvent(event);
  }
  await app.frame();
  const orbit = [...app.uniforms];
  const [pitch, yaw] = orbit.slice(5, 7);
  const radius = 5 / orbit[3];
  const expected = [
    radius * Math.cos(pitch) * Math.sin(yaw) + Math.cos(yaw) * orbit[8] - Math.sin(pitch) * Math.sin(yaw) * orbit[9],
    radius * Math.sin(pitch) + Math.cos(pitch) * orbit[9],
    radius * Math.cos(pitch) * Math.cos(yaw) - Math.sin(yaw) * orbit[8] - Math.sin(pitch) * Math.cos(yaw) * orbit[9],
  ];
  app.elements.flyButton.dispatchEvent(new Event('click'));
  await app.frame();
  assert.equal(app.uniforms[15], 1);
  expected.forEach((v, i) => assert.ok(Math.abs(app.uniforms[12 + i] - v) < 1e-6));
  assert.equal(app.elements.flyButton.attributes['aria-pressed'], 'true');
  assert.equal(app.elements.flyReticle.hidden, false);
  const position = app.uniforms.slice(12, 15);
  for (const modifier of ['ctrlKey', 'metaKey', 'altKey']) app.key('w', { [modifier]: true });
  await app.frame();
  assert.deepEqual(app.uniforms.slice(12, 15), position, 'browser shortcuts do not move the camera');
  app.key('ArrowRight');
  await app.frame();
  assert.deepEqual(app.uniforms.slice(12, 15), position, 'looking does not orbit the camera');
  assert.ok(app.uniforms[6] < yaw, 'right arrow looks right in Fly');
  app.key('a');
  await app.frame();
  assert.ok(app.uniforms[12] < position[0], 'A strafes left');
  app.elements.orbitButton.dispatchEvent(new Event('click'));
  await app.frame();
  assert.deepEqual(app.uniforms.slice(0, 12), orbit.slice(0, 12));
  assert.equal(app.uniforms[15], 0);
  assert.equal(app.elements.flyReticle.hidden, true);
  assert.equal(app.pending, 0);
});

test('flight slows near the cut, backs out, and resets safely', async () => {
  const app = await viewer();
  app.elements.flyButton.dispatchEvent(new Event('click'));
  await app.frame();
  const start = app.uniforms[14];
  app.key('w');
  await app.frame();
  const firstStep = start - app.uniforms[14];
  const wheel = new Event('wheel');
  Object.assign(wheel, { deltaY: -100000, deltaMode: 0 });
  app.elements.gfx.dispatchEvent(wheel);
  await app.frame();
  assert.ok(app.uniforms[14] > 0, 'large wheel events do not jump through the cut');
  for (let i = 0; i < 30; i++) app.key('w');
  await app.frame();
  const near = app.uniforms[14];
  app.key('w');
  await app.frame();
  assert.ok(near - app.uniforms[14] < firstStep / 10);
  assert.equal(app.uniforms[3], 1, 'Fly precision is chosen per ray in the shader');
  const closer = app.uniforms[14];
  app.key('s');
  await app.frame();
  assert.ok(app.uniforms[14] > closer, 'backward movement remains available');
  app.input('slider', '0.2');
  app.input('clipSlider', '0.3');
  for (const reset of [() => app.key('r'), () => app.elements.resetButton.dispatchEvent(new Event('click'))]) {
    app.key('w');
    app.key('ArrowRight');
    reset();
    await app.frame();
    assert.deepEqual(app.uniforms.slice(12, 15), [0, 0, 5]);
    assert.deepEqual(app.uniforms.slice(5, 7), [0, 0]);
    assert.equal(app.uniforms[15], 1);
    assert.equal(app.elements.slider.value, '0.2');
    assert.equal(app.elements.clipSlider.value, '0.3');
  }
  app.window.dispatchEvent(new Event('pageshow'));
  await app.frame();
  assert.equal(app.uniforms[15], 0);
  assert.equal(app.elements.flyReticle.hidden, true);
});


test('Cutaway crosses solid material, changes size without moving, and is inactive in Orbit', async () => {
  const app = await viewer();
  assert.equal(app.elements.flyControls.hidden, true);
  app.elements.flyButton.dispatchEvent(new Event('click'));
  assert.equal(app.elements.flyControls.hidden, false);
  assert.equal(app.elements.bubbleSlider.disabled, true);
  app.elements.cutawayButton.dispatchEvent(new Event('click'));
  await app.frame();
  assert.equal(app.elements.bubbleSlider.disabled, false);
  assert.ok(Math.abs(app.uniforms[16] - 0.001) < 1e-9);
  app.input('bubbleSlider', '300');
  // Enter the retained solid side of the default Mandelbrot cut.
  for (let i = 0; i < 100 && app.uniforms[14] > -0.3; i++) {
    app.key('w');
    await app.frame();
  }
  const position = app.uniforms.slice(12, 15);
  assert.ok(position[2] < -0.2 && position[2] > -0.9);
  assert.equal(app.distance(position), 0);
  app.input('bubbleSlider', '200');
  await app.frame();
  assert.deepEqual(app.uniforms.slice(12, 15), position);
  assert.ok(Math.abs(app.uniforms[16] - 0.1) < 1e-7);
  assert.equal(app.elements.bubbleValue.textContent, '0.1');
  app.key('w');
  await app.frame();
  const smallStep = position[2] - app.uniforms[14];
  app.input('bubbleSlider', '300');
  await app.frame();
  const before = app.uniforms[14];
  app.key('w');
  await app.frame();
  assert.ok(Math.abs((before - app.uniforms[14]) / smallStep - 10) < 0.001);
  app.elements.cutawayButton.dispatchEvent(new Event('click'));
  await app.frame();
  assert.equal(app.uniforms[16], 0);
  assert.deepEqual(app.uniforms.slice(12, 15), [0, 0, 5], 'disabling inside solid returns outside');
  app.elements.cutawayButton.dispatchEvent(new Event('click'));
  app.input('bubbleSlider', '100');
  app.key('r');
  await app.frame();
  assert.ok(Math.abs(app.uniforms[16] - 0.01) < 1e-9, 'Reset view preserves Cutaway settings');
  app.elements.orbitButton.dispatchEvent(new Event('click'));
  await app.frame();
  assert.equal(app.uniforms[16], 0);
  assert.equal(app.elements.flyControls.hidden, true);
  app.elements.flyButton.dispatchEvent(new Event('click'));
  await app.frame();
  assert.ok(Math.abs(app.uniforms[16] - 0.01) < 1e-9, 'returning to Fly restores the bubble setting');
  app.window.dispatchEvent(new Event('pageshow'));
  await app.frame();
  assert.equal(app.uniforms[16], 0);
  assert.equal(app.elements.cutawayButton.attributes['aria-pressed'], 'false');
  assert.equal(app.elements.bubbleSlider.value, '0');
});


test('logarithmic bubble sizes retain tiny values with matching movement speed', async () => {
  const app = await viewer();
  app.elements.flyButton.dispatchEvent(new Event('click'));
  app.elements.cutawayButton.dispatchEvent(new Event('click'));
  for (const [position, radius] of [[-100, 0.0001], [-99, 0.000102], [0, 0.001], [1, 0.00102], [100, 0.01], [200, 0.1], [300, 1], [330, 2]]) {
    app.input('bubbleSlider', String(position));
    await app.frame();
    assert.ok(Math.abs(app.uniforms[16] - radius) < 1e-7);
    assert.equal(app.elements.bubbleValue.textContent, String(radius));
    assert.equal(app.elements.bubbleSlider.attributes['aria-valuetext'], String(radius));
  }
  app.input('bubbleSlider', '300');
  await app.frame();
  for (let i = 0; i < 100 && app.uniforms[14] > -0.3; i++) {
    app.key('w');
    await app.frame();
  }
  const before = app.uniforms.slice(12, 15);
  assert.equal(app.distance(before), 0);
  app.input('bubbleSlider', '0');
  await app.frame();
  assert.deepEqual(app.uniforms.slice(12, 15), before, 'resizing does not move the camera');
  assert.equal(app.uniforms[3], 1, 'bubble size does not impose a view-wide detail level');
  app.key('w');
  await app.frame();
  assert.ok(Math.abs((before[2] - app.uniforms[14]) - 0.00012) < 1e-7, 'movement scales to the tiny radius');
  const smallerStart = app.uniforms.slice(12, 15);
  app.input('bubbleSlider', '-100');
  await app.frame();
  assert.deepEqual(app.uniforms.slice(12, 15), smallerStart);
  app.key('w');
  await app.frame();
  assert.ok(Math.abs((smallerStart[2] - app.uniforms[14]) - 0.000012) < 1e-7, 'the new minimum also reduces movement tenfold');
});


test('slow GPU coalesces repeated movement into the newest pose without a frame backlog', async () => {
  const app = await viewer({ holdGPU: true });
  await app.frame();
  assert.equal(app.submitted, 1);
  app.elements.flyButton.dispatchEvent(new Event('click'));
  app.elements.cutawayButton.dispatchEvent(new Event('click'));
  for (let i = 0; i < 20; i++) { app.key('ArrowRight'); app.key('w'); await app.frame(); }
  assert.equal(app.submitted, 1, 'no extra work is submitted while the GPU is busy');
  await app.finishGPU(100);
  assert.equal(app.pending, 1);
  await app.frame();
  assert.equal(app.submitted, 2);
  assert.ok(Math.abs(app.uniforms[6] + 2) < 1e-6, 'the next frame has the latest orientation');
  assert.ok(app.uniforms[0] * app.uniforms[1] <= 600_000, 'moving Fly frame is lightweight');
  const detailHeight = app.uniforms[17];
  await app.finishGPU(200);
  app.key('w');
  await app.frame();
  assert.ok(app.uniforms[0] * app.uniforms[1] <= 300_000, 'slow previews reduce their pixel budget');
  await app.finishGPU(50);
  app.timers(240);
  await app.frame();
  assert.ok(app.uniforms[0] * app.uniforms[1] > 1_990_000, 'pausing restores High detail');
  assert.equal(app.uniforms[17], detailHeight, 'preview and resting view use the same geometric precision');
  await app.finishGPU(1000);
  assert.equal(app.pending, 0, 'refinement stops when complete');
});

test('fresh input cancels idle refinement and hidden tabs submit no GPU work', async () => {
  const app = await viewer();
  app.elements.flyButton.dispatchEvent(new Event('click'));
  await app.frame();
  app.timers(240); // A high-detail frame is requested, but has not started.
  app.key('w');
  await app.frame();
  assert.ok(app.uniforms[0] * app.uniforms[1] <= 600_000);
  app.document.hidden = true;
  app.document.dispatchEvent(new Event('visibilitychange'));
  const submitted = app.submitted;
  app.timers(1000);
  await app.frame();
  assert.equal(app.submitted, submitted);
});

test('W preserves a fixed landmark at the reticle across orientations and bubble sizes', async () => {
  const app = await viewer();
  app.elements.flyButton.dispatchEvent(new Event('click'));
  app.elements.cutawayButton.dispatchEvent(new Event('click'));
  // A fixed landmark placed along the viewing axis must remain centered after
  // forward travel. Test projections independently of the movement function.
  for (const radius of ['0', '200', '330']) {
    app.input('bubbleSlider', radius);
    for (let turn = 0; turn < 8; turn++) {
      app.key('ArrowRight'); app.key('ArrowUp');
      await app.frame();
      const u = app.uniforms, pitch = u[5], yaw = u[6];
      const forward = [-Math.cos(pitch)*Math.sin(yaw), -Math.sin(pitch), -Math.cos(pitch)*Math.cos(yaw)];
      const right = [Math.cos(yaw), 0, -Math.sin(yaw)];
      const up = [-Math.sin(pitch)*Math.sin(yaw), Math.cos(pitch), -Math.sin(pitch)*Math.cos(yaw)];
      const before = u.slice(12,15);
      const landmark = before.map((v,i) => v + 10*forward[i]);
      app.key('w'); await app.frame();
      const after = app.uniforms.slice(12,15);
      const delta = after.map((v,i) => v-before[i]);
      const dot = (a,b) => a.reduce((sum,v,i) => sum+v*b[i],0);
      assert.ok(dot(delta,forward)>0, 'W travels forward');
      const toLandmark = landmark.map((v,i)=>v-after[i]);
      assert.ok(Math.abs(dot(toLandmark,right))<2e-6, 'no horizontal drift of fixed geometry');
      assert.ok(Math.abs(dot(toLandmark,up))<2e-6, 'no vertical drift of fixed geometry');
    }
  }
});


test('slow Fly previews never fall below screen resolution', async () => {
  const app = await viewer({ holdGPU: true, width: 1200, height: 800 });
  app.elements.flyButton.dispatchEvent(new Event('click'));
  for (let i = 0; i < 8; i++) {
    app.key('w'); await app.frame();
    assert.ok(app.uniforms[0] >= 1200 && app.uniforms[1] >= 800);
    assert.ok(app.uniforms[0] * app.uniforms[1] <= 2_000_000);
    await app.finishGPU(1000);
  }
});


test('ordinary Fly recognizes late-escaping space and keeps approaching a wall with no bubble', async () => {
  const app = await viewer();
  app.elements.flyButton.dispatchEvent(new Event('click'));
  app.input('clipSlider', '-3');
  assert.ok(app.distance([0.2504, 0, 0]) > 0, 'space escaping after 80 iterations is navigable');
  assert.ok(app.distance([0.250025, 0, 0]) > 0, 'space escaping after 512 iterations is navigable');
  assert.ok(app.distance([0.250005, 0, 0]) > 0, 'space escaping after 1024 iterations is navigable');
  assert.equal(app.distance([0.25, 0, 0]), 0, 'solid material is still solid');
  app.pose([0.2516, 0, 0], 0, Math.PI / 2);
  await app.frame();
  app.key('w'); await app.frame();
  assert.ok(app.uniforms[12] < 0.2516, 'no invisible stop at the coarse iteration boundary');

  app.input('clipSlider', '0');
  const finalPositions = [];
  for (const bubbleSetting of ['-100', '330']) {
    app.input('bubbleSlider', bubbleSetting);
    app.pose([0, 0, 0.00001]);
    await app.frame();
    for (let step = 0; step < 100; step++) app.key('w');
    await app.frame();
    assert.equal(app.uniforms[16], 0, 'Cutaway off never supplies a bubble radius');
    assert.ok(app.uniforms[14] > 0, 'ordinary Fly does not cross into solid material');
    assert.ok(app.uniforms[14] < 1e-9, 'small clearances do not get stuck at the minimum movement step');
    finalPositions.push(app.uniforms.slice(12, 15));
  }
  assert.deepEqual(finalPositions[0], finalPositions[1], 'stored bubble size cannot affect ordinary flight');
});
