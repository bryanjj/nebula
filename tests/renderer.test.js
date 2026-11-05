import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

// Exercise scheduling and GPU failure recovery without depending on a driver.
async function viewer({ failPipeline = false, reducedMotion = false, restoredControls = false, width = 360, height = 400 } = {}) {
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
  const ids = ['gfx', 'slider', 'sliderValue', 'clipSlider', 'clipValue', 'modeSelect', 'status', 'retryButton', 'resetButton', 'alignClipButton', 'axisGuide', 'axisXLabel', 'axisYLabel', 'axisZLabel', 'axisXArrow', 'axisYArrow', 'axisZArrow'];
  const elements = Object.fromEntries(ids.map((id) => [id, new Element()]));
  if (restoredControls) {
    elements.slider.value = '0.7';
    elements.clipSlider.value = '1.2';
    elements.modeSelect.value = '2';
  }
  let submitted = 0;
  let uniforms;
  let lose;
  const device = new EventTarget();
  Object.assign(device, {
    limits: { maxTextureDimension2D: 8192 },
    lost: new Promise((resolve) => { lose = resolve; }),
    queue: { writeBuffer(buffer, offset, data) { uniforms = Array.from(new Float32Array(data)); }, submit() { submitted++; } },
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
  const window = new EventTarget();
  Object.assign(window, { isSecureContext: true, devicePixelRatio: 2, matchMedia: () => ({ matches: reducedMotion }) });
  const context = vm.createContext({
    document, window, console: { error() {} },
    navigator: { gpu: { requestAdapter: async () => ({ requestDevice: async () => device }), getPreferredCanvasFormat: () => 'bgra8unorm' } },
    GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2 },
    Float32Array, performance: { now: () => now },
    requestAnimationFrame(callback) { const id = ++frameId; frames.set(id, callback); return id; },
    cancelAnimationFrame(id) { frames.delete(id); },
    ResizeObserver: class { observe() {} },
  });
  vm.runInContext(await readFile(new URL('../explorer.js', import.meta.url), 'utf8'), context);
  await new Promise((resolve) => setImmediate(resolve));
  return {
    elements, document, window, lose,
    get uniforms() { return uniforms; },
    key(key) { const event = new Event('keydown'); Object.assign(event, { key }); elements.gfx.dispatchEvent(event); },
    get pending() { return frames.size; },
    get submitted() { return submitted; },
    frame(elapsed = 0) { now += elapsed; const batch = [...frames.values()]; frames.clear(); batch.forEach((callback) => callback(now)); },
    input(id, value) { elements[id].value = value; elements[id].dispatchEvent(new Event('input')); },
  };
}

test('renders once, stays idle, and coalesces input into one frame', async () => {
  const app = await viewer();
  assert.equal(app.pending, 1);
  app.frame();
  assert.equal(app.submitted, 1);
  assert.equal(app.pending, 0);
  app.input('slider', '0.3');
  app.input('slider', '0.4');
  app.input('clipSlider', '0.5');
  assert.equal(app.pending, 1);
  app.frame();
  assert.equal(app.submitted, 2);
  assert.equal(app.pending, 0);
  assert.equal(app.elements.sliderValue.textContent, '0.400');
});

test('uses High render scale and caps large canvases at two million pixels', async () => {
  const small = await viewer({ width: 200, height: 150 });
  small.frame();
  assert.deepEqual(small.uniforms.slice(0, 2), [800, 600]);
  const large = await viewer({ width: 1200, height: 800 });
  large.frame();
  const [width, height] = large.uniforms;
  assert.ok(width * height <= 2_000_000);
  assert.ok(width * height > 1_990_000);
});

test('hidden tabs defer rendering until visible', async () => {
  const app = await viewer();
  app.frame();
  app.document.hidden = true;
  app.input('slider', '0.2');
  assert.equal(app.pending, 0);
  app.document.hidden = false;
  app.document.dispatchEvent(new Event('visibilitychange'));
  assert.equal(app.pending, 1);
  app.frame();
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
  app.frame();
  const initial = app.uniforms.slice(10, 12);
  app.key('ArrowRight');
  app.key('ArrowDown');
  app.key('+');
  app.frame();
  assert.deepEqual(app.uniforms.slice(10, 12), initial);
  assert.notDeepEqual(app.uniforms.slice(5, 7), initial);
  app.elements.alignClipButton.dispatchEvent(new Event('click'));
  app.frame(450);
  assert.deepEqual(app.uniforms.slice(10, 12), app.uniforms.slice(5, 7));
  assert.equal(app.uniforms[7], 0.25);
  app.key('ArrowRight');
  app.frame();
  assert.notDeepEqual(app.uniforms.slice(10, 12), app.uniforms.slice(5, 7));
  const before = [...app.uniforms];
  for (const trigger of [() => app.elements.resetButton.dispatchEvent(new Event('click')), () => app.key('r')]) {
    app.key('ArrowRight');
    app.key('+');
    app.frame();
    assert.notEqual(app.uniforms[3], 1);
    trigger();
    app.frame();
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
  app.frame();
  app.elements.alignClipButton.dispatchEvent(new Event('click'));
  app.frame();
  assert.equal(app.uniforms[11], 0);
  assert.equal(app.pending, 1);
  app.frame(225);
  assert.ok(Math.abs(app.uniforms[11] - 0.1) < 1e-6);
  assert.ok(Math.abs(app.uniforms[7] - 0.4) < 1e-6);
  app.frame(225);
  assert.ok(Math.abs(app.uniforms[11] - 0.2) < 1e-6);
  assert.equal(app.pending, 0);
});

test('alignment takes the short arc, can be retargeted, and survives camera reset', async () => {
  const app = await viewer();
  for (let i = 0; i < 64; i++) app.key('ArrowRight');
  app.frame();
  app.elements.alignClipButton.dispatchEvent(new Event('click'));
  app.frame(225);
  assert.ok(app.uniforms[11] > 0 && app.uniforms[11] < 0.1);
  const midway = app.uniforms[11];
  app.key('ArrowRight');
  app.elements.alignClipButton.dispatchEvent(new Event('click'));
  app.frame();
  assert.equal(app.uniforms[11], midway);
  app.frame(450);
  assert.ok(Math.abs(app.uniforms[11] - (6.5 - 2 * Math.PI)) < 1e-6);
  assert.equal(app.pending, 0);
  app.key('ArrowRight');
  app.elements.alignClipButton.dispatchEvent(new Event('click'));
  app.frame(100);
  app.elements.resetButton.dispatchEvent(new Event('click'));
  app.frame();
  assert.equal(app.uniforms[6], 0);
  assert.equal(app.pending, 1);
  app.frame(350);
  assert.ok(Math.abs(app.uniforms[11] - (6.6 - 2 * Math.PI)) < 1e-6);
  assert.equal(app.pending, 0);
});

test('reduced motion aligns in one frame', async () => {
  const app = await viewer({ reducedMotion: true });
  app.key('ArrowRight');
  app.elements.alignClipButton.dispatchEvent(new Event('click'));
  app.frame();
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
  app.frame();
  assert.equal(app.elements.slider.value, '0');
  assert.equal(app.elements.clipSlider.value, '0');
  assert.equal(app.elements.modeSelect.value, '1');
  assert.equal(app.uniforms[2], 0);
  assert.equal(app.uniforms[4], 1);
  assert.deepEqual(app.uniforms.slice(5, 7), [0, 0]);
  assert.deepEqual(app.uniforms.slice(10, 12), [0, 0]);
  assert.equal(app.uniforms[7], 0);
  app.input('slider', '0.001');
  app.frame();
  assert.ok(Math.abs(app.uniforms[2] - 0.001) < 1e-8);
  app.input('clipSlider', '1');
  app.key('ArrowRight');
  app.window.dispatchEvent(new Event('pageshow'));
  app.frame();
  assert.equal(app.uniforms[2], 0);
  assert.equal(app.uniforms[6], 0);
  assert.equal(app.uniforms[7], 0);
});
