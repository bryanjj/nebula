import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

// Exercise scheduling and GPU failure recovery without depending on a driver.
async function viewer({ failPipeline = false, reducedMotion = false } = {}) {
  class Element extends EventTarget {
    value = '0';
    textContent = '';
    style = {};
    dataset = {};
    hidden = true;
    width = 1;
    height = 1;
    focus() {}
    setPointerCapture() {}
    getBoundingClientRect() { return { width: 360, height: 400 }; }
  }
  const ids = ['gfx', 'slider', 'sliderValue', 'clipSlider', 'clipValue', 'modeSelect', 'status', 'qualitySelect', 'retryButton', 'resetButton', 'alignClipButton'];
  const elements = Object.fromEntries(ids.map((id) => [id, new Element()]));
  elements.qualitySelect.value = '1';
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
    elements, document, lose,
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


test('clip orientation stays fixed through orbit and zoom until aligned, and reset restores it', async () => {
  const app = await viewer();
  app.input('clipSlider', '0.25');
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
  app.elements.resetButton.dispatchEvent(new Event('click'));
  app.frame();
  assert.deepEqual(app.uniforms.slice(10, 12), initial);
  assert.equal(app.uniforms[7], -3);
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

test('alignment takes the short arc after a full orbit and can be retargeted or reset', async () => {
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
  assert.equal(app.uniforms[11], 0);
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
