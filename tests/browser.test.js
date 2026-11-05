import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { probeDistances, probeCutaway, probeAdaptiveDetail, referenceDistance } from './shader-probe.js';
import { chromium } from 'playwright';

let server;
let browser;
const origin = 'http://127.0.0.1:8766';

before(async () => {
  server = spawn(process.env.PYTHON || 'python3', ['-m', 'http.server', '8766', '--bind', '127.0.0.1'], { stdio: 'ignore' });
  for (let attempt = 0; attempt < 100; attempt++) {
    try { if ((await fetch(origin)).ok) break; } catch {}
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  const args = ['--enable-unsafe-webgpu'];
  if (process.platform === 'linux') args.push(
    '--enable-features=Vulkan', '--use-angle=vulkan', '--use-vulkan=swiftshader',
    '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface',
  );
  browser = await chromium.launch({ channel: 'chromium', args });
});
after(async () => { await browser?.close(); server?.kill(); });

async function withoutGPU(page) {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
}

test('unsupported GPU shows recovery instructions and retry', async () => {
  const page = await browser.newPage();
  try {
    await withoutGPU(page);
    await page.goto(origin);
    await page.waitForFunction(() => document.querySelector('#status').dataset.error === 'true');
    assert.match(await page.locator('#status').innerText(), /WebGPU is unavailable/);
    assert.equal(await page.getByRole('button', { name: 'Retry', exact: true }).isVisible(), true);
  } finally { await page.close(); }
});

test('adapter failure is caught and displayed', async () => {
  const page = await browser.newPage();
  try {
    await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: {
      requestAdapter: async () => { throw new Error('Adapter test failure'); },
    } }));
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(origin);
    await page.waitForFunction(() => document.querySelector('#status').dataset.error === 'true');
    assert.match(await page.locator('#status').innerText(), /Adapter test failure/);
    assert.deepEqual(errors, []);
  } finally { await page.close(); }
});

test('controls fit small screens and retain slice settings when resetting orientation', async () => {
  const page = await browser.newPage({ viewport: { width: 360, height: 780 } });
  try {
    await withoutGPU(page);
    await page.goto(origin);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    const axisLabels = page.locator('#axisGuide text');
    const mappings = [
      ['Re(z₀)', 'Im(z₀)', 'Re(c)'], ['Re(c)', 'Im(c)', 'Im(z₀)'],
      ['Re(z₀)', 'Re(c)', 'Im(c)'], ['Re(z₀)', 'Im(z₀)', 'Im(c)'],
    ];
    for (let mode = 0; mode < mappings.length; mode++) {
      await page.getByLabel('Slider dimension').selectOption(String(mode));
      assert.deepEqual(await axisLabels.allTextContents(), mappings[mode]);
      const arrowBefore = await page.locator('#axisXArrow').getAttribute('d');
      await page.locator('#gfx').press('ArrowRight');
      assert.notEqual(await page.locator('#axisXArrow').getAttribute('d'), arrowBefore);
      assert.deepEqual(await axisLabels.allTextContents(), mappings[mode]);
    }
    await page.getByLabel('Slider dimension').selectOption('2');
    assert.equal(await page.locator('#slider').getAttribute('aria-label'), 'Im(z₀)');
    await page.locator('#slider').press('ArrowRight');
    await page.getByLabel('Clip plane', { exact: true }).press('End');
    assert.equal(await page.locator('#sliderValue').innerText(), '0.001');
    assert.equal(await page.locator('#clipValue').innerText(), '3.00');
    await page.getByRole('button', { name: 'Reset view' }).click();
    assert.equal(await page.locator('#sliderValue').innerText(), '0.001');
    assert.equal(await page.locator('#clipValue').innerText(), '3.00');
    assert.equal(await page.getByLabel('Slider dimension').inputValue(), '2');
    assert.equal(await page.locator('#slider').getAttribute('aria-label'), 'Im(z₀)');
    assert.deepEqual(await axisLabels.allTextContents(), mappings[2]);
  } finally { await page.close(); }
});

test('slider thumbs snap around zero without trapping arrow-key adjustments', async () => {
  const page = await browser.newPage();
  try {
    await withoutGPU(page);
    await page.goto(origin);
    for (const [id, output, step, zero] of [
      ['slider', 'sliderValue', '0.001', '0.000'],
      ['clipSlider', 'clipValue', '0.01', '0.00'],
    ]) {
      const control = page.locator(`#${id}`);
      const box = await control.boundingBox();
      for (const fraction of [0.49, 0.51]) {
        await control.press('End');
        await page.mouse.click(box.x + box.width * fraction, box.y + box.height / 2);
        assert.equal(await control.inputValue(), '0');
        assert.equal(await page.locator(`#${output}`).innerText(), zero);
        await control.press('ArrowRight');
        assert.equal(await control.inputValue(), step);
      }
    }
  } finally { await page.close(); }
});

test('Fly controls fit on mobile and return to Orbit on refresh', async () => {
  const page = await browser.newPage({ viewport: { width: 360, height: 780 } });
  try {
    await withoutGPU(page);
    await page.goto(origin);
    await page.getByRole('button', { name: 'Fly', exact: true }).click();
    assert.equal(await page.locator('#flyButton').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('#flyReticle').isVisible(), true);
    assert.equal(await page.locator('#bubbleSlider').isEnabled(), false);
    await page.getByRole('button', { name: 'Cutaway', exact: true }).click();
    assert.equal(await page.locator('#bubbleSlider').isEnabled(), true);
    const canvasBox = await page.locator('#gfx').boundingBox();
    const reticleBox = await page.locator('#flyReticle').boundingBox();
    assert.ok(Math.abs(reticleBox.x + reticleBox.width / 2 - canvasBox.x - canvasBox.width / 2) < 0.5);
    assert.ok(Math.abs(reticleBox.y + reticleBox.height / 2 - canvasBox.y - canvasBox.height / 2) < 0.5);
    await page.locator('#bubbleSlider').press('Home');
    assert.equal(await page.locator('#bubbleValue').innerText(), '0.0001');
    await page.locator('#bubbleSlider').press('ArrowRight');
    assert.equal(await page.locator('#bubbleValue').innerText(), '0.000102');
    assert.equal(await page.locator('#bubbleSlider').getAttribute('aria-valuetext'), '0.000102');
    await page.locator('#bubbleSlider').press('End');
    assert.equal(await page.locator('#bubbleValue').innerText(), '2');
    await page.locator('#gfx').focus();
    assert.match(await page.locator('#help').innerText(), /Scroll or W\/S/);
    assert.equal(await page.evaluate(() => document.activeElement.id), 'gfx');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.locator('#gfx').press('w');
    await page.getByRole('button', { name: 'Reset view', exact: true }).click();
    assert.equal(await page.locator('#flyButton').getAttribute('aria-pressed'), 'true');
    await page.getByRole('button', { name: 'Orbit', exact: true }).click();
    assert.equal(await page.locator('#flyReticle').isVisible(), false);
    assert.match(await page.locator('#help').innerText(), /Drag to orbit/);
    assert.equal(await page.locator('#flyControls').isVisible(), false);
    await page.getByRole('button', { name: 'Fly', exact: true }).click();
    await page.reload();
    assert.equal(await page.locator('#orbitButton').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('#cutawayButton').getAttribute('aria-pressed'), 'false');
    assert.equal(await page.locator('#bubbleSlider').inputValue(), '0');
    assert.equal(await page.locator('#flyReticle').isVisible(), false);
  } finally { await page.close(); }
});

test('refresh resets controls before the next slider input', async () => {
  const page = await browser.newPage();
  try {
    await withoutGPU(page);
    await page.goto(origin);
    await page.getByLabel('Slider dimension').selectOption('2');
    await page.locator('#slider').press('End');
    await page.getByLabel('Clip plane', { exact: true }).press('End');
    await page.reload();
    assert.equal(await page.getByLabel('Slider dimension').inputValue(), '1');
    assert.equal(await page.locator('#slider').inputValue(), '0');
    assert.equal(await page.locator('#clipSlider').inputValue(), '0');
    assert.equal(await page.locator('#sliderValue').innerText(), '0.000');
    assert.equal(await page.locator('#clipValue').innerText(), '0.00');
    await page.locator('#slider').press('ArrowRight');
    assert.equal(await page.locator('#sliderValue').innerText(), '0.001');
  } finally { await page.close(); }
});

test('legacy entry point preserves URL suffix and opens the canonical explorer', async () => {
  const page = await browser.newPage();
  try {
    await withoutGPU(page);
    await page.goto(origin + '/four_d_fractal_webgpu.html?example=1#view');
    await page.waitForURL('**/index.html?example=1#view');
    assert.equal(await page.title(), 'Nebula · 4D Fractal Explorer');
  } finally { await page.close(); }
});

test('real GPU compiles the shader and responds to controls', async (t) => {
  const page = await browser.newPage({ viewport: { width: 640, height: 480 } });
  page.on('console', (message) => { if (message.type() === 'error') console.error(message.text()); });
  try {
    // A listed adapter can still fail in a headless environment. Probe a tiny
    // independent shader before deciding whether renderer validation can run.
    await page.route('**/gpu-probe', (route) => route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>GPU probe</title><canvas width="16" height="16"></canvas>' }));
    await page.goto(origin + '/gpu-probe');
    const probe = await page.evaluate(async () => {
      try {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return 'No usable WebGPU adapter';
        const device = await adapter.requestDevice();
        const unavailable = device.lost.then((info) => 'GPU probe lost device: ' + info.message);
        const work = (async () => {
          const context = document.querySelector('canvas').getContext('webgpu');
          const format = navigator.gpu.getPreferredCanvasFormat();
          context.configure({ device, format });
          const module = device.createShaderModule({ code: `
            @vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
              var positions = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
              return vec4f(positions[i], 0, 1);
            }
            @fragment fn fragment() -> @location(0) vec4f { return vec4f(1, 0, 0, 1); }
          ` });
          const pipeline = await device.createRenderPipelineAsync({ layout: 'auto', vertex: { module, entryPoint: 'vertex' }, fragment: { module, entryPoint: 'fragment', targets: [{ format }] } });
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
          pass.setPipeline(pipeline);
          pass.draw(3);
          pass.end();
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
          return null;
        })();
        const result = await Promise.race([work, unavailable]);
        device.destroy();
        return result;
      } catch (error) { return 'GPU probe failed: ' + error.message; }
    });
    if (probe) {
      if (process.env.NEBULA_TEST_GPU === '1') assert.fail(probe);
      t.skip(probe);
      return;
    }
    const source = await readFile(new URL('../explorer.js', import.meta.url), 'utf8');
    const shader = source.split('const shaderCode = /* wgsl */`')[1].split('`;')[0];
    const adaptive = await page.evaluate(probeAdaptiveDetail, shader);
    assert.deepEqual(adaptive[0], adaptive[1], 'small and large bubbles cost the same for a distant surface');
    assert.equal(adaptive[0][1], 80, 'distant surfaces retain the base iteration budget');
    assert.ok(adaptive[2][0] < adaptive[0][0] && adaptive[2][1] > adaptive[0][1]);
    assert.ok(adaptive[3][0] < adaptive[2][0] && adaptive[3][1] > adaptive[2][1]);
    assert.ok(adaptive[4][0] < adaptive[2][0], 'more canvas pixels resolve finer features');
    assert.equal(adaptive[2][2], 0, 'the coarse field fills a late-escaping point');
    assert.ok(adaptive[3][3] > adaptive[3][0], 'refinement opens that point at the visible scale');
    for (const row of adaptive) {
      assert.ok(row.every(Number.isFinite));
      assert.equal(row[4], 0, 'true solid remains solid');
      assert.equal(row[5], row[6], 'early escape stays unchanged');
      assert.ok(row[7] <= 512, 'iteration work is bounded even at zero distance');
    }
    const cutSamples = await page.evaluate(probeCutaway, shader);
    const expectedCuts = [
      [0.5, 0.25, 0, 0, 0.25, 0.5, 1], [0.25, 0, 0, 0, 0, 0.25, 1],
      [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
      [0.5, 0.25, 0, 0, 0.35, 0.5, 1], [0.001, 0, 0, 0, 0, 0.001, 1], [0.0001, 0, 0, 0, 0, 0.0001, 1],
    ];
    cutSamples.forEach((sample, i) => sample.forEach((value, j) => {
      assert.ok(Math.abs(value - expectedCuts[i][j]) < 1e-5, `bubble field case ${i}, sample ${j}`);
    }));
    const samples = await page.evaluate(probeDistances, shader);
    for (const sample of samples) {
      for (let mode = 0; mode < 4; mode++) {
        assert.ok(Math.abs(sample[mode] - referenceDistance(mode)) < 1e-4, `distance derivative for mode ${mode}`);
        assert.equal(sample[mode + 4], sample[mode], 'minimum clipping leaves the distance field unchanged');
      }
    }
    await page.goto(origin);
    await page.waitForFunction(() => document.querySelector('#status').dataset.error === 'true' || document.querySelector('#status').hidden);
    const status = await page.locator('#status').innerText();
    if (/unavailable|No WebGPU adapter/.test(status) && process.env.NEBULA_TEST_GPU !== '1') {
      t.skip(status);
      return;
    }
    assert.equal(await page.locator('#status').isVisible(), false);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.getByLabel('Slider dimension').selectOption('1');
    await page.getByLabel('Clip plane', { exact: true }).press('Home');
    for (let i = 0; i < 5; i++) await page.getByLabel('Clip plane', { exact: true }).press('PageUp');
    await page.getByRole('button', { name: 'Align to view', exact: true }).click();
    await page.locator('#gfx').focus();
    for (let i = 0; i < 32; i++) await page.keyboard.press('ArrowRight');
    await page.keyboard.press('+');
    await page.keyboard.press('ArrowLeft');
    await page.getByRole('button', { name: 'Fly', exact: true }).click();
    await page.getByRole('button', { name: 'Cutaway', exact: true }).click();
    for (let i = 0; i < 50; i++) await page.keyboard.press('w');
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('a');
    await page.keyboard.press('e');
    await page.getByRole('button', { name: 'Reset view', exact: true }).click();
    await page.getByRole('button', { name: 'Orbit', exact: true }).click();
    await page.waitForTimeout(200);
    assert.equal(await page.locator('#status').getAttribute('data-error'), null);
    assert.deepEqual(errors, []);
    assert.equal(await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      return canvas.width > 0 && canvas.height > 0 && canvas.width * canvas.height <= 2_000_000;
    }), true);
  } finally { await page.close(); }
});
