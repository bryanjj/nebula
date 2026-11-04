import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
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

test('controls fit small screens and reset after mode, slice and clip changes', async () => {
  const page = await browser.newPage({ viewport: { width: 360, height: 780 } });
  try {
    await withoutGPU(page);
    await page.goto(origin);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.getByLabel('Slider dimension').selectOption('2');
    assert.equal(await page.locator('#slider').getAttribute('aria-label'), 'Im(z₀)');
    await page.locator('#slider').press('ArrowRight');
    await page.getByLabel('Clip plane', { exact: true }).press('End');
    assert.equal(await page.locator('#sliderValue').innerText(), '0.001');
    assert.equal(await page.locator('#clipValue').innerText(), '3.00');
    await page.getByRole('button', { name: 'Reset view' }).click();
    assert.equal(await page.locator('#sliderValue').innerText(), '0.000');
    assert.equal(await page.locator('#clipValue').innerText(), '-3.00');
    assert.equal(await page.getByLabel('Slider dimension').inputValue(), '0');
    assert.equal(await page.locator('#slider').getAttribute('aria-label'), 'Im(c)');
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
    await page.goto(origin);
    await page.waitForFunction(() => document.querySelector('#status').dataset.error === 'true' || document.querySelector('#status').textContent.includes('ready'));
    const status = await page.locator('#status').innerText();
    if (/unavailable|No WebGPU adapter/.test(status) && process.env.NEBULA_TEST_GPU !== '1') {
      t.skip(status);
      return;
    }
    assert.match(status, /WebGPU ready/);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.getByLabel('Slider dimension').selectOption('1');
    await page.getByLabel('Clip plane', { exact: true }).press('ArrowRight');
    await page.locator('#gfx').focus();
    await page.keyboard.press('+');
    await page.keyboard.press('ArrowLeft');
    await page.getByLabel('Resolution', { exact: true }).selectOption('0.5');
    await page.waitForTimeout(200);
    assert.equal(await page.locator('#status').getAttribute('data-error'), null);
    assert.deepEqual(errors, []);
    assert.equal(await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      return canvas.width > 0 && canvas.height > 0 && canvas.width * canvas.height <= 2_000_000;
    }), true);
  } finally { await page.close(); }
});
