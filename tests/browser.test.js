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
  browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-angle=swiftshader'] });
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
    await page.getByLabel('Slice mode').selectOption('2');
    assert.equal(await page.locator('#sliderLabel').innerText(), 'Im(z0)');
    await page.locator('#slider').press('ArrowRight');
    await page.getByLabel('Clip plane', { exact: true }).press('End');
    assert.equal(await page.locator('#sliderValue').innerText(), '0.001');
    assert.equal(await page.locator('#clipValue').innerText(), '3.00');
    await page.getByRole('button', { name: 'Reset view' }).click();
    assert.equal(await page.locator('#sliderValue').innerText(), '0.000');
    assert.equal(await page.locator('#clipValue').innerText(), '-3.00');
    assert.equal(await page.getByLabel('Slice mode').inputValue(), '0');
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
  try {
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
    await page.getByLabel('Slice mode').selectOption('1');
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
