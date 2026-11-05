// Independent finite-difference oracle: differentiate the iterated orbit
// numerically instead of repeating the shader's derivative recurrence.
export function referenceDistance(mode) {
  const point = [1.7, 0.2, -0.6];
  function orbit(p, steps = 80) {
    const [x, y, z] = p, slice = 0.35;
    let [a, b, c, d] = [
      [x, y, z, slice], [slice, z, x, y],
      [x, slice, y, z], [x, y, slice, z],
    ][mode];
    for (let i = 0; i < steps; i++) {
      [a, b] = [a * a - b * b + c, 2 * a * b + d];
      if (steps === 80 && a * a + b * b > 64) return { value: [a, b], steps: i + 1 };
    }
    return { value: [a, b], steps };
  }
  const center = orbit(point);
  let squaredNorm = 0;
  for (let axis = 0; axis < 3; axis++) {
    const plus = [...point], minus = [...point], h = 1e-6;
    plus[axis] += h;
    minus[axis] -= h;
    const a = orbit(plus, center.steps).value, b = orbit(minus, center.steps).value;
    squaredNorm += ((a[0] - b[0]) / (2 * h)) ** 2 + ((a[1] - b[1]) / (2 * h)) ** 2;
  }
  const radius = Math.hypot(...center.value);
  return Math.min(1, 0.5 * Math.log(radius) * radius / Math.sqrt(squaredNorm));
}

// Run the actual renderer WGSL as a compute probe. Additional outputs check
// that minimum clipping leaves the field unchanged at different orientations.
export async function probeDistances(shader) {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const uniform = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const module = device.createShaderModule({ code: shader + `
      @group(0) @binding(1) var<storage, read_write> results: array<f32>;
      @compute @workgroup_size(1) fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
        let p = vec3<f32>(1.7, 0.2, -0.6);
        results[id.x] = fractal_distance(p, i32(id.x), 0.35);
        results[id.x + 4u] = clipped_distance(p, i32(id.x), 0.35);
      }
    ` });
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'probe' } });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: output } },
    ] });
    const samples = [];
    for (const [pitch, yaw] of [[0.3, 0], [-0.6, 1.6], [1.2, 3.1]]) {
      const data = new Float32Array(20);
      data[7] = -3;
      data[10] = pitch;
      data[11] = yaw;
      device.queue.writeBuffer(uniform, 0, data);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(4);
      pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, 32);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      samples.push(Array.from(new Float32Array(readback.getMappedRange())));
      readback.unmap();
    }
    return samples;
  } finally {
    uniform.destroy(); output.destroy(); readback.destroy(); device.destroy();
  }
}

// Sample the actual cut field inside solid material, including translated
// camera centers, disabled states, and the intersection with the plane.
export async function probeCutaway(shader) {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const uniform = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: 24, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 24, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const module = device.createShaderModule({ code: shader + `
      @group(0) @binding(1) var<storage, read_write> results: array<f32>;
      @compute @workgroup_size(1) fn probe() {
        let center = uniforms.data4.xyz;
        results[0] = clipped_distance(center, 1, 0.0);
        results[1] = clipped_distance(center + vec3f(0, 0, -0.25), 1, 0.0);
        results[2] = clipped_distance(center + vec3f(0, 0, -0.5), 1, 0.0);
        results[3] = clipped_distance(center + vec3f(0, 0, -0.75), 1, 0.0);
        results[4] = clipped_distance(center + vec3f(0, 0, 0.25), 1, 0.0);
        results[5] = getBubbleRadius();
      }
    ` });
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'probe' } });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: output } },
    ] });
    const samples = [];
    for (const [flying, radius, clip, center] of [
      [1, 0.5, -3, [0, 0, 0]], [1, 0.25, -3, [0, 0, 0]],
      [0, 0.5, -3, [0, 0, 0]], [1, 0, -3, [0, 0, 0]],
      [1, 0.5, 0, [0.1, 0, 0.1]],
    ]) {
      const data = new Float32Array(20);
      data[7] = clip;
      data.set(center, 12);
      data[15] = flying;
      data[16] = radius;
      device.queue.writeBuffer(uniform, 0, data);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, 24);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      samples.push(Array.from(new Float32Array(readback.getMappedRange())));
      readback.unmap();
    }
    return samples;
  } finally {
    uniform.destroy(); output.destroy(); readback.destroy(); device.destroy();
  }
}
