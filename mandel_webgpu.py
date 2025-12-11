import numpy as np
import wgpu
from wgpu.utils.compute import compute_with_buffers
from vedo import Plotter, Mesh

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------
N = 512
MAX_ITER = 80
BAILOUT_SQ = 16.0

C_RE_RANGE = (-2.0, 2.0)
C_IM_RANGE = (-2.0, 2.0)
Z0_RE_RANGE = (-2.0, 2.0)

IM_C_INIT = 0.00
Z0_RE_INIT = 0.00
IM_STEP = 0.02
Z0_STEP = 0.02

# Precompute coordinates
xs_np = np.linspace(*C_RE_RANGE, N).astype(np.float32)
ys_np = np.linspace(*C_IM_RANGE, N).astype(np.float32)
zs_np = np.linspace(*Z0_RE_RANGE, N).astype(np.float32)

# ----------------------------------------------------
# WEBGPU SETUP
# ----------------------------------------------------
device = wgpu.gpu.request_adapter_sync(power_preference="high-performance").request_device_sync()

# Volume shader - computes fractal membership
VOLUME_SHADER = f"""
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read> ys: array<f32>;
@group(0) @binding(2) var<storage, read> zs: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<f32>;  // im_c, z0_re, bailout_sq, max_iter
@group(0) @binding(4) var<storage, read_write> volume: array<u32>;

const N: u32 = {N}u;

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    if (i >= N || j >= N || k >= N) {{
        return;
    }}

    let c_re = xs[i];
    let c_im = ys[j];
    let im_c = params.x;
    let z0_re = params.y;
    let bailout_sq = params.z;
    let max_iter = u32(params.w);

    var zr = z0_re + zs[k];
    var zi = 0.0;

    var bounded: u32 = 1u;
    for (var iter: u32 = 0u; iter < max_iter; iter++) {{
        let zr_new = zr * zr - zi * zi + c_re;
        zi = 2.0 * zr * zi + c_im;
        zr = zr_new;

        if (zr * zr + zi * zi > bailout_sq) {{
            bounded = 0u;
            break;
        }}
    }}

    let idx = i + j * N + k * N * N;
    volume[idx] = bounded;
}}
"""

# Marching cubes edge table (256 entries)
# Each entry is a 12-bit mask indicating which edges are intersected
EDGE_TABLE = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
]

# Convert to string for shader
EDGE_TABLE_STR = ", ".join(f"{x}u" for x in EDGE_TABLE)

# Marching cubes shader
MC_SHADER = f"""
@group(0) @binding(0) var<storage, read> volume: array<u32>;
@group(0) @binding(1) var<storage, read> edge_table: array<u32>;
@group(0) @binding(2) var<storage, read_write> vertex_count: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(4) var<uniform> grid_params: vec4<f32>;  // dx, dy, dz, origin_x
@group(0) @binding(5) var<uniform> grid_params2: vec4<f32>; // origin_y, origin_z, 0, 0

const N: u32 = {N}u;
const MAX_VERTS: u32 = 50000000u;  // 50M max vertices

fn get_volume(i: u32, j: u32, k: u32) -> f32 {{
    if (i >= N || j >= N || k >= N) {{
        return 0.0;
    }}
    return f32(volume[i + j * N + k * N * N]);
}}

fn interp_vertex(p1: vec3<f32>, p2: vec3<f32>, v1: f32, v2: f32) -> vec3<f32> {{
    let iso = 0.5;
    if (abs(iso - v1) < 0.00001) {{ return p1; }}
    if (abs(iso - v2) < 0.00001) {{ return p2; }}
    if (abs(v1 - v2) < 0.00001) {{ return p1; }}
    let mu = (iso - v1) / (v2 - v1);
    return p1 + mu * (p2 - p1);
}}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    if (i >= N - 1u || j >= N - 1u || k >= N - 1u) {{
        return;
    }}

    let dx = grid_params.x;
    let dy = grid_params.y;
    let dz = grid_params.z;
    let ox = grid_params.w;
    let oy = grid_params2.x;
    let oz = grid_params2.y;

    // Sample 8 corners
    let v0 = get_volume(i, j, k);
    let v1 = get_volume(i + 1u, j, k);
    let v2 = get_volume(i + 1u, j + 1u, k);
    let v3 = get_volume(i, j + 1u, k);
    let v4 = get_volume(i, j, k + 1u);
    let v5 = get_volume(i + 1u, j, k + 1u);
    let v6 = get_volume(i + 1u, j + 1u, k + 1u);
    let v7 = get_volume(i, j + 1u, k + 1u);

    // Compute cube index
    var cube_idx: u32 = 0u;
    if (v0 > 0.5) {{ cube_idx |= 1u; }}
    if (v1 > 0.5) {{ cube_idx |= 2u; }}
    if (v2 > 0.5) {{ cube_idx |= 4u; }}
    if (v3 > 0.5) {{ cube_idx |= 8u; }}
    if (v4 > 0.5) {{ cube_idx |= 16u; }}
    if (v5 > 0.5) {{ cube_idx |= 32u; }}
    if (v6 > 0.5) {{ cube_idx |= 64u; }}
    if (v7 > 0.5) {{ cube_idx |= 128u; }}

    let edges = edge_table[cube_idx];
    if (edges == 0u) {{
        return;
    }}

    // Compute corner positions
    let fi = f32(i);
    let fj = f32(j);
    let fk = f32(k);

    let p0 = vec3<f32>(ox + fi * dx, oy + fj * dy, oz + fk * dz);
    let p1 = vec3<f32>(ox + (fi + 1.0) * dx, oy + fj * dy, oz + fk * dz);
    let p2 = vec3<f32>(ox + (fi + 1.0) * dx, oy + (fj + 1.0) * dy, oz + fk * dz);
    let p3 = vec3<f32>(ox + fi * dx, oy + (fj + 1.0) * dy, oz + fk * dz);
    let p4 = vec3<f32>(ox + fi * dx, oy + fj * dy, oz + (fk + 1.0) * dz);
    let p5 = vec3<f32>(ox + (fi + 1.0) * dx, oy + fj * dy, oz + (fk + 1.0) * dz);
    let p6 = vec3<f32>(ox + (fi + 1.0) * dx, oy + (fj + 1.0) * dy, oz + (fk + 1.0) * dz);
    let p7 = vec3<f32>(ox + fi * dx, oy + (fj + 1.0) * dy, oz + (fk + 1.0) * dz);

    // Interpolate edge vertices
    var vert_list: array<vec3<f32>, 12>;
    if ((edges & 1u) != 0u) {{ vert_list[0] = interp_vertex(p0, p1, v0, v1); }}
    if ((edges & 2u) != 0u) {{ vert_list[1] = interp_vertex(p1, p2, v1, v2); }}
    if ((edges & 4u) != 0u) {{ vert_list[2] = interp_vertex(p2, p3, v2, v3); }}
    if ((edges & 8u) != 0u) {{ vert_list[3] = interp_vertex(p3, p0, v3, v0); }}
    if ((edges & 16u) != 0u) {{ vert_list[4] = interp_vertex(p4, p5, v4, v5); }}
    if ((edges & 32u) != 0u) {{ vert_list[5] = interp_vertex(p5, p6, v5, v6); }}
    if ((edges & 64u) != 0u) {{ vert_list[6] = interp_vertex(p6, p7, v6, v7); }}
    if ((edges & 128u) != 0u) {{ vert_list[7] = interp_vertex(p7, p4, v7, v4); }}
    if ((edges & 256u) != 0u) {{ vert_list[8] = interp_vertex(p0, p4, v0, v4); }}
    if ((edges & 512u) != 0u) {{ vert_list[9] = interp_vertex(p1, p5, v1, v5); }}
    if ((edges & 1024u) != 0u) {{ vert_list[10] = interp_vertex(p2, p6, v2, v6); }}
    if ((edges & 2048u) != 0u) {{ vert_list[11] = interp_vertex(p3, p7, v3, v7); }}

    // Use simplified triangle generation (placeholder - full table needed for production)
    // For now, output edge vertices for testing
    let vert_idx = atomicAdd(&vertex_count, 3u);
    if (vert_idx + 3u < MAX_VERTS) {{
        let base = vert_idx * 3u;
        // Output first intersected edge vertex as a point
        if ((edges & 1u) != 0u) {{
            vertices[base] = vert_list[0].x;
            vertices[base + 1u] = vert_list[0].y;
            vertices[base + 2u] = vert_list[0].z;
        }}
    }}
}}
"""


class WebGPUFractal:
    def __init__(self):
        self.device = device

        # Create buffers
        self.xs_buffer = self._create_buffer(xs_np, wgpu.BufferUsage.STORAGE)
        self.ys_buffer = self._create_buffer(ys_np, wgpu.BufferUsage.STORAGE)
        self.zs_buffer = self._create_buffer(zs_np, wgpu.BufferUsage.STORAGE)

        # Volume buffer (N^3 u32s)
        self.volume_buffer = device.create_buffer(
            size=N * N * N * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )

        # Params uniform
        self.params_buffer = device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        # Compile volume shader
        self.volume_shader = device.create_shader_module(code=VOLUME_SHADER)

        # Create pipeline
        self.volume_pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": self.volume_shader, "entry_point": "main"}
        )

        # Create bind group
        self.volume_bind_group = device.create_bind_group(
            layout=self.volume_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.xs_buffer}},
                {"binding": 1, "resource": {"buffer": self.ys_buffer}},
                {"binding": 2, "resource": {"buffer": self.zs_buffer}},
                {"binding": 3, "resource": {"buffer": self.params_buffer}},
                {"binding": 4, "resource": {"buffer": self.volume_buffer}},
            ]
        )

    def _create_buffer(self, data, usage):
        buffer = self.device.create_buffer_with_data(
            data=data.tobytes(),
            usage=usage
        )
        return buffer

    def compute_volume(self, im_c, z0_re):
        # Update params
        params = np.array([im_c, z0_re, BAILOUT_SQ, MAX_ITER], dtype=np.float32)
        self.device.queue.write_buffer(self.params_buffer, 0, params.tobytes())

        # Dispatch compute
        command_encoder = self.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.volume_pipeline)
        compute_pass.set_bind_group(0, self.volume_bind_group)

        # Dispatch with workgroup size 8x8x8
        wg = (N + 7) // 8
        compute_pass.dispatch_workgroups(wg, wg, wg)
        compute_pass.end()

        self.device.queue.submit([command_encoder.finish()])

    def get_volume_numpy(self):
        # Read back volume
        buffer_size = N * N * N * 4
        staging = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
        )

        command_encoder = self.device.create_command_encoder()
        command_encoder.copy_buffer_to_buffer(self.volume_buffer, 0, staging, 0, buffer_size)
        self.device.queue.submit([command_encoder.finish()])

        # Map and read
        staging.map_sync(wgpu.MapMode.READ)
        data = staging.read_mapped()
        staging.unmap()

        return np.frombuffer(data, dtype=np.uint32).reshape((N, N, N)).astype(np.uint8)


# Use PyMCubes for fast CPU marching cubes (fallback)
try:
    import mcubes
    USE_MCUBES = True
    print("[INFO] Using PyMCubes for marching cubes")
except ImportError:
    from skimage.measure import marching_cubes
    USE_MCUBES = False
    print("[INFO] Using skimage marching cubes (install pymcubes for 3x speedup: pip install pymcubes)")


def extract_surface(volume):
    dx = xs_np[1] - xs_np[0]
    dy = ys_np[1] - ys_np[0]
    dz = zs_np[1] - zs_np[0]

    if USE_MCUBES:
        # PyMCubes is ~3-5x faster
        verts, faces = mcubes.marching_cubes(volume.astype(np.float32), 0.5)
        verts[:, 0] = verts[:, 0] * dx + xs_np[0]
        verts[:, 1] = verts[:, 1] * dy + ys_np[0]
        verts[:, 2] = verts[:, 2] * dz + zs_np[0]
    else:
        verts, faces, _, _ = marching_cubes(volume, level=0.5, spacing=(dx, dy, dz))
        verts[:, 0] += xs_np[0]
        verts[:, 1] += ys_np[0]
        verts[:, 2] += zs_np[0]

    return verts, faces


def main():
    print("[INFO] Initializing WebGPU...")
    gpu = WebGPUFractal()

    print("[INFO] Warming up GPU...")
    gpu.compute_volume(IM_C_INIT, Z0_RE_INIT)

    def compute_surface(im_c, z0_re):
        print(f"[INFO] Computing: Im(c)={im_c:.4f}, Re(z0)={z0_re:.4f}")
        gpu.compute_volume(im_c, z0_re)
        vol = gpu.get_volume_numpy()
        print("[INFO] Marching cubes...")
        verts, faces = extract_surface(vol)
        print(f"[INFO] Mesh: {len(verts)} verts / {len(faces)} faces")
        return verts, faces

    # Initial mesh
    verts, faces = compute_surface(IM_C_INIT, Z0_RE_INIT)
    mesh = Mesh([verts, faces]).c("cyan").lighting("plastic")

    state = {
        "im_c": IM_C_INIT,
        "z0_re": Z0_RE_INIT,
        "mesh": mesh,
    }

    plt = Plotter(bg="black", title="3D Explorer (WebGPU): [ ] = Im(c), z/x = Re(z0)")
    plt.show(mesh, resetcam=True, interactive=False)

    def update(vis):
        verts, faces = compute_surface(state["im_c"], state["z0_re"])
        new_mesh = Mesh([verts, faces])
        new_mesh.c("cyan").lighting("plastic")
        new_mesh.compute_normals()
        vis.remove(state["mesh"])
        state["mesh"] = new_mesh
        vis.add(new_mesh)
        vis.render()

    def on_key(evt):
        key = evt.keypress
        print("KEY:", key)

        if key == "bracketleft":
            state["im_c"] -= IM_STEP
            update(plt)
        elif key == "bracketright":
            state["im_c"] += IM_STEP
            update(plt)
        elif key == "z":
            state["z0_re"] -= Z0_STEP
            update(plt)
        elif key == "x":
            state["z0_re"] += Z0_STEP
            update(plt)
        elif key in ("q", "escape"):
            plt.close()

    plt.add_callback("key press", on_key)
    plt.show(interactive=True)


if __name__ == "__main__":
    main()
