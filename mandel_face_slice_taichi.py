import numpy as np
import taichi as ti
from vedo import Plotter, Mesh

# Initialize Taichi with Metal backend for M1
ti.init(arch=ti.metal)

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------
N = 1024
MAX_ITER = 256
BAILOUT = 4.0
BAILOUT_SQ = BAILOUT * BAILOUT

# Domain for the 3-D grid
C_RE_RANGE = (-2.0, 2.0)
C_IM_RANGE = (-2.0, 2.0)
Z0_RE_RANGE = (-2.0, 2.0)

# PARAMETER SLIDERS
IM_C_INIT = 0.00
Z0_RE_INIT = 0.00
IM_STEP = 0.02
Z0_STEP = 0.02

# ----------------------------------------------------
# TAICHI FIELDS
# ----------------------------------------------------
volume = ti.field(dtype=ti.u8, shape=(N, N, N))

# Precompute coordinate arrays
xs_np = np.linspace(*C_RE_RANGE, N).astype(np.float32)
ys_np = np.linspace(*C_IM_RANGE, N).astype(np.float32)
zs_np = np.linspace(*Z0_RE_RANGE, N).astype(np.float32)

xs = ti.field(dtype=ti.f32, shape=N)
ys = ti.field(dtype=ti.f32, shape=N)
zs = ti.field(dtype=ti.f32, shape=N)

xs.from_numpy(xs_np)
ys.from_numpy(ys_np)
zs.from_numpy(zs_np)


# ----------------------------------------------------
# TAICHI KERNEL - GPU ACCELERATED
# ----------------------------------------------------
@ti.kernel
def compute_volume_gpu(im_c: ti.f32, z0_re: ti.f32):
    for i, j, k in volume:
        c_re = xs[i]
        c_im = ys[j]

        zr = z0_re + zs[k]
        zi = 0.0

        bounded = ti.u8(1)
        for _ in range(MAX_ITER):
            zr_new = zr * zr - zi * zi + c_re
            zi = 2.0 * zr * zi + c_im
            zr = zr_new

            if zr * zr + zi * zi > BAILOUT_SQ:
                bounded = ti.u8(0)
                break

        volume[i, j, k] = bounded


# ----------------------------------------------------
# MARCHING CUBES (using skimage for now, Taichi MC is experimental)
# ----------------------------------------------------
from skimage.measure import marching_cubes

def extract_surface():
    vol_np = volume.to_numpy()

    dx = xs_np[1] - xs_np[0]
    dy = ys_np[1] - ys_np[0]
    dz = zs_np[1] - zs_np[0]

    verts, faces, normals, values = marching_cubes(
        vol_np,
        level=0.5,
        spacing=(dx, dy, dz)
    )

    # Shift to coordinate system
    verts[:, 0] += xs_np[0]
    verts[:, 1] += ys_np[0]
    verts[:, 2] += zs_np[0]

    return verts, faces


def compute_surface(im_c, z0_re):
    print(f"[INFO] Computing volume: Im(c)={im_c:.4f}, Re(z0)={z0_re:.4f}")
    compute_volume_gpu(im_c, z0_re)
    ti.sync()  # Wait for GPU to finish

    print("[INFO] Marching cubes...")
    verts, faces = extract_surface()
    print(f"[INFO] Mesh: {len(verts)} verts / {len(faces)} faces")
    return verts, faces


# ----------------------------------------------------
# MAIN EXPLORER
# ----------------------------------------------------
def main():
    print("[INFO] Warming up Taichi GPU kernel...")
    compute_volume_gpu(IM_C_INIT, Z0_RE_INIT)
    ti.sync()
    print("[INFO] GPU warm-up complete")

    # Initial mesh
    verts, faces = compute_surface(IM_C_INIT, Z0_RE_INIT)
    mesh = Mesh([verts, faces]).c("cyan").lighting("plastic")

    state = {
        "im_c": IM_C_INIT,
        "z0_re": Z0_RE_INIT,
        "mesh": mesh,
    }

    plt = Plotter(bg="black", title="3D Explorer (Taichi GPU): [ ] = Im(c), z/x = Re(z0)")
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

        if key == "bracketleft":  # [
            state["im_c"] -= IM_STEP
            update(plt)
        elif key == "bracketright":  # ]
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
