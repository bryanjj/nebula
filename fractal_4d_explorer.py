import numpy as np
from numba import njit, prange
from skimage.measure import marching_cubes
from vedo import Plotter, Mesh, Text2D

# ============================================================
# 4D → 3D CONFIG
# ============================================================

# Real dimensions indices
DIM_NAMES = ["Z0_RE", "Z0_IM", "C_RE", "C_IM"]
Z0_RE, Z0_IM, C_RE, C_IM = 0, 1, 2, 3  # for readability

# Ranges for each real dimension
dim_min = np.array([
    0.0,        # Z0_RE: 0..1 so z0=0 face gives Mandelbrot
    -1.0,       # Z0_IM
    -2.0,       # C_RE
    -1.5,       # C_IM
], dtype=np.float64)

dim_max = np.array([
    1.0,        # Z0_RE
    1.0,        # Z0_IM
    1.0,        # C_RE
    1.5,        # C_IM
], dtype=np.float64)

N = 512
MAX_ITER = 80
BAILOUT = 4.0
SLIDER_STEP = 0.02

# Default config A:
# X = C_RE, Y = C_IM, Z = Z0_RE, Slider = Z0_IM
axis_dim = np.array([C_RE, C_IM, Z0_RE], dtype=np.int64)  # [X, Y, Z] as dim indices


def compute_slider_dim(axis_dim):
    """Given the 3 axis dims, return the index of the remaining slider dim."""
    used = set(axis_dim.tolist())
    for d in range(4):
        if d not in used:
            return d
    # Should never happen
    return 0


slider_dim = compute_slider_dim(axis_dim)
slider_val_default = 0.0
slider_val = slider_val_default


# ============================================================
# NUMBA VOLUME COMPUTATION
# ============================================================

@njit(parallel=True, fastmath=True)
def compute_volume(n,
                   axis_dim,         # shape (3,), int64
                   slider_dim,       # int64
                   slider_val,       # float64
                   dim_min, dim_max, # shape (4,), float64
                   max_iter, bailout):

    volume = np.zeros((n, n, n), dtype=np.uint8)
    b2 = bailout * bailout

    # Precompute axis mins and steps for X/Y/Z
    axis_mins = np.empty(3, dtype=np.float64)
    axis_steps = np.empty(3, dtype=np.float64)
    for a in range(3):
        d = axis_dim[a]
        axis_mins[a] = dim_min[d]
        axis_steps[a] = (dim_max[d] - dim_min[d]) / (n - 1.0)

    for ix in prange(n):
        for iy in range(n):
            for iz in range(n):
                # coords_for_dims[d] = value of that real dimension at this voxel
                coords_for_dims = np.empty(4, dtype=np.float64)

                # Start with slider
                coords_for_dims[slider_dim] = slider_val

                # Axis X
                x_val = axis_mins[0] + axis_steps[0] * ix
                coords_for_dims[axis_dim[0]] = x_val

                # Axis Y
                y_val = axis_mins[1] + axis_steps[1] * iy
                coords_for_dims[axis_dim[1]] = y_val

                # Axis Z
                z_val = axis_mins[2] + axis_steps[2] * iz
                coords_for_dims[axis_dim[2]] = z_val

                z0_re = coords_for_dims[Z0_RE]
                z0_im = coords_for_dims[Z0_IM]
                c_re  = coords_for_dims[C_RE]
                c_im  = coords_for_dims[C_IM]

                # Iterate z_{n+1} = z_n^2 + c
                zr = z0_re
                zi = z0_im
                bounded = 1
                for _ in range(max_iter):
                    zr2 = zr * zr - zi * zi
                    zi2 = 2.0 * zr * zi
                    zr = zr2 + c_re
                    zi = zi2 + c_im
                    if zr * zr + zi * zi > b2:
                        bounded = 0
                        break

                volume[ix, iy, iz] = bounded

    return volume


def compute_surface(n, axis_dim, slider_dim, slider_val, dim_min, dim_max):
    print(f"[INFO] Volume: slider {DIM_NAMES[slider_dim]} = {slider_val:.4f}")
    vol = compute_volume(n, axis_dim, slider_dim, slider_val,
                         dim_min, dim_max, MAX_ITER, BAILOUT)

    # Compute axis spacing from associated dims
    axis_mins = np.empty(3, dtype=np.float64)
    axis_steps = np.empty(3, dtype=np.float64)
    for a in range(3):
        d = axis_dim[a]
        axis_mins[a] = dim_min[d]
        axis_steps[a] = (dim_max[d] - dim_min[d]) / (n - 1.0)

    dx, dy, dz = axis_steps[0], axis_steps[1], axis_steps[2]

    print("[INFO] Marching cubes...")
    verts, faces, normals, values = marching_cubes(
        vol,
        level=0.5,
        spacing=(dx, dy, dz)
    )

    # Shift to real coordinate system
    verts[:, 0] += axis_mins[0]
    verts[:, 1] += axis_mins[1]
    verts[:, 2] += axis_mins[2]

    print(f"[INFO] Mesh: {len(verts)} verts, {len(faces)} faces")
    return verts, faces


# ============================================================
# RUNTIME EXPLORER
# ============================================================

def make_hud_text(axis_dim, slider_dim, slider_val):
    s = []
    s.append(f"X axis = {DIM_NAMES[axis_dim[0]]}")
    s.append(f"Y axis = {DIM_NAMES[axis_dim[1]]}")
    s.append(f"Z axis = {DIM_NAMES[axis_dim[2]]}")
    s.append(f"Slider = {DIM_NAMES[slider_dim]} = {slider_val:.4f}")
    return "\n".join(s)


def main():
    global axis_dim, slider_dim, slider_val

    # Warm up Numba
    print("[INFO] Warming Numba...")
    _ = compute_volume(N, axis_dim, slider_dim, slider_val,
                       dim_min, dim_max, MAX_ITER, BAILOUT)

    # Initial surface
    verts, faces = compute_surface(N, axis_dim, slider_dim, slider_val,
                                   dim_min, dim_max)
    mesh = Mesh([verts, faces]).c("cyan").lighting("plastic")

    plt = Plotter(bg="black", title="4D→3D Fractal Explorer")
    hud = Text2D(make_hud_text(axis_dim, slider_dim, slider_val),
             pos="top-left", s=0.8, c="white", font="Courier")
    plt.add(hud)

    plt.show(mesh, hud, resetcam=True, interactive=False)

    state = {
        "mesh": mesh,
        "hud": hud,
    }

    def clamp_slider():
        global slider_val
        mn = dim_min[slider_dim]
        mx = dim_max[slider_dim]
        if slider_val < mn:
            slider_val = mn
        if slider_val > mx:
            slider_val = mx

    def rebuild_mesh():
        verts, faces = compute_surface(N, axis_dim, slider_dim, slider_val,
                                       dim_min, dim_max)
        new_mesh = Mesh([verts, faces]).c("cyan").lighting("plastic")
        plt.remove(state["mesh"])
        state["mesh"] = new_mesh
        plt.add(new_mesh)

    def rebuild_hud():
        plt.remove(state["hud"])
        text = make_hud_text(axis_dim, slider_dim, slider_val)
        state["hud"] = Text2D(text, pos="top-left", s=0.8,
                      c="white", font="Courier")
        plt.add(state["hud"])

    def update():
        clamp_slider()
        rebuild_mesh()
        rebuild_hud()
        plt.render()

    def recalc_slider_dim_and_reset():
        global slider_dim, slider_val
        slider_dim = compute_slider_dim(axis_dim)
        slider_val = slider_val_default
        print(f"[INFO] New axes: X={DIM_NAMES[axis_dim[0]]}, "
              f"Y={DIM_NAMES[axis_dim[1]]}, Z={DIM_NAMES[axis_dim[2]]}")
        print(f"[INFO] Slider = {DIM_NAMES[slider_dim]} (reset to {slider_val_default})")

    def cycle_axis(idx):
        """Cycle axis_dim[idx] through the 4 dims, keeping uniqueness."""
        used = set(axis_dim.tolist())
        current = axis_dim[idx]
        for offset in range(1, 5):  # at most 4 tries
            candidate = (current + offset) % 4
            if candidate not in used or candidate == current:
                axis_dim[idx] = candidate
                break
        # ensure all three are distinct: if not, repair
        while len(set(axis_dim.tolist())) < 3:
            # brute force assign a free dim
            used_now = set(axis_dim.tolist())
            for d in range(4):
                if d not in used_now:
                    # find a duplicate to replace
                    for j in range(3):
                        if list(axis_dim).count(axis_dim[j]) > 1:
                            axis_dim[j] = d
                            break
                    break
        recalc_slider_dim_and_reset()
        update()

    def on_key(evt):
        global slider_val
        key = evt.keypress
        print("KEY:", key)

        # Slider controls
        if key == "bracketleft":      # [
            slider_val -= SLIDER_STEP
            update()

        elif key == "bracketright":   # ]
            slider_val += SLIDER_STEP
            update()

        # Axis cycling
        elif key == "x":
            cycle_axis(0)  # X axis
        elif key == "y":
            cycle_axis(1)  # Y axis
        elif key == "z":
            cycle_axis(2)  # Z axis

        # Reset to default config A
        elif key == "r":
            axis_dim[0] = C_RE
            axis_dim[1] = C_IM
            axis_dim[2] = Z0_RE
            recalc_slider_dim_and_reset()
            update()

        # Quit
        elif key in ("q", "escape"):
            plt.close()

    plt.add_callback("key press", on_key)
    plt.show(interactive=True)


if __name__ == "__main__":
    main()
