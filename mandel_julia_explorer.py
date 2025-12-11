import numpy as np
from numba import njit, prange
from skimage.measure import marching_cubes
from vedo import Plotter, Mesh

# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
N = 256
MAX_ITER = 128
BAILOUT = 4.0

X_RANGE = (-2.0, 2.0)
Y_RANGE = (-2.0, 2.0)
A_RANGE = (-2.0, 2.0)

IM_C_FIXED_INIT = 0
M_STEP = 0.02

# ----------------------------------------------------
# Numba 3D volume computation
# ----------------------------------------------------
@njit(parallel=True, fastmath=True)
def compute_volume(xs, ys, aas, m, max_iter, bailout):
    nx, ny, na = len(xs), len(ys), len(aas)
    volume = np.zeros((nx, ny, na), dtype=np.uint8)
    b2 = bailout * bailout

    for i in prange(nx):
        x = xs[i]
        for j in range(ny):
            y = ys[j]
            for k in range(na):
                a = aas[k]
                zr = x
                zi = y
                cr = a
                ci = m
                bounded = 1

                for _ in range(max_iter):
                    zr2 = zr*zr - zi*zi
                    zi2 = 2*zr*zi
                    zr = zr2 + cr
                    zi = zi2 + ci
                    if zr*zr + zi*zi > b2:
                        bounded = 0
                        break

                volume[i, j, k] = bounded

    return volume


def compute_surface(xs, ys, aas, m):
    print(f"[INFO] Computing for Im(c) = {m:.4f}")

    vol = compute_volume(xs, ys, aas, m, MAX_ITER, BAILOUT)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    da = aas[1] - aas[0]

    # marching cubes
    verts, faces, normals, values = marching_cubes(vol, level=0.5, spacing=(dx, dy, da))

    # shift coordinates so grid is real coordinates
    verts[:, 0] += xs[0]
    verts[:, 1] += ys[0]
    verts[:, 2] += aas[0]

    print(f"[INFO] Mesh: {len(verts)} vertices, {len(faces)} faces")
    return verts, faces


# ----------------------------------------------------
# Real-time explorer using vedo
# ----------------------------------------------------

def main():
    xs = np.linspace(*X_RANGE, N)
    ys = np.linspace(*Y_RANGE, N)
    aas = np.linspace(*A_RANGE, N)

    print("[INFO] Warming up Numba...")
    _ = compute_volume(xs, ys, aas, IM_C_FIXED_INIT, MAX_ITER, BAILOUT)

    # Initial mesh
    verts, faces = compute_surface(xs, ys, aas, IM_C_FIXED_INIT)
    mesh = Mesh([verts, faces]).c("cyan").lighting("plastic")

    plt = Plotter(bg="black", title="3D Mandelbrot–Julia Slice Explorer")
    plt.show(mesh, resetcam=True, interactive=False)

    state = {"m": IM_C_FIXED_INIT, "mesh": mesh}

    def update(delta_m):
        state["m"] += delta_m
        print(f"Updating M → {state['m']:.4f}")

        verts, faces = compute_surface(xs, ys, aas, state["m"])

        new_mesh = Mesh([verts, faces]).c("cyan").lighting("plastic")

        plt.remove(state["mesh"])
        state["mesh"] = new_mesh
        plt.add(new_mesh)
        plt.render()

    # Key bindings
    def keypress(evt):
        key = evt.keypress
        print(f"Key pressed: {key}")
        if key == "bracketleft":
            update(-M_STEP)
        elif key == "bracketright":
            update(+M_STEP)
        elif key == "r":
            update(IM_C_FIXED_INIT - state["m"])
        elif key in ("q", "Escape"):
            plt.close()

    plt.add_callback("key press", keypress)
    plt.show(interactive=True)


if __name__ == "__main__":
    main()
