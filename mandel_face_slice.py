import numpy as np
from numba import njit, prange
from skimage.measure import marching_cubes
from vedo import Plotter, Mesh

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------
N = 512
MAX_ITER = 80
BAILOUT = 4.0

# Domain for the 3-D grid:
# x = Re(c), y = Im(c), z = Re(z0)
C_RE_RANGE  = (-2.0,  2.0)
C_IM_RANGE  = (-2.0,  2.0)
Z0_RE_RANGE = (-2.0,  2.0)   # z=0 face gives Mandelbrot if z0=0

# PARAMETER SLIDERS
IM_C_INIT = 0.20      # starting imaginary part of c
Z0_RE_INIT = 0.00      # starting real part of z0

IM_STEP = 0.02
Z0_STEP = 0.02


# ----------------------------------------------------
# NUMBA ITERATION ENGINE
# ----------------------------------------------------
@njit(parallel=True, fastmath=True)
def compute_volume(xs, ys, zs, im_c, z0_re, max_iter, bailout):
    """
    xs = Re(c)
    ys = Im(c)
    zs = Re(z0) grid axis (not parameter)
    im_c = parameter Im(c)
    z0_re = parameter Re(z0)
    """
    nx, ny, nz = len(xs), len(ys), len(zs)
    V = np.zeros((nx, ny, nz), dtype=np.uint8)
    b2 = bailout * bailout

    for i in prange(nx):
        c_re = xs[i]
        for j in range(ny):
            c_im = ys[j]
            for k in range(nz):

                # z0 real part varies with slider, imaginary = 0
                zr = z0_re
                zi = 0.0

                # but we give the "thickness" shape along zs[k]
                # by modifying the orbit differently per z-slice
                zr += zs[k]   # OR zr = zs[k]  if you want pure volume
                # If you prefer instead: zr = zs[k] and z0_re is global offset:
                # zr = zs[k] + z0_re

                bounded = 1
                for _ in range(max_iter):
                    zr2 = zr*zr - zi*zi
                    zi2 = 2.0*zr*zi
                    zr = zr2 + c_re
                    zi = zi2 + c_im

                    # escape check
                    if zr*zr + zi*zi > b2:
                        bounded = 0
                        break

                V[i,j,k] = bounded

    return V



# ----------------------------------------------------
# SURFACE COMPUTATION
# ----------------------------------------------------
def compute_surface(xs, ys, zs, im_c, z0_re):
    print(f"[INFO] Computing volume: Im(c)={im_c:.4f}, Re(z0)={z0_re:.4f}")
    V = compute_volume(xs, ys, zs, im_c, z0_re, MAX_ITER, BAILOUT)

    dx = xs[1]-xs[0]
    dy = ys[1]-ys[0]
    dz = zs[1]-zs[0]

    print("[INFO] Marching cubes...")
    verts, faces, normals, values = marching_cubes(
        V,
        level=0.5,
        spacing=(dx, dy, dz)
    )

    # shift to coordinate system
    verts[:,0] += xs[0]
    verts[:,1] += ys[0]
    verts[:,2] += zs[0]

    print(f"[INFO] Mesh: {len(verts)} verts / {len(faces)} faces")
    return verts, faces, normals



# ----------------------------------------------------
# MAIN EXPLORER
# ----------------------------------------------------
def main():

    xs = np.linspace(*C_RE_RANGE,  N)
    ys = np.linspace(*C_IM_RANGE,  N)
    zs = np.linspace(*Z0_RE_RANGE, N)

    print("[INFO] Warming Numba...")
    _ = compute_volume(xs, ys, zs, IM_C_INIT, Z0_RE_INIT, MAX_ITER, BAILOUT)

    # initial mesh
    verts, faces, normals = compute_surface(xs, ys, zs, IM_C_INIT, Z0_RE_INIT)
    mesh = Mesh([verts, faces]).c("cyan").lighting("plastic")

    state = {
        "im_c": IM_C_INIT,
        "z0_re": Z0_RE_INIT,
        "mesh": mesh,
        "xs": xs, "ys": ys, "zs": zs
    }

    plt = Plotter(bg="black", title="3D Explorer: Im(c) and Re(z0) sliders")
    plt.show(mesh, resetcam=True, interactive=False)


    # ------------- Mesh updater ---------------------
    def update(vis):
        verts, faces, normals = compute_surface(
            state["xs"], state["ys"], state["zs"],
            state["im_c"], state["z0_re"]
        )

        # Rebuild mesh from scratch
        new_mesh = Mesh([verts, faces])
        new_mesh.c("cyan").lighting("plastic")
        new_mesh.compute_normals()

        # Replace old mesh
        vis.remove(state["mesh"])
        state["mesh"] = new_mesh
        vis.add(new_mesh)

        vis.render()


    # ------------- KEY HANDLER ----------------------
    def on_key(evt):
        key = evt.keypress
        print("KEY:", key)

        if key == "bracketleft":      # [
            state["im_c"] -= IM_STEP
            update(plt)

        elif key == "bracketright":   # ]
            state["im_c"] += IM_STEP
            update(plt)

        elif key == "z":          # ,
            state["z0_re"] -= Z0_STEP
            update(plt)

        elif key == "x":         # .
            state["z0_re"] += Z0_STEP
            update(plt)

        elif key in ("q", "escape"):
            plt.close()


    plt.add_callback("key press", on_key)
    plt.show(interactive=True)



if __name__ == "__main__":
    main()
