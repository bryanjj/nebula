import argparse
from pathlib import Path
from nebula_config import grid_size
import numpy as np
import numba
from nebula_surface import marching_cubes
from stl import mesh

# -------------------------
# Parameters for the volume
# -------------------------
N = grid_size()
max_iter = 512        # iterations for the escape test
bailout = 4.0        # escape radius

# 3-D slice: Im(c) = fixed value
IM_C_FIXED = 0.2     # this defines the slice through C

# Domain ranges
bound = 2.0
x_range = (-bound, bound)   # Re(z0)
y_range = (-bound, bound)   # Im(z0)
a_range = (-bound, bound)   # Re(c)

# -------------------------
# Iteration function (JIT compiled)
# -------------------------
@numba.jit(nopython=True)
def iterate_point(zr, zi, cr, ci, max_iter, bailout_sq):
    """Return 1 if bounded, 0 if escaped."""
    for _ in range(max_iter):
        zr_new = zr * zr - zi * zi + cr
        zi = 2 * zr * zi + ci
        zr = zr_new
        if zr * zr + zi * zi > bailout_sq:
            return 0
    return 1


@numba.jit(nopython=True, parallel=True)
def fill_volume(xs, ys, as_, im_c, max_iter, bailout_sq):
    """Fill the voxel grid in parallel."""
    n = len(xs)
    volume = np.zeros((n, n, n), dtype=np.uint8)
    for i in numba.prange(n):
        x = xs[i]
        for j in range(n):
            y = ys[j]
            for k in range(n):
                a = as_[k]
                volume[i, j, k] = iterate_point(x, y, a, im_c, max_iter, bailout_sq)
    return volume


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export a 3D fractal slice as an STL mesh.")
    parser.add_argument("--output", type=Path, default=Path("generated/fractal_slice_surface.stl"))
    args = parser.parse_args(argv)

    print(f"Generating {N}³ voxel grid…")
    xs = np.linspace(*x_range, N)
    ys = np.linspace(*y_range, N)
    as_ = np.linspace(*a_range, N)
    volume = fill_volume(xs, ys, as_, IM_C_FIXED, max_iter, bailout * bailout)
    verts, faces, _, _ = marching_cubes(
        volume, level=0.5,
        spacing=(xs[1] - xs[0], ys[1] - ys[0], as_[1] - as_[0]),
    )
    if not len(faces):
        raise RuntimeError("This slice contains no surface to export.")
    # Marching cubes returns coordinates relative to the volume origin.
    verts += np.array([xs[0], ys[0], as_[0]])
    result = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    result.vectors = verts[faces]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(args.output))
    print(f"Saved {len(faces):,} triangles to {args.output}")


if __name__ == "__main__":
    main()
