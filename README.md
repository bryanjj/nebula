# Nebula

[Open the live explorer on GitHub Pages](https://bryanjj.github.io/nebula/)

Interactive 3D slices of the four-dimensional Mandelbrot–Julia family:

```text
z(n + 1) = z(n)² + c
four real dimensions = Re(z₀), Im(z₀), Re(c), Im(c)
```

Choose three dimensions for the visible volume and move through the fourth with a slider. The browser explorer raymarches a pseudo-distance field; the Python experiments compute voxel grids and extract meshes with marching cubes. The browser distance estimate is a visual approximation, so thin details and lighting can vary with zoom and resolution.

## Quick start: browser

No build step, npm install, or third-party runtime packages are needed. From the repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open **http://localhost:8000** in a WebGPU-enabled browser with hardware acceleration. Use localhost or HTTPS; opening the HTML directly is not the supported startup path. If WebGPU cannot initialize, the status area shows the failure and a Retry button.

- Drag (or single-finger drag) to orbit.
- Right-drag or Shift-drag to pan.
- Scroll up to zoom in; scroll down to zoom out.
- Choose **Re(z₀)**, **Im(z₀)**, **Re(c)**, or **Im(c)** in the dropdown above the slice slider. The slider controls that coordinate; the other three form the visible volume. An axis guide in the canvas corner rotates with the camera to show the visible coordinates’ directions. Its labels change with the selected slider dimension.
- **Clip plane** cuts into the volume; its minimum disables clipping. The cut stays fixed to the fractal as you orbit, zoom, or pan. **Align to view** smoothly rotates the plane to face the current camera while keeping the depth value.
- **Reset view** (or `R`) restores only the camera orientation. It preserves the slider dimension, both slider values, clip-plane orientation, zoom, and pan. Refresh the page to restore all defaults.
- Focus the canvas for arrow-key orbit, `+` / `−` zoom, and `R` reset.

Rendering always uses the High render scale, capped at two million pixels, and only runs after input, resize, or returning to the tab.

`four_d_fractal_webgpu.html` forwards to the same explorer. `cam_test.html` and `test_webgpu.html` remain small standalone camera/WebGPU experiments.

## Python setup

Use **Python 3.12** (the tested interpreter). A virtual environment keeps these scientific and graphics packages separate from other projects:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fractal_4d_explorer.py
```

The interactive Python viewers use Vedo/VTK and need a desktop display. Their keyboard shortcuts are printed in the source/window titles. The main CPU explorer uses `[` / `]` for the slice, `X` / `Y` / `Z` to cycle axis mappings, `R` to reset, and `Q` / Escape to quit.

Every Python demo starts at **96³** voxels. The first CPU computation also compiles Numba kernels. Increase detail explicitly:

```sh
NEBULA_RESOLUTION=128 python fractal_4d_explorer.py
```

Allowed resolutions are 8–512. Work and memory grow roughly with the cube of resolution: a 512³ grid has about 152 times as many voxels as a 96³ grid. Start small before increasing it. Slices with no boundary produce an empty mesh instead of a marching-cubes error.

| Script | Purpose | Dependencies |
| --- | --- | --- |
| `fractal_4d_explorer.py` | CPU explorer with configurable axes and a fourth-coordinate slider | `requirements.txt` |
| `mandel_julia_explorer.py` | CPU slice through Re(z₀), Im(z₀), Re(c), with fixed Im(c) | `requirements.txt` |
| `mandel_face_slice.py` | CPU volume with Im(c) and Re(z₀) offsets | `requirements.txt` |
| `fractal_slice_poc.py` | Headless STL export | `requirements.txt` |
| `mandel_webgpu.py` | WebGPU volume computation with CPU surface extraction | `requirements-gpu.txt` |
| `mandel_full_gpu.py` | WebGPU volume computation and marching cubes, then Vedo display | `requirements-gpu.txt` |
| `mandel_face_slice_taichi.py` | Taichi volume computation with CPU surface extraction | `requirements-taichi.txt` |

For the face-slice variants, X/Y/Z are Re(c), Im(c), Re(z₀). Their `Im(c)` control adds an offset to the Y coordinate, and `Re(z₀)` adds an offset to the Z coordinate.

### GPU variants

```sh
python -m pip install -r requirements-gpu.txt
python mandel_full_gpu.py
```

Native WebGPU requires an available graphics adapter and working system drivers. The full GPU explorer uses `A` / `S` / `D` / `F` to choose the slider dimension, `[` / `]` to move it, `Z` / `C` to change the sampling domain, and `X` to reset the domain. Geometry is computed on the GPU and read back for display in Vedo.

The GPU demos request the volume buffer limits they need, use portable 4×4×4 workgroups, and size the full GPU triangle buffer within the device limits. If a volume or surface exceeds available capacity, reduce `NEBULA_RESOLUTION`. The full GPU path reports surface overflow rather than returning a silently truncated mesh.

For Taichi:

```sh
python -m pip install -r requirements-taichi.txt
python mandel_face_slice_taichi.py
```

Taichi chooses an available GPU backend and can fall back to CPU. It no longer assumes an Apple Metal GPU. To force CPU execution when a GPU backend is detected but cannot run, use `TI_ARCH=x64 python mandel_face_slice_taichi.py`.

### Export an STL

```sh
NEBULA_RESOLUTION=64 python fractal_slice_poc.py
# Or choose a destination:
NEBULA_RESOLUTION=64 python fractal_slice_poc.py --output generated/my-slice.stl
```

Exports go to `generated/fractal_slice_surface.stl` by default, in fractal world coordinates. This avoids overwriting the large historical `fractal_slice_surface.stl` checked into the repository. Importing the exporter no longer computes or writes a mesh.

## Development checks

CPU tests exercise the complex recurrence, all four axis mappings, slider offsets, empty slices, STL coordinates, and GPU limit requests:

```sh
NEBULA_RESOLUTION=16 python -m unittest discover -s tests -v
```

On a machine with native WebGPU, install `requirements-gpu.txt` and include the GPU volume/mesh tests:

```sh
NEBULA_RESOLUTION=16 NEBULA_TEST_GPU=1 python -m unittest discover -s tests -v
```

Driver-independent renderer scheduling and GPU recovery checks can run with `npm run test:renderer`. For the full browser checks, use Node.js 22 or newer:

```sh
npm ci
npx playwright install chromium
npm test
```

The browser tests cover initialization failures, narrow-screen layout, controls/reset, the old entry point, and shader rendering when canvas presentation is available. An independent triangle-rendering probe checks that capability first; headless systems can expose an adapter that cannot present a canvas. Set `NEBULA_TEST_GPU=1 npm test` to require this capability rather than skip the renderer check. The test server uses port 8766; keep it free.

GitHub Actions runs the CPU and browser suites on pushes and pull requests. The browser viewer remains a static site and does not depend on Node.js in production.
