"""
Full GPU Mandelbrot 4D Explorer using WebGPU
- Volume computation: GPU compute shader
- Marching cubes: GPU compute shader
- All rendering on GPU for real-time interaction

4D Fractal dimensions:
  0 = Re(z0)  - Real part of initial z
  1 = Im(z0)  - Imaginary part of initial z
  2 = Re(c)   - Real part of c
  3 = Im(c)   - Imaginary part of c

Press a/s/d/f to select which dimension is the slider.
Press [ ] to adjust slider value.
Press z to zoom in 1.3x (recompute at higher detail).
Press c to zoom out 1.3x.
Press x to reset to full domain.
"""

import numpy as np
import wgpu
import vedo
vedo.settings.enable_default_keyboard_callbacks = False
from vedo import Plotter, Mesh
import time

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------
N = 512  # Resolution (512^3 fits in GPU buffer limits)
MAX_ITER = 256
BAILOUT_SQ = 16.0

# Domain ranges for all 4 dimensions
RANGES = {
    0: (-2.0, 2.0),  # Re(z0)
    1: (-2.0, 2.0),  # Im(z0)
    2: (-2.0, 2.0),  # Re(c)
    3: (-2.0, 2.0),  # Im(c)
}

# Dimension names for display
DIM_NAMES = {
    0: "Re(z0)",
    1: "Im(z0)",
    2: "Re(c)",
    3: "Im(c)",
}

# ----------------------------------------------------
# DIMENSION CONFIGURATION
# ----------------------------------------------------
# Initial slider dimension (0-3). The other 3 dims auto-map to X, Y, Z.
# Press a/s/d/f keys to change which dim is the slider.
INITIAL_SLIDER_DIM = 1  # Start with Im(z0) as slider

# Fixed values for all dimensions (slider will override its dim)
INITIAL_DIM_VALUES = {
    0: 0.0,   # Re(z0)
    1: 0.0,   # Im(z0)
    2: 0.0,   # Re(c)
    3: 0.0,   # Im(c)
}

KEY_STEP = 0.001

def get_spatial_dims(slider_dim):
    """Return the 3 spatial dimensions (all dims except slider)."""
    return tuple(d for d in range(4) if d != slider_dim)

# ----------------------------------------------------
# Precompute coordinates for all dimensions
# ----------------------------------------------------
coords = {}
for dim in range(4):
    coords[dim] = np.linspace(*RANGES[dim], N).astype(np.float32)

# ----------------------------------------------------
# WEBGPU SETUP
# ----------------------------------------------------
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()

# ----------------------------------------------------
# MARCHING CUBES LOOKUP TABLES
# ----------------------------------------------------
EDGE_TABLE = np.array([
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
], dtype=np.int32)

TRI_TABLE = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
], dtype=np.int32)

# ----------------------------------------------------
# SHADER - Volume computation
# ----------------------------------------------------
def create_volume_shader(spatial_dims):
    """Generate WGSL shader with configured dimension mapping.

    Outputs a value encoding both boundedness and "depth":
    - Escaped points: value = escape_iter / max_iter (0 to ~0.5)
    - Bounded points: value = 0.5 + 0.5 * (1 - max_r2 / bailout_sq)
      where max_r2 is the maximum |z|² reached during iteration.
      Points that almost escaped get ~0.5, deep interior points get ~1.0
    """
    return f"""
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read> ys: array<f32>;
@group(0) @binding(2) var<storage, read> zs: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<f32>;  // Re(z0), Im(z0), Re(c), Im(c)
@group(0) @binding(4) var<storage, read_write> volume: array<f32>;
@group(0) @binding(5) var<uniform> iter_params: vec2<f32>;  // bailout_sq, max_iter

const N: u32 = {N}u;

// Dimension mapping: which fractal dim does each spatial axis use
const X_DIM: u32 = {spatial_dims[0]}u;  // 0=Re(z0), 1=Im(z0), 2=Re(c), 3=Im(c)
const Y_DIM: u32 = {spatial_dims[1]}u;
const Z_DIM: u32 = {spatial_dims[2]}u;

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    if (i >= N || j >= N || k >= N) {{
        return;
    }}

    // Start with uniform params (for non-spatial dims)
    var dims = params;  // [Re(z0), Im(z0), Re(c), Im(c)]

    // Override with spatial coordinates
    dims[X_DIM] = xs[i];
    dims[Y_DIM] = ys[j];
    dims[Z_DIM] = zs[k];

    // Extract fractal parameters
    let z0_re = dims[0];
    let z0_im = dims[1];
    let c_re = dims[2];
    let c_im = dims[3];

    let bailout_sq = iter_params.x;
    let max_iter = u32(iter_params.y);

    // Mandelbrot/Julia iteration: z = z^2 + c
    var zr = z0_re;
    var zi = z0_im;

    var escape_iter: u32 = max_iter;
    var sum_r2: f32 = 0.0;  // Accumulate |z|² for orbit trap coloring

    for (var iter: u32 = 0u; iter < max_iter; iter++) {{
        let r2 = zr * zr + zi * zi;
        sum_r2 += r2;

        if (r2 > bailout_sq) {{
            escape_iter = iter;
            break;
        }}

        let zr_new = zr * zr - zi * zi + c_re;
        zi = 2.0 * zr * zi + c_im;
        zr = zr_new;
    }}

    var result: f32;
    if (escape_iter < max_iter) {{
        // Escaped: value < 0.5 based on escape time
        result = f32(escape_iter) / f32(max_iter) * 0.5;
    }} else {{
        // Bounded: value > 0.5, use average orbit radius for variation
        // This creates variation based on how the orbit behaved
        let avg_r2 = sum_r2 / f32(max_iter);
        // Normalize: typical avg_r2 ranges from ~0 to ~bailout_sq
        let normalized = clamp(avg_r2 / bailout_sq, 0.0, 1.0);
        result = 0.5 + 0.5 * normalized;
    }}

    let idx = i + j * N + k * N * N;
    volume[idx] = result;
}}
"""

# ----------------------------------------------------
# SHADER - Marching cubes (GPU)
# ----------------------------------------------------
MC_GEN_SHADER = f"""
@group(0) @binding(0) var<storage, read> volume: array<f32>;
@group(0) @binding(1) var<storage, read> edge_table: array<i32>;
@group(0) @binding(2) var<storage, read> tri_table: array<i32>;
@group(0) @binding(3) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(4) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(5) var<uniform> grid_info: vec4<f32>;  // dx, dy, dz, iso_level

const N: u32 = {N}u;
const MAX_TRIS_PER_CELL: u32 = 5u;

fn get_volume(i: u32, j: u32, k: u32) -> f32 {{
    if (i >= N || j >= N || k >= N) {{
        return 0.0;
    }}
    return volume[i + j * N + k * N * N];
}}

// Returns (position, value for coloring)
fn interp_vertex(iso: f32, p1: vec3<f32>, p2: vec3<f32>, v1: f32, v2: f32) -> vec4<f32> {{
    if (abs(iso - v1) < 0.00001) {{ return vec4<f32>(p1, v1); }}
    if (abs(iso - v2) < 0.00001) {{ return vec4<f32>(p2, v2); }}
    if (abs(v1 - v2) < 0.00001) {{ return vec4<f32>(p1, v1); }}
    let mu = (iso - v1) / (v2 - v1);
    let pos = p1 + mu * (p2 - p1);
    // Use the BOUNDED side's value for coloring (the one > iso_level)
    // This gives us the orbit metric variation, not the constant iso_level
    var val = v1;
    if (v2 > v1) {{
        val = v2;
    }}
    return vec4<f32>(pos, val);
}}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    if (i >= N - 1u || j >= N - 1u || k >= N - 1u) {{
        return;
    }}

    let dx = grid_info.x;
    let dy = grid_info.y;
    let dz = grid_info.z;
    let iso_level = grid_info.w;

    // Get the 8 corner values
    let v0 = get_volume(i, j, k);
    let v1 = get_volume(i + 1u, j, k);
    let v2 = get_volume(i + 1u, j + 1u, k);
    let v3 = get_volume(i, j + 1u, k);
    let v4 = get_volume(i, j, k + 1u);
    let v5 = get_volume(i + 1u, j, k + 1u);
    let v6 = get_volume(i + 1u, j + 1u, k + 1u);
    let v7 = get_volume(i, j + 1u, k + 1u);

    // Calculate cube index
    var cube_index: u32 = 0u;
    if (v0 < iso_level) {{ cube_index |= 1u; }}
    if (v1 < iso_level) {{ cube_index |= 2u; }}
    if (v2 < iso_level) {{ cube_index |= 4u; }}
    if (v3 < iso_level) {{ cube_index |= 8u; }}
    if (v4 < iso_level) {{ cube_index |= 16u; }}
    if (v5 < iso_level) {{ cube_index |= 32u; }}
    if (v6 < iso_level) {{ cube_index |= 64u; }}
    if (v7 < iso_level) {{ cube_index |= 128u; }}

    let edge_bits = edge_table[cube_index];
    if (edge_bits == 0) {{
        return;
    }}

    // Corner positions
    let fi = f32(i);
    let fj = f32(j);
    let fk = f32(k);

    let p0 = vec3<f32>(fi * dx, fj * dy, fk * dz);
    let p1 = vec3<f32>((fi + 1.0) * dx, fj * dy, fk * dz);
    let p2 = vec3<f32>((fi + 1.0) * dx, (fj + 1.0) * dy, fk * dz);
    let p3 = vec3<f32>(fi * dx, (fj + 1.0) * dy, fk * dz);
    let p4 = vec3<f32>(fi * dx, fj * dy, (fk + 1.0) * dz);
    let p5 = vec3<f32>((fi + 1.0) * dx, fj * dy, (fk + 1.0) * dz);
    let p6 = vec3<f32>((fi + 1.0) * dx, (fj + 1.0) * dy, (fk + 1.0) * dz);
    let p7 = vec3<f32>(fi * dx, (fj + 1.0) * dy, (fk + 1.0) * dz);

    // Interpolate vertices on edges (position + value)
    var vert_list: array<vec4<f32>, 12>;

    if ((edge_bits & 1) != 0) {{ vert_list[0] = interp_vertex(iso_level, p0, p1, v0, v1); }}
    if ((edge_bits & 2) != 0) {{ vert_list[1] = interp_vertex(iso_level, p1, p2, v1, v2); }}
    if ((edge_bits & 4) != 0) {{ vert_list[2] = interp_vertex(iso_level, p2, p3, v2, v3); }}
    if ((edge_bits & 8) != 0) {{ vert_list[3] = interp_vertex(iso_level, p3, p0, v3, v0); }}
    if ((edge_bits & 16) != 0) {{ vert_list[4] = interp_vertex(iso_level, p4, p5, v4, v5); }}
    if ((edge_bits & 32) != 0) {{ vert_list[5] = interp_vertex(iso_level, p5, p6, v5, v6); }}
    if ((edge_bits & 64) != 0) {{ vert_list[6] = interp_vertex(iso_level, p6, p7, v6, v7); }}
    if ((edge_bits & 128) != 0) {{ vert_list[7] = interp_vertex(iso_level, p7, p4, v7, v4); }}
    if ((edge_bits & 256) != 0) {{ vert_list[8] = interp_vertex(iso_level, p0, p4, v0, v4); }}
    if ((edge_bits & 512) != 0) {{ vert_list[9] = interp_vertex(iso_level, p1, p5, v1, v5); }}
    if ((edge_bits & 1024) != 0) {{ vert_list[10] = interp_vertex(iso_level, p2, p6, v2, v6); }}
    if ((edge_bits & 2048) != 0) {{ vert_list[11] = interp_vertex(iso_level, p3, p7, v3, v7); }}

    // Generate triangles
    let tri_row = cube_index * 16u;
    for (var t: u32 = 0u; t < 15u; t += 3u) {{
        let e0 = tri_table[tri_row + t];
        if (e0 < 0) {{ break; }}
        let e1 = tri_table[tri_row + t + 1u];
        let e2 = tri_table[tri_row + t + 2u];

        let v_a = vert_list[e0];
        let v_b = vert_list[e1];
        let v_c = vert_list[e2];

        // Atomic increment to get vertex index (4 floats per vertex: x, y, z, escape_time)
        let base_idx = atomicAdd(&counter, 12u);

        // Store 3 vertices with escape time (12 floats)
        vertices[base_idx + 0u] = v_a.x;
        vertices[base_idx + 1u] = v_a.y;
        vertices[base_idx + 2u] = v_a.z;
        vertices[base_idx + 3u] = v_a.w;  // escape time
        vertices[base_idx + 4u] = v_b.x;
        vertices[base_idx + 5u] = v_b.y;
        vertices[base_idx + 6u] = v_b.z;
        vertices[base_idx + 7u] = v_b.w;  // escape time
        vertices[base_idx + 8u] = v_c.x;
        vertices[base_idx + 9u] = v_c.y;
        vertices[base_idx + 10u] = v_c.z;
        vertices[base_idx + 11u] = v_c.w;  // escape time
    }}
}}
"""


class FullGPUFractal:
    """GPU-accelerated fractal volume and marching cubes."""

    def __init__(self, spatial_dims, custom_ranges=None):
        self.spatial_dims = spatial_dims
        self.custom_ranges = custom_ranges  # Override RANGES if provided
        self._setup_buffers_and_pipelines(spatial_dims)

    def _setup_buffers_and_pipelines(self, spatial_dims):
        """Setup GPU buffers and pipelines for given spatial dimensions."""
        self.spatial_dims = spatial_dims

        # Use custom ranges if provided, otherwise use global RANGES
        ranges = self.custom_ranges if self.custom_ranges else RANGES

        # Get coordinate arrays for spatial dimensions
        self.xs_np = np.linspace(*ranges[spatial_dims[0]], N).astype(np.float32)
        self.ys_np = np.linspace(*ranges[spatial_dims[1]], N).astype(np.float32)
        self.zs_np = np.linspace(*ranges[spatial_dims[2]], N).astype(np.float32)

        self.dx = self.xs_np[1] - self.xs_np[0]
        self.dy = self.ys_np[1] - self.ys_np[0]
        self.dz = self.zs_np[1] - self.zs_np[0]

        # Coordinate origin for shifting mesh to world coordinates
        self.origin = np.array([
            ranges[spatial_dims[0]][0],
            ranges[spatial_dims[1]][0],
            ranges[spatial_dims[2]][0]
        ], dtype=np.float32)

        # Create coordinate buffers
        self.xs_buf = device.create_buffer_with_data(data=self.xs_np.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        self.ys_buf = device.create_buffer_with_data(data=self.ys_np.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        self.zs_buf = device.create_buffer_with_data(data=self.zs_np.tobytes(), usage=wgpu.BufferUsage.STORAGE)

        # Volume buffer: N^3 floats
        vol_size = N * N * N * 4
        self.vol_buf = device.create_buffer(size=vol_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

        # Params: [Re(z0), Im(z0), Re(c), Im(c)]
        self.params_buf = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        # Iter params: [bailout_sq, max_iter]
        self.iter_params_buf = device.create_buffer(size=8, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        # Create volume shader
        vol_shader_code = create_volume_shader(spatial_dims)
        self.vol_shader = device.create_shader_module(code=vol_shader_code)
        self.vol_pipeline = device.create_compute_pipeline(layout="auto", compute={"module": self.vol_shader, "entry_point": "main"})

        self.vol_bind_group = device.create_bind_group(
            layout=self.vol_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.xs_buf}},
                {"binding": 1, "resource": {"buffer": self.ys_buf}},
                {"binding": 2, "resource": {"buffer": self.zs_buf}},
                {"binding": 3, "resource": {"buffer": self.params_buf}},
                {"binding": 4, "resource": {"buffer": self.vol_buf}},
                {"binding": 5, "resource": {"buffer": self.iter_params_buf}},
            ]
        )

        # --- Marching cubes GPU setup ---
        # Lookup tables
        self.edge_table_buf = device.create_buffer_with_data(data=EDGE_TABLE.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        self.tri_table_buf = device.create_buffer_with_data(data=TRI_TABLE.tobytes(), usage=wgpu.BufferUsage.STORAGE)

        # Max triangles: realistic estimate - surface area scales as N^2, not N^3
        # For a 512^3 volume, surface is roughly proportional to 6*N^2 cells on boundary
        # Plus some interior detail. 10M triangles is generous for most fractals.
        max_tris = 10_000_000
        # 3 vertices per tri, 4 floats per vertex (x, y, z, escape_time)
        max_floats = max_tris * 3 * 4
        self.verts_buf = device.create_buffer(size=max_floats * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

        # Atomic counter
        self.counter_buf = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

        # Grid info: [dx, dy, dz, iso_level]
        self.grid_info_buf = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        # MC shader and pipeline
        self.mc_shader = device.create_shader_module(code=MC_GEN_SHADER)
        self.mc_pipeline = device.create_compute_pipeline(layout="auto", compute={"module": self.mc_shader, "entry_point": "main"})

        self.mc_bind_group = device.create_bind_group(
            layout=self.mc_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.vol_buf}},
                {"binding": 1, "resource": {"buffer": self.edge_table_buf}},
                {"binding": 2, "resource": {"buffer": self.tri_table_buf}},
                {"binding": 3, "resource": {"buffer": self.verts_buf}},
                {"binding": 4, "resource": {"buffer": self.counter_buf}},
                {"binding": 5, "resource": {"buffer": self.grid_info_buf}},
            ]
        )

        # Staging buffers
        self.counter_staging = device.create_buffer(size=4, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
        # Staging buffer - same size as verts_buf
        self.max_verts_bytes = max_floats * 4
        self.verts_staging = device.create_buffer(size=self.max_verts_bytes, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)

    def set_slider_dim(self, slider_dim):
        """Reconfigure for a new slider dimension."""
        new_spatial = get_spatial_dims(slider_dim)
        if new_spatial != self.spatial_dims:
            self._setup_buffers_and_pipelines(new_spatial)

    def set_ranges(self, custom_ranges):
        """Update the rendering domain ranges and rebuild buffers."""
        self.custom_ranges = custom_ranges
        self._setup_buffers_and_pipelines(self.spatial_dims)

    def compute(self, dim_values):
        """Compute the fractal surface using full GPU pipeline."""
        t0 = time.perf_counter()

        # Build params array
        params = np.array([
            dim_values.get(0, 0.0),
            dim_values.get(1, 0.0),
            dim_values.get(2, 0.0),
            dim_values.get(3, 0.0),
        ], dtype=np.float32)
        device.queue.write_buffer(self.params_buf, 0, params.tobytes())

        # Iter params
        iter_params = np.array([BAILOUT_SQ, MAX_ITER], dtype=np.float32)
        device.queue.write_buffer(self.iter_params_buf, 0, iter_params.tobytes())

        # Grid info for MC
        grid_info = np.array([self.dx, self.dy, self.dz, 0.5], dtype=np.float32)
        device.queue.write_buffer(self.grid_info_buf, 0, grid_info.tobytes())

        # Reset counter
        device.queue.write_buffer(self.counter_buf, 0, np.array([0], dtype=np.uint32).tobytes())

        # Create command encoder
        encoder = device.create_command_encoder()

        # Volume compute pass
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.vol_pipeline)
        compute_pass.set_bind_group(0, self.vol_bind_group)
        wg = (N + 7) // 8
        compute_pass.dispatch_workgroups(wg, wg, wg)
        compute_pass.end()

        # Marching cubes pass
        mc_pass = encoder.begin_compute_pass()
        mc_pass.set_pipeline(self.mc_pipeline)
        mc_pass.set_bind_group(0, self.mc_bind_group)
        mc_wg = (N - 1 + 7) // 8
        mc_pass.dispatch_workgroups(mc_wg, mc_wg, mc_wg)
        mc_pass.end()

        # Copy counter to staging
        encoder.copy_buffer_to_buffer(self.counter_buf, 0, self.counter_staging, 0, 4)

        device.queue.submit([encoder.finish()])

        # Read counter
        self.counter_staging.map_sync(wgpu.MapMode.READ)
        num_floats = np.frombuffer(self.counter_staging.read_mapped(), dtype=np.uint32)[0]
        self.counter_staging.unmap()

        t1 = time.perf_counter()

        if num_floats == 0:
            print(f"[GPU] No surface found. Time: {(t1-t0)*1000:.1f}ms")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.float32)

        # Check for buffer overflow
        if num_floats * 4 > self.max_verts_bytes:
            print(f"[GPU] WARNING: Triangle count exceeded buffer! Truncating.")
            num_floats = self.max_verts_bytes // 4

        # Copy vertices
        encoder2 = device.create_command_encoder()
        encoder2.copy_buffer_to_buffer(self.verts_buf, 0, self.verts_staging, 0, num_floats * 4)
        device.queue.submit([encoder2.finish()])

        self.verts_staging.map_sync(wgpu.MapMode.READ)
        verts_data = np.frombuffer(self.verts_staging.read_mapped()[:num_floats * 4], dtype=np.float32).copy()
        self.verts_staging.unmap()

        # Reshape: each vertex is 4 floats (x, y, z, escape_time)
        num_verts = num_floats // 4
        verts_data = verts_data.reshape(-1, 4)

        # Extract positions and escape times
        verts = verts_data[:, :3].copy()
        escape_times = verts_data[:, 3]

        # Shift to world coordinates
        verts += self.origin

        # Convert escape time to RGB colors using a colormap
        # Debug: print the actual range of values
        print(f"[DEBUG] escape_times range: {escape_times.min():.3f} - {escape_times.max():.3f}")
        colors = self._escape_time_to_color(escape_times)

        # Create faces (every 3 vertices is a triangle)
        num_tris = num_verts // 3
        faces = np.arange(num_verts).reshape(-1, 3)

        t2 = time.perf_counter()

        print(f"[GPU] Vol+MC: {(t1-t0)*1000:.1f}ms, Read: {(t2-t1)*1000:.1f}ms, Tris: {num_tris}")

        return verts, faces, colors

    def _escape_time_to_color(self, escape_times):
        """Convert escape times to RGB colors using a nice colormap.

        Values should vary based on the "depth" metric from the volume shader.
        """
        # Get actual range for normalization
        vmin, vmax = escape_times.min(), escape_times.max()
        if vmax - vmin < 0.001:
            # No variation - use fallback
            t = np.zeros_like(escape_times)
        else:
            # Normalize to [0, 1] based on actual range
            t = (escape_times - vmin) / (vmax - vmin)

        # Inferno-like colormap: black -> purple -> red -> orange -> yellow
        r = np.clip(1.5 * t - 0.1, 0, 1)
        g = np.clip(1.5 * t - 0.5, 0, 1)
        b = np.clip(2.5 * (0.4 - abs(t - 0.3)), 0, 1)  # Peak blue/purple early

        colors = np.stack([r, g, b], axis=1)
        return (colors * 255).astype(np.uint8)


def main():
    from vedo import Text2D

    # Initialize state
    slider_dim = INITIAL_SLIDER_DIM
    spatial_dims = get_spatial_dims(slider_dim)
    dim_values = dict(INITIAL_DIM_VALUES)

    # Print configuration
    print("=" * 60)
    print("4D Mandelbrot Explorer - Full GPU Pipeline")
    print("=" * 60)
    print(f"Resolution: {N}^3 = {N**3:,} voxels")
    print("Controls:")
    print("  [ ] = adjust slider value")
    print("  a/s/d/f = select slider dimension:")
    print("    a = Re(z0), s = Im(z0), d = Re(c), f = Im(c)")
    print("  z = zoom in 1.3x (recompute at higher detail)")
    print("  c = zoom out 1.3x")
    print("  x = reset to full domain")
    print("  ESC = quit")
    print("=" * 60)

    gpu = FullGPUFractal(spatial_dims)

    print("[INFO] Warming up...")
    gpu.compute(dim_values)

    def compute_surface():
        return gpu.compute(dim_values)

    verts, faces, colors = compute_surface()
    mesh = Mesh([verts, faces])
    mesh.pointcolors = colors
    mesh.lighting("plastic")

    # Track current domain ranges (for zoom)
    current_ranges = {d: RANGES[d] for d in range(4)}

    state = {
        "slider_dim": slider_dim,
        "dim_values": dim_values,
        "mesh": mesh,
        "axis_text": None,
        "slider_text": None,
        "gpu": gpu,
        "ranges": current_ranges,
    }

    plt = Plotter(bg="black", title="4D Mandelbrot Explorer", axes=0)
    plt.show(mesh, resetcam=True, interactive=False)

    # Remove all existing KeyPress observers from the interactor to disable vedo defaults
    plt.interactor.RemoveObservers("KeyPressEvent")

    # Custom interaction style: right-click to pan
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

    class CustomInteractorStyle(vtkInteractorStyleTrackballCamera):
        def __init__(self):
            super().__init__()
            self.AddObserver("RightButtonPressEvent", self._right_down)
            self.AddObserver("RightButtonReleaseEvent", self._right_up)

        def _right_down(self, obj, event):
            self.OnMiddleButtonDown()

        def _right_up(self, obj, event):
            self.OnMiddleButtonUp()

    style = CustomInteractorStyle()
    plt.interactor.SetInteractorStyle(style)

    def get_axis_info():
        sd = state["slider_dim"]
        sp = get_spatial_dims(sd)
        return f"X: {DIM_NAMES[sp[0]]}\nY: {DIM_NAMES[sp[1]]}\nZ: {DIM_NAMES[sp[2]]}"

    def get_slider_info():
        sd = state["slider_dim"]
        val = state["dim_values"][sd]
        return f"Slider: {DIM_NAMES[sd]} = {val:.3f}  ([ ] to adjust, a/s/d/f to switch)"

    # UI text
    axis_text = Text2D(get_axis_info(), pos="bottom-left", c="white", s=0.8)
    plt.add(axis_text)
    state["axis_text"] = axis_text

    slider_text = Text2D(get_slider_info(), pos="top-center", c="yellow", s=0.9)
    plt.add(slider_text)
    state["slider_text"] = slider_text

    def update_mesh(vis):
        verts, faces, colors = compute_surface()
        if len(verts) == 0:
            return

        new_mesh = Mesh([verts, faces])
        new_mesh.pointcolors = colors
        new_mesh.lighting("plastic")

        vis.remove(state["mesh"])
        state["mesh"] = new_mesh
        vis.add(new_mesh)
        vis.render()

    def update_ui():
        state["axis_text"].text(get_axis_info())
        state["slider_text"].text(get_slider_info())

    def switch_slider(new_slider_dim):
        if new_slider_dim == state["slider_dim"]:
            return
        print(f"[INFO] Switching slider to {DIM_NAMES[new_slider_dim]}")
        state["slider_dim"] = new_slider_dim
        state["gpu"].set_slider_dim(new_slider_dim)
        update_ui()
        update_mesh(plt)

    def refine_to_view(zoom_factor=0.5):
        """Zoom into the center of current view by a factor (0.5 = half the size)."""
        sd = state["slider_dim"]
        sp = get_spatial_dims(sd)

        # Get current ranges
        old_ranges = state["ranges"]

        # Get focal point as center for zoom
        cam = plt.camera
        focal = np.array(cam.GetFocalPoint())
        cx, cy, cz = focal

        new_ranges = dict(old_ranges)

        # For each spatial dimension, zoom in around the focal point
        for i, dim in enumerate(sp):
            old_min, old_max = old_ranges[dim]
            old_size = old_max - old_min
            new_size = old_size * zoom_factor

            # Center of zoom (use focal point coordinate)
            center = [cx, cy, cz][i]

            # Clamp center to be within current range
            center = max(old_min + new_size/2, min(old_max - new_size/2, center))

            new_min = center - new_size / 2
            new_max = center + new_size / 2

            # Clamp to original RANGES bounds
            orig_min, orig_max = RANGES[dim]
            new_min = max(new_min, orig_min)
            new_max = min(new_max, orig_max)

            new_ranges[dim] = (float(new_min), float(new_max))

        # Calculate effective zoom level
        old_size = old_ranges[sp[0]][1] - old_ranges[sp[0]][0]
        new_size = new_ranges[sp[0]][1] - new_ranges[sp[0]][0]
        total_zoom = RANGES[sp[0]][1] - RANGES[sp[0]][0]

        print(f"[REFINE] Zooming {zoom_factor}x around focal point")
        print(f"  X range: {new_ranges[sp[0]][0]:.4f} to {new_ranges[sp[0]][1]:.4f}")
        print(f"  Y range: {new_ranges[sp[1]][0]:.4f} to {new_ranges[sp[1]][1]:.4f}")
        print(f"  Z range: {new_ranges[sp[2]][0]:.4f} to {new_ranges[sp[2]][1]:.4f}")
        print(f"  Total zoom: {total_zoom / new_size:.1f}x from original")

        state["ranges"] = new_ranges
        state["gpu"].set_ranges(new_ranges)
        update_mesh(plt)

    def reset_view():
        """Reset to full domain."""
        print("[RESET] Restoring full domain")
        state["ranges"] = {d: RANGES[d] for d in range(4)}
        state["gpu"].set_ranges(state["ranges"])
        update_mesh(plt)
        plt.reset_camera()

    # Use VTK interactor directly to intercept keys before vedo's default handling
    def on_key_press(obj, event):
        key = obj.GetKeySym()
        sd = state["slider_dim"]
        dv = state["dim_values"]

        print(f"KEY: {key} | Slider: {DIM_NAMES[sd]}={dv[sd]:.3f}")

        # Handle our custom keys
        if key == "bracketleft":  # [
            dv[sd] -= KEY_STEP
            update_ui()
            update_mesh(plt)
        elif key == "bracketright":  # ]
            dv[sd] += KEY_STEP
            update_ui()
            update_mesh(plt)
        # Use a, s, d, f for dimension selection
        elif key == "a":
            switch_slider(0)  # Re(z0)
        elif key == "s":
            switch_slider(1)  # Im(z0)
        elif key == "d":
            switch_slider(2)  # Re(c)
        elif key == "f":
            switch_slider(3)  # Im(c)
        # Zoom refinement (gentler zoom to avoid breaking through surface)
        elif key == "z":
            refine_to_view(0.95)  # Zoom in ~1.33x
        elif key == "c":
            refine_to_view(1.33)  # Zoom out ~1.33x
        elif key == "x":
            reset_view()
        elif key == "Escape":
            plt.close()

    # Add observer directly to VTK interactor (priority over vedo's handlers)
    plt.interactor.AddObserver("KeyPressEvent", on_key_press, 10.0)
    plt.show(interactive=True)


if __name__ == "__main__":
    main()
