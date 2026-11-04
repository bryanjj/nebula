"""Surface extraction shared by the CPU and hybrid experiments."""

import numpy as np
from skimage.measure import marching_cubes as _marching_cubes


def marching_cubes(volume, level=0.5, spacing=(1.0, 1.0, 1.0)):
    # A perfectly valid slice may contain no boundary at all.
    if not volume.min() < level < volume.max():
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    return _marching_cubes(volume, level=level, spacing=spacing)
