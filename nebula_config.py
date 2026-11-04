"""Shared startup settings for the desktop experiments."""

import os


def grid_size():
    """Use a modest first-run grid; allow explicit detail via the environment."""
    try:
        value = int(os.environ.get("NEBULA_RESOLUTION", "96"))
    except ValueError as error:
        raise ValueError("NEBULA_RESOLUTION must be an integer from 8 to 512") from error
    if not 8 <= value <= 512:
        raise ValueError("NEBULA_RESOLUTION must be an integer from 8 to 512")
    return value


def request_gpu_device(wgpu, volume_bytes):
    """Request the limits actually needed for a volume, or fail before allocation."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    required = {}
    for name in ("max-buffer-size", "max-storage-buffer-binding-size"):
        if volume_bytes > adapter.limits[name]:
            raise RuntimeError(
                f"This GPU cannot hold the requested volume ({volume_bytes:,} bytes). "
                "Lower NEBULA_RESOLUTION, for example to 96."
            )
        required[name] = max(volume_bytes, min(adapter.limits[name], 128 * 1024**2))
    return adapter.request_device_sync(required_limits=required)
