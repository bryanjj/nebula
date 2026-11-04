import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from stl import mesh

from nebula_config import grid_size, request_gpu_device
from nebula_surface import marching_cubes
import fractal_4d_explorer as explorer
import fractal_slice_poc as exporter
import mandel_face_slice as face
import mandel_julia_explorer as julia


def bounded(z, c, iterations=24, bailout=4):
    for _ in range(iterations):
        z = z * z + c
        if abs(z) ** 2 > bailout**2:
            return 0
    return 1


class FractalTests(unittest.TestCase):
    def test_axis_mappings_match_complex_recurrence(self):
        # Asymmetric domains expose swapped axes and initial-z/c mixups.
        low = np.array([-1.3, -0.7, -1.8, -1.1])
        high = np.array([0.8, 1.2, 0.6, 0.9])
        for slider_dim in range(4):
            axes = np.array([d for d in range(4) if d != slider_dim])
            volume = explorer.compute_volume(5, axes, slider_dim, 0.2, low, high, 24, 4)
            expected = np.empty_like(volume)
            for index in np.ndindex(volume.shape):
                dims = np.zeros(4)
                dims[slider_dim] = 0.2
                for axis, dim in enumerate(axes):
                    dims[dim] = np.linspace(low[dim], high[dim], 5)[index[axis]]
                expected[index] = bounded(complex(*dims[:2]), complex(*dims[2:]))
            np.testing.assert_array_equal(volume, expected)

    def test_face_slider_offsets_change_the_orbit(self):
        xs = np.array([-1.4, -0.3, 0.4])
        ys = np.array([-0.8, 0.1, 0.6])
        zs = np.array([-1.0, 0.0, 0.7])
        for offset in (0.0, 0.6):
            volume = face.compute_volume(xs, ys, zs, offset, 0.15, 24, 4)
            expected = np.empty_like(volume)
            for i, j, k in np.ndindex(volume.shape):
                expected[i, j, k] = bounded(zs[k] + 0.15, complex(xs[i], ys[j] + offset))
            np.testing.assert_array_equal(volume, expected)
        self.assertFalse(np.array_equal(
            face.compute_volume(xs, ys, zs, 0.0, 0.15, 24, 4),
            face.compute_volume(xs, ys, zs, 0.6, 0.15, 24, 4),
        ))

    def test_julia_grid_matches_complex_recurrence(self):
        coords = np.linspace(-1.5, 1.5, 5)
        volume = julia.compute_volume(coords, coords, coords, 0.2, 24, 4)
        for i, j, k in np.ndindex(volume.shape):
            self.assertEqual(volume[i, j, k], bounded(complex(coords[i], coords[j]), complex(coords[k], 0.2)))

    def test_empty_and_solid_slices_return_empty_meshes(self):
        for value in (0, 1):
            vertices, faces, normals, values = marching_cubes(np.full((4, 4, 4), value, dtype=np.uint8))
            self.assertEqual(vertices.shape, (0, 3))
            self.assertEqual(faces.shape, (0, 3))
            self.assertEqual(normals.shape, (0, 3))
            self.assertEqual(values.shape, (0,))

    def test_export_is_import_safe_and_uses_world_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'nested' / 'slice.stl'
            with patch.object(exporter, 'N', 16):
                exporter.main(['--output', str(output)])
            vertices = mesh.Mesh.from_file(str(output)).vectors.reshape(-1, 3)
            self.assertTrue(np.all(vertices >= -2.00001))
            self.assertTrue(np.all(vertices <= 2.00001))
            self.assertTrue(np.any(vertices < -0.5))
            self.assertTrue(np.all(np.isfinite(vertices)))

    def test_resolution_validation(self):
        for value in ('x', '7', '513'):
            with patch.dict(os.environ, {'NEBULA_RESOLUTION': value}):
                with self.assertRaisesRegex(ValueError, 'NEBULA_RESOLUTION'):
                    grid_size()
        with patch.dict(os.environ, {'NEBULA_RESOLUTION': '128'}):
            self.assertEqual(grid_size(), 128)

    def test_gpu_limits_requested_before_allocating_volume(self):
        adapter = Mock()
        adapter.limits = {'max-buffer-size': 1024**3, 'max-storage-buffer-binding-size': 1024**3}
        wgpu = Mock()
        wgpu.gpu.request_adapter_sync.return_value = adapter
        request_gpu_device(wgpu, 512**3 * 4)
        adapter.request_device_sync.assert_called_once_with(required_limits={
            'max-buffer-size': 512**3 * 4,
            'max-storage-buffer-binding-size': 512**3 * 4,
        })
        adapter.limits['max-storage-buffer-binding-size'] = 128 * 1024**2
        with self.assertRaisesRegex(RuntimeError, 'Lower NEBULA_RESOLUTION'):
            request_gpu_device(wgpu, 512**3 * 4)


@unittest.skipUnless(os.environ.get('NEBULA_TEST_GPU') == '1', 'Set NEBULA_TEST_GPU=1 for native WebGPU smoke tests')
class NativeGPUTests(unittest.TestCase):
    def test_hybrid_volume_matches_cpu_axes_and_slider(self):
        gpu = importlib.import_module('mandel_webgpu')
        fractal = gpu.WebGPUFractal()
        fractal.compute_volume(0.3, 0.1)
        actual = fractal.get_volume_numpy()
        expected = face.compute_volume(gpu.xs_np, gpu.ys_np, gpu.zs_np, 0.3, 0.1, gpu.MAX_ITER, 4)
        np.testing.assert_array_equal(actual, expected)

    def test_full_gpu_mesh_is_finite_and_reconfigures(self):
        gpu = importlib.import_module('mandel_full_gpu')
        fractal = gpu.FullGPUFractal((0, 2, 3))
        for slider_dim in (1, 0):
            fractal.set_slider_dim(slider_dim)
            vertices, faces, colors = fractal.compute({slider_dim: 0.2})
            self.assertGreater(len(faces), 0)
            self.assertTrue(np.all(np.isfinite(vertices)))
            self.assertEqual(colors.shape, vertices.shape)
            self.assertTrue(np.all(vertices >= -2.001))
            self.assertTrue(np.all(vertices <= 2.001))


if __name__ == '__main__':
    unittest.main()
