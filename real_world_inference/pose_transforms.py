from __future__ import annotations

"""Pose representation conversions shared by the Unitree wire adapter.

Rot6D is represented by the first two *columns* of a rotation matrix, matching
the representation used by scripts/convert_mobile_transfer_to_lingbot_v3.py.
Quaternions use scipy/LingBot ordering: ``xyzw``.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def _as_rows(values: object, width: int, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    was_vector = array.ndim == 1
    if was_vector:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} expected shape ({width},) or (N,{width}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array, was_vector


def rot6d_to_matrix(rot6d: object) -> np.ndarray:
    rows, was_vector = _as_rows(rot6d, 6, "Rot6D")
    first = rows[:, :3].copy()
    second = rows[:, 3:6].copy()
    first_norm = np.linalg.norm(first, axis=1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("Rot6D first column contains a near-zero vector")
    first /= first_norm
    second -= np.sum(first * second, axis=1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("Rot6D columns are degenerate")
    second /= second_norm
    third = np.cross(first, second)
    matrices = np.stack((first, second, third), axis=2).astype(np.float32)
    return matrices[0] if was_vector else matrices


def matrix_to_rot6d(matrix: object) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    was_matrix = array.ndim == 2
    if was_matrix:
        array = array.reshape(1, 3, 3)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(f"rotation matrix expected shape (3,3) or (N,3,3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("rotation matrix contains NaN or infinity")
    rot6d = np.concatenate((array[:, :, 0], array[:, :, 1]), axis=1).astype(np.float32)
    return rot6d[0] if was_matrix else rot6d


def xyzw_to_matrix(quaternion: object) -> np.ndarray:
    rows, was_vector = _as_rows(quaternion, 4, "xyzw quaternion")
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("xyzw quaternion contains a near-zero quaternion")
    q = rows / norm
    x, y, z, w = q.T
    matrices = np.empty((len(q), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    matrices = matrices.astype(np.float32)
    return matrices[0] if was_vector else matrices


def matrix_to_xyzw(matrix: object) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    was_matrix = array.ndim == 2
    if was_matrix:
        array = array.reshape(1, 3, 3)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(f"rotation matrix expected shape (3,3) or (N,3,3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("rotation matrix contains NaN or infinity")

    output = Rotation.from_matrix(array).as_quat()
    output /= np.linalg.norm(output, axis=1, keepdims=True)
    output[output[:, 3] < 0.0] *= -1.0
    output = output.astype(np.float32)
    return output[0] if was_matrix else output


def rot6d_to_xyzw(rot6d: object) -> np.ndarray:
    return matrix_to_xyzw(rot6d_to_matrix(rot6d))


def xyzw_to_rot6d(quaternion: object) -> np.ndarray:
    return matrix_to_rot6d(xyzw_to_matrix(quaternion))
