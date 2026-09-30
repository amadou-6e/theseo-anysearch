"""Shared rigid-body rotation math for independent world-source adapters."""

from __future__ import annotations

import math

import numpy as np


def rotation_from_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    """Return a float64 rotation matrix for roll, pitch, and yaw in radians."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
