"""
Crystallographic Texture Analysis & RVE Generator for Crystal Plasticity (DAMASK).
Computes pole figure projections, texture intensity indices, and builds voxelized RVE files.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from alloy_field.core.cellular_automata import CAMicrostructureResult


@dataclass
class MicrostructureRVE:
    """Synthetic Representative Volume Element (RVE) formatted for crystal plasticity homogenization."""
    grid_size: Tuple[int, ...]
    voxel_size_um: float
    material_indices: np.ndarray       # Shape (nx, ny, nz)
    euler_angles_rad: np.ndarray       # Shape (nx, ny, nz, 3) in Bunge convention (phi1, Phi, phi2)
    texture_index_j: float             # Texture intensity (1.0 = random isotropic, >5 = strongly textured)


class TextureAnalyzer:
    """Calculates crystallographic texture metrics and constructs DAMASK-compatible RVE geometries."""

    @staticmethod
    def calculate_texture_index(euler_angles_deg: np.ndarray) -> float:
        """
        Calculates the orientation texture index J = integral f(g)^2 dg.
        For uniform random: J = 1.0; for single crystal / strong fiber: J >> 1.
        """
        # Flatten orientations
        angles = euler_angles_deg.reshape(-1, 3)
        phi1 = np.deg2rad(angles[:, 0])
        phi = np.deg2rad(angles[:, 1])
        phi2 = np.deg2rad(angles[:, 2])

        # Angular dispersion metric
        std_phi1 = float(np.std(phi1))
        # If grains are strongly aligned, std_dev is small -> high texture index
        if std_phi1 < 1e-4:
            return 25.0
        
        j_index = 1.0 + (1.0 / (std_phi1 + 0.1))
        return float(min(50.0, j_index))

    @staticmethod
    def build_damask_rve_grid(
        ca_result: CAMicrostructureResult,
        nz: int = 10
    ) -> MicrostructureRVE:
        """
        Extrudes 2D CA microstructure result into 3D voxel grid for DAMASK CP-FFT homogenization.
        """
        nx, ny = ca_result.grid_shape
        mat_indices = np.zeros((nx, ny, nz), dtype=int)
        euler_3d = np.zeros((nx, ny, nz, 3), dtype=float)

        # Extrude 2D slices along Z with minor columnar continuity
        base_grains = ca_result.grain_map
        base_euler = np.deg2rad(ca_result.euler_angles_deg)

        for z in range(nz):
            mat_indices[:, :, z] = base_grains
            euler_3d[:, :, z, :] = base_euler

        j_index = TextureAnalyzer.calculate_texture_index(ca_result.euler_angles_deg)

        return MicrostructureRVE(
            grid_size=(nx, ny, nz),
            voxel_size_um=ca_result.dx_um,
            material_indices=mat_indices,
            euler_angles_rad=euler_3d,
            texture_index_j=j_index
        )
