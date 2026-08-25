"""
Unit tests for texture analysis and DAMASK RVE generator.
"""

import pytest
import numpy as np
from alloy_field.core.cellular_automata import CellularAutomataSolidificationSolver
from alloy_field.core.texture import TextureAnalyzer, MicrostructureRVE


def test_texture_index_calculation():
    # Random orientations -> J ~ 1-5
    random_angles = np.random.uniform(-45.0, 45.0, size=(40, 40, 3))
    j_rand = TextureAnalyzer.calculate_texture_index(random_angles)
    assert 1.0 <= j_rand <= 10.0

    # Perfect single crystal alignment -> J >> 10
    single_crystal = np.zeros((40, 40, 3))
    j_single = TextureAnalyzer.calculate_texture_index(single_crystal)
    assert j_single > 20.0


def test_damask_rve_generation():
    ca = CellularAutomataSolidificationSolver(nx=30, ny=30, dx_um=1.5, seed=123)
    ca_res = ca.simulate(total_time_s=0.0003)
    
    rve = TextureAnalyzer.build_damask_rve_grid(ca_res, nz=8)
    assert isinstance(rve, MicrostructureRVE)
    assert rve.grid_size == (30, 30, 8)
    assert rve.material_indices.shape == (30, 30, 8)
    assert rve.euler_angles_rad.shape == (30, 30, 8, 3)
    assert rve.texture_index_j > 0.0
