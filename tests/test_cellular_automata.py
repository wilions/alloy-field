"""
Unit tests for 2D Cellular Automata solidification solver.
"""

import pytest
import numpy as np
from alloy_field.core.cellular_automata import CellularAutomataSolidificationSolver, CAMicrostructureResult


def test_cellular_automata_simulation():
    ca = CellularAutomataSolidificationSolver(nx=50, ny=50, dx_um=1.0, seed=42)
    res = ca.simulate(
        thermal_gradient_k_m=1e6,
        cooling_rate_k_s=1e5,
        total_time_s=0.0005
    )

    assert isinstance(res, CAMicrostructureResult)
    assert res.grid_shape == (50, 50)
    assert res.num_grains > 1
    assert res.mean_grain_size_um > 0.0
    assert np.all(res.grain_map > 0)
    assert res.euler_angles_deg.shape == (50, 50, 3)
