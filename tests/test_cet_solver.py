"""
Unit tests for Columnar-to-Equiaxed Transition (CET) solver.
"""

import pytest
from alloy_field.core.cet_solver import CETSolver, CETRegime, CETPredictionResult


def test_cet_columnar_and_equiaxed_regimes():
    solver = CETSolver(nucleant_density_m3=1e12, nucleation_undercooling_k=2.0)

    # Steep thermal gradient + low velocity -> Columnar
    res_col = solver.predict_regime(
        thermal_gradient_k_m=5e6,
        solidification_velocity_m_s=0.001
    )
    assert res_col.regime == CETRegime.COLUMNAR
    assert res_col.equiaxed_volume_fraction < 0.10

    # Low thermal gradient + high velocity -> Equiaxed
    res_eq = solver.predict_regime(
        thermal_gradient_k_m=1e4,
        solidification_velocity_m_s=0.1
    )
    assert res_eq.regime == CETRegime.EQUIAXED
    assert res_eq.equiaxed_volume_fraction > 0.49


def test_cet_map_generation():
    solver = CETSolver()
    cet_map = solver.generate_cet_map(num_points=10)
    assert cet_map["thermal_gradient_k_m"].shape == (10, 10)
    assert cet_map["equiaxed_fraction"].shape == (10, 10)
