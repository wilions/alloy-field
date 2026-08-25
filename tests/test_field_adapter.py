"""
Unit tests for FieldAdapter and canonical alloy-core MicrostructureState integration.
"""

import pytest
from alloy_field.core.cellular_automata import CellularAutomataSolidificationSolver
from alloy_field.core.cet_solver import CETSolver
from alloy_field.adapters.core_adapter import FieldAdapter
from alloy_core.schemas.microstructure import MicrostructureState


def test_field_adapter_to_microstructure_state():
    ca = CellularAutomataSolidificationSolver(nx=40, ny=40, dx_um=1.0, seed=42)
    ca_res = ca.simulate(total_time_s=0.0004)

    cet_solver = CETSolver()
    cet_res = cet_solver.predict_regime(thermal_gradient_k_m=1e6, solidification_velocity_m_s=0.05)

    micro_state = FieldAdapter.ca_result_to_microstructure_state(
        ca_result=ca_res,
        cet_result=cet_res,
        base_phase="BCC_Matrix"
    )

    assert isinstance(micro_state, MicrostructureState)
    assert micro_state.grains.mean_grain_size_um == ca_res.mean_grain_size_um
    assert "BCC_Matrix" in micro_state.phases
    assert micro_state.grains.morphology_type == "columnar"
