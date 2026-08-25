"""
Unit tests for FieldMCPServer tool dispatching.
"""

import pytest
from alloy_field.mcp_server import FieldMCPServer


def test_field_mcp_spacing():
    server = FieldMCPServer()
    res = server.dispatch("field_calculate_dendrite_spacing", {
        "thermal_gradient_k_m": 1e6,
        "solidification_velocity_m_s": 0.05,
        "solute_content_wt_pct": 4.5
    })
    assert res["primary_arm_spacing_lambda1_um"] > 0.0
    assert res["secondary_arm_spacing_lambda2_um"] > 0.0


def test_field_mcp_cet():
    server = FieldMCPServer()
    res = server.dispatch("field_predict_cet", {
        "thermal_gradient_k_m": 5e5,
        "solidification_velocity_m_s": 0.02
    })
    assert "predicted_regime" in res
    assert "equiaxed_volume_fraction" in res


def test_field_mcp_ca():
    server = FieldMCPServer()
    res = server.dispatch("field_simulate_microstructure_ca", {
        "nx": 30,
        "ny": 30,
        "dx_um": 1.0,
        "cooling_rate_k_s": 1e5
    })
    assert res["num_grains"] > 0
    assert "texture_intensity_index_j" in res
