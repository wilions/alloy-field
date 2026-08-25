"""
Unit tests for KGT dendritic kinetics and arm spacing scaling laws.
"""

import pytest
from alloy_field.core.kgt_kinetics import KGTDendriteKinetics, DendriteTipState


def test_kgt_velocity_monotonicity():
    kgt = KGTDendriteKinetics()
    v1 = kgt.compute_tip_velocity(undercooling_k=2.0)
    v2 = kgt.compute_tip_velocity(undercooling_k=5.0)
    v3 = kgt.compute_tip_velocity(undercooling_k=10.0)

    assert 0.0 < v1 < v2 < v3


def test_dendrite_arm_spacings():
    kgt = KGTDendriteKinetics()
    
    # LPBF high cooling rate conditions: G = 1e6 K/m, R = 0.05 m/s (cooling rate = 5e4 K/s)
    res_fast = kgt.calculate_dendrite_spacings(
        thermal_gradient_k_m=1e6,
        solidification_velocity_m_s=0.05,
        solute_content_wt_pct=4.0
    )

    assert isinstance(res_fast, DendriteTipState)
    assert 0.5 < res_fast.primary_arm_spacing_um < 20.0
    assert 0.1 < res_fast.secondary_arm_spacing_um < 10.0

    # Slow casting conditions: G = 1e3 K/m, R = 1e-4 m/s (cooling rate = 0.1 K/s)
    res_slow = kgt.calculate_dendrite_spacings(
        thermal_gradient_k_m=1e3,
        solidification_velocity_m_s=1e-4,
        solute_content_wt_pct=4.0
    )

    # Casting dendrites should be 1-2 orders of magnitude coarser than LPBF
    assert res_slow.primary_arm_spacing_um > res_fast.primary_arm_spacing_um * 5.0
    assert res_slow.secondary_arm_spacing_um > res_fast.secondary_arm_spacing_um * 5.0
