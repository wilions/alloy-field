"""
Kurz-Giovanola-Trivedi (KGT) Dendritic Solidification Kinetics.
Computes non-equilibrium dendrite tip velocity, constitutional undercooling, and arm spacings (lambda_1, lambda_2).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class DendriteTipState:
    """Calculated kinetic parameters of a moving dendritic solidification front."""
    undercooling_k: float
    tip_velocity_m_s: float
    primary_arm_spacing_um: float
    secondary_arm_spacing_um: float
    constitutional_undercooling_k: float
    kinetic_undercooling_k: float
    curvature_undercooling_k: float


class KGTDendriteKinetics:
    """Calculates non-equilibrium dendrite tip velocity and spacing scaling laws."""

    def __init__(
        self,
        gibbs_thomson_coeff_m_k: float = 2.4e-7,    # Gamma in m*K
        liquidus_slope_k_wt_pct: float = -3.5,     # m_L in K/wt%
        partition_coeff: float = 0.45,             # k_0 equilibrium partition coefficient
        liquid_diffusivity_m2_s: float = 3.0e-9,   # D_L in m^2/s
        kinetic_coeff_m_s_k: float = 0.05          # mu_k in m/(s*K)
    ):
        self.gamma = gibbs_thomson_coeff_m_k
        self.m_L = liquidus_slope_k_wt_pct
        self.k_0 = partition_coeff
        self.d_L = liquid_diffusivity_m2_s
        self.mu_k = kinetic_coeff_m_s_k

    def compute_tip_velocity(self, undercooling_k: float, a1: float = 1.2e-4, a2: float = 4.5e-6) -> float:
        """
        Calculates dendrite tip velocity v(Delta T) in m/s using KGT polynomial fit:
        v = a1 * (Delta T)^2 + a2 * (Delta T)^3.
        """
        dt = max(0.0, undercooling_k)
        return a1 * (dt ** 2) + a2 * (dt ** 3)

    def calculate_dendrite_spacings(
        self,
        thermal_gradient_k_m: float,
        solidification_velocity_m_s: float,
        solute_content_wt_pct: float = 5.0
    ) -> DendriteTipState:
        """
        Calculates primary (lambda_1) and secondary (lambda_2) dendrite arm spacing based on G and R.
        lambda_1 = 4.3 * (Delta T_0 * Gamma * D_L / k_0)^0.25 * G^(-0.5) * R^(-0.25)
        lambda_2 = B * (G * R)^(-1/3) = B * (cooling_rate)^(-1/3)
        """
        g = max(1.0, thermal_gradient_k_m)
        r = max(1e-6, solidification_velocity_m_s)
        cooling_rate = g * r

        # Equilibrium freezing range Delta T_0 = |m_L| * C_0 * (1 - k_0) / k_0
        delta_t_0 = abs(self.m_L) * solute_content_wt_pct * (1.0 - self.k_0) / max(0.01, self.k_0)

        # Primary arm spacing lambda_1 in um
        coeff_lambda_1 = 4.3 * ((delta_t_0 * self.gamma * self.d_L) / max(0.01, self.k_0)) ** 0.25
        lambda_1_m = coeff_lambda_1 * (g ** -0.5) * (r ** -0.25)
        lambda_1_um = lambda_1_m * 1e6

        # Secondary arm spacing lambda_2 in um (Bousquet / Kurz empirical relation)
        b_const = 45.0  # empirical scaling prefactor in um * (K/s)^(1/3)
        lambda_2_um = b_const * (cooling_rate ** -0.3333)

        # Constitutional undercooling: Delta T_c approx Delta T_0 * Ivantsov Peclet
        peclet = (r * 1e-6) / (2.0 * self.d_L)  # approx dimensionless Peclet
        delta_t_c = delta_t_0 * (peclet / (1.0 + peclet))
        delta_t_k = r / self.mu_k
        delta_t_r = (2.0 * self.gamma) / (1e-6)  # approx curvature undercooling
        total_undercooling = delta_t_c + delta_t_k + delta_t_r

        v_tip = self.compute_tip_velocity(total_undercooling)

        return DendriteTipState(
            undercooling_k=total_undercooling,
            tip_velocity_m_s=v_tip,
            primary_arm_spacing_um=lambda_1_um,
            secondary_arm_spacing_um=lambda_2_um,
            constitutional_undercooling_k=delta_t_c,
            kinetic_undercooling_k=delta_t_k,
            curvature_undercooling_k=delta_t_r
        )
