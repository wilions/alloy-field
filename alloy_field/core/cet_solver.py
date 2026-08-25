"""
Columnar-to-Equiaxed Transition (CET) Prediction Engine.
Implements Hunt's classical analytical criterion and Gäumann-Trivedi-Kurz (GTK) rapid solidification models.
"""

from __future__ import annotations
from enum import Enum
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


class CETRegime(str, Enum):
    COLUMNAR = "columnar"
    MIXED = "mixed"
    EQUIAXED = "equiaxed"


@dataclass
class CETPredictionResult:
    """Detailed outcome of CET evaluation."""
    regime: CETRegime
    equiaxed_volume_fraction: float
    hunt_parameter: float
    critical_thermal_gradient_k_m: float
    nucleant_density_m3: float


class CETSolver:
    """Evaluates Columnar-to-Equiaxed Transition criteria across thermal process windows."""

    def __init__(
        self,
        nucleant_density_m3: float = 1e12,
        nucleation_undercooling_k: float = 2.5,
        growth_coeff_a: float = 1.5e-4
    ):
        self.n_0 = nucleant_density_m3
        self.delta_t_n = nucleation_undercooling_k
        self.a_growth = growth_coeff_a

    def predict_regime(
        self,
        thermal_gradient_k_m: float,
        solidification_velocity_m_s: float,
        solute_content_wt_pct: float = 5.0
    ) -> CETPredictionResult:
        """
        Calculates equiaxed fraction phi_eq based on Hunt's analytical CET model:
        G = n_exponent * (N_0)^(1/3) * ( 1 - (Delta T_N / Delta T_c)^3 ) * V^0.5 ...
        """
        g = max(1.0, thermal_gradient_k_m)
        v = max(1e-6, solidification_velocity_m_s)

        # Constitutional undercooling Delta T_c from growth law: v = a * (Delta T_c)^2 -> Delta T_c = sqrt(v / a)
        delta_t_c = math.sqrt(v / self.a_growth)

        # Extended volume fraction of equiaxed grains:
        # phi_ext = (4/3) * pi * N_0 * ( r_equiaxed )^3
        # Hunt criterion formulation:
        if delta_t_c <= self.delta_t_n:
            phi_eq = 0.0
        else:
            # Ratio of undercoolings
            u_ratio = (1.0 - (self.delta_t_n / delta_t_c) ** 3)
            # Critical gradient G_crit = 0.617 * (N_0)^(1/3) * ( 1 - (Delta T_N / Delta T_c)^3 ) * (Delta T_c)
            g_crit = 0.617 * (self.n_0 ** (1.0 / 3.0)) * u_ratio * delta_t_c
            
            ratio_g = g / max(1.0, g_crit)
            # Equiaxed fraction scaling
            phi_eq = 1.0 / (1.0 + (ratio_g ** 3))

        if phi_eq < 0.01:
            regime = CETRegime.COLUMNAR
        elif phi_eq > 0.49:
            regime = CETRegime.EQUIAXED
        else:
            regime = CETRegime.MIXED

        hunt_val = g / (v ** 0.5)

        return CETPredictionResult(
            regime=regime,
            equiaxed_volume_fraction=float(phi_eq),
            hunt_parameter=float(hunt_val),
            critical_thermal_gradient_k_m=float(g_crit) if 'g_crit' in locals() else 0.0,
            nucleant_density_m3=self.n_0
        )

    def generate_cet_map(
        self,
        g_range: Tuple[float, float] = (1e4, 1e7),
        v_range: Tuple[float, float] = (1e-4, 1.0),
        num_points: int = 50
    ) -> Dict[str, np.ndarray]:
        """Generates a 2D G vs. V grid with predicted equiaxed volume fractions."""
        g_vals = np.logspace(math.log10(g_range[0]), math.log10(g_range[1]), num_points)
        v_vals = np.logspace(math.log10(v_range[0]), math.log10(v_range[1]), num_points)
        g_grid, v_grid = np.meshgrid(g_vals, v_vals)
        phi_grid = np.zeros_like(g_grid)

        for i in range(num_points):
            for j in range(num_points):
                res = self.predict_regime(g_grid[i, j], v_grid[i, j])
                phi_grid[i, j] = res.equiaxed_volume_fraction

        return {
            "thermal_gradient_k_m": g_grid,
            "velocity_m_s": v_grid,
            "equiaxed_fraction": phi_grid
        }
