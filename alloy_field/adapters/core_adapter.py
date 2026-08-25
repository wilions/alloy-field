"""
Adapter bridging alloy-field simulation outputs with canonical alloy-core MicrostructureState contracts.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from alloy_core.schemas.microstructure import (
    MicrostructureState,
    GrainMorphology,
    PhaseConstituent,
    ComplexionState
)
from alloy_field.core.cellular_automata import CAMicrostructureResult
from alloy_field.core.cet_solver import CETPredictionResult
from alloy_field.core.texture import TextureAnalyzer


class FieldAdapter:
    """Converts alloy-field CA and CET simulation results to canonical MicrostructureState schemas."""

    @staticmethod
    def ca_result_to_microstructure_state(
        ca_result: CAMicrostructureResult,
        cet_result: Optional[CETPredictionResult] = None,
        base_phase: str = "FCC_Matrix"
    ) -> MicrostructureState:
        """Converts CA solidification result to canonical MicrostructureState."""
        j_index = TextureAnalyzer.calculate_texture_index(ca_result.euler_angles_deg)

        # Grain morphology descriptor
        aspect = max(1.0, float(ca_result.grain_aspect_ratio))
        morph_type = "columnar"
        if cet_result and cet_result.regime.value == "equiaxed":
            aspect = 1.1
            morph_type = "equiaxed"

        grains = GrainMorphology(
            mean_grain_size_um=float(ca_result.mean_grain_size_um),
            aspect_ratio=aspect,
            morphology_type=morph_type,
            grain_size_d10_um=float(ca_result.mean_grain_size_um * 0.6),
            grain_size_d90_um=float(ca_result.mean_grain_size_um * 1.5)
        )

        phases = {
            base_phase: PhaseConstituent(
                phase_name=base_phase,
                fraction=1.0
            )
        }

        return MicrostructureState(
            grains=grains,
            phases=phases,
            solidified_fraction=1.0
        )
