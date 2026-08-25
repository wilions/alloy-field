"""
AlloyField: Solidification Microstructure, Cellular Automata, and Texture Generation Engine.
"""

from alloy_field.core.kgt_kinetics import KGTDendriteKinetics, DendriteTipState
from alloy_field.core.cellular_automata import CellularAutomataSolidificationSolver, CAMicrostructureResult
from alloy_field.core.cet_solver import CETSolver, CETRegime
from alloy_field.core.texture import TextureAnalyzer, MicrostructureRVE

__version__ = "0.1.0"
__all__ = [
    "KGTDendriteKinetics",
    "DendriteTipState",
    "CellularAutomataSolidificationSolver",
    "CAMicrostructureResult",
    "CETSolver",
    "CETRegime",
    "TextureAnalyzer",
    "MicrostructureRVE"
]
