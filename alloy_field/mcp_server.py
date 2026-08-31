"""
Official Model Context Protocol (MCP) Server for Solidification & Microstructure Field Engine.
Exposes tools for KGT dendrite tip kinetics, Hunt Columnar-to-Equiaxed (CET) transition,
and 2D Cellular Automata grain evolution with DAMASK texture indexing.
"""

import sys
import json
from typing import Dict, Any, List, Optional

from alloy_field.core.kgt_kinetics import KGTDendriteKinetics
from alloy_field.core.cet_solver import CETSolver
from alloy_field.core.cellular_automata import CellularAutomataSolidificationSolver
from alloy_field.core.texture import TextureAnalyzer


class FieldMCPServer:
    """Standard MCP Dispatcher for alloy-field physics tools."""

    def __init__(self, default_nucleant_density: float = 1e12):
        self.default_nucleant_density = default_nucleant_density

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an MCP tool call to the corresponding physics engine."""
        if tool_name == "field_calculate_dendrite_spacing":
            return self.calculate_dendrite_spacing(arguments)
        elif tool_name == "field_predict_cet":
            return self.predict_cet(arguments)
        elif tool_name == "field_simulate_microstructure_ca":
            return self.simulate_microstructure_ca(arguments)
        else:
            raise ValueError(f"Unknown tool name: {tool_name}")

    def calculate_dendrite_spacing(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        g = float(arguments.get("thermal_gradient_k_m", 1e6))
        r = float(arguments.get("solidification_velocity_m_s", 0.05))
        solute = float(arguments.get("solute_content_wt_pct", 5.0))

        kgt = KGTDendriteKinetics()
        res = kgt.calculate_dendrite_spacings(
            thermal_gradient_k_m=g,
            solidification_velocity_m_s=r,
            solute_content_wt_pct=solute
        )

        return {
            "thermal_gradient_k_m": g,
            "solidification_velocity_m_s": r,
            "cooling_rate_k_s": g * r,
            "undercooling_k": res.undercooling_k,
            "dendrite_tip_velocity_m_s": res.tip_velocity_m_s,
            "primary_arm_spacing_lambda1_um": res.primary_arm_spacing_um,
            "secondary_arm_spacing_lambda2_um": res.secondary_arm_spacing_um,
            "constitutional_undercooling_k": res.constitutional_undercooling_k
        }

    def predict_cet(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        g = float(arguments.get("thermal_gradient_k_m", 5e5))
        v = float(arguments.get("solidification_velocity_m_s", 0.01))
        n0 = float(arguments.get("nucleant_density_m3", self.default_nucleant_density))

        solver = CETSolver(nucleant_density_m3=n0)
        res = solver.predict_regime(
            thermal_gradient_k_m=g,
            solidification_velocity_m_s=v
        )

        return {
            "thermal_gradient_k_m": g,
            "solidification_velocity_m_s": v,
            "predicted_regime": res.regime.value,
            "equiaxed_volume_fraction": res.equiaxed_volume_fraction,
            "hunt_parameter": res.hunt_parameter,
            "critical_thermal_gradient_k_m": res.critical_thermal_gradient_k_m
        }

    def simulate_microstructure_ca(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        nx = int(arguments.get("nx", 50))
        ny = int(arguments.get("ny", 50))
        dx_um = float(arguments.get("dx_um", 1.0))
        cooling_rate = float(arguments.get("cooling_rate_k_s", 1e5))
        g = float(arguments.get("thermal_gradient_k_m", 1e6))

        ca = CellularAutomataSolidificationSolver(nx=nx, ny=ny, dx_um=dx_um)
        ca_res = ca.simulate(thermal_gradient_k_m=g, cooling_rate_k_s=cooling_rate)
        j_index = TextureAnalyzer.calculate_texture_index(ca_res.euler_angles_deg)
        morphology_type = "equiaxed" if ca_res.grain_aspect_ratio < 1.5 else "columnar"

        return {
            "grid_shape": [nx, ny],
            "dx_um": dx_um,
            "num_grains": ca_res.num_grains,
            "mean_grain_size_um": ca_res.mean_grain_size_um,
            "grain_aspect_ratio": ca_res.grain_aspect_ratio,
            "texture_intensity_index_j": j_index,
            "morphology_type": morphology_type
        }


def main():
    server = FieldMCPServer()
    if len(sys.argv) > 1 and sys.argv[1] == "--call":
        tool_name = sys.argv[2]
        args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        try:
            res = server.dispatch(tool_name, args)
            print(json.dumps(res, indent=2))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "field_calculate_dendrite_spacing",
                                "description": "Compute KGT dendrite tip velocity and arm spacings (lambda_1, lambda_2).",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "thermal_gradient_k_m": {"type": "number"},
                                        "solidification_velocity_m_s": {"type": "number"},
                                        "solute_content_wt_pct": {"type": "number"}
                                    },
                                    "required": ["thermal_gradient_k_m", "solidification_velocity_m_s"]
                                }
                            },
                            {
                                "name": "field_predict_cet",
                                "description": "Evaluate Hunt's Columnar-to-Equiaxed Transition (CET) criterion.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "thermal_gradient_k_m": {"type": "number"},
                                        "solidification_velocity_m_s": {"type": "number"}
                                    },
                                    "required": ["thermal_gradient_k_m", "solidification_velocity_m_s"]
                                }
                            },
                            {
                                "name": "field_simulate_microstructure_ca",
                                "description": "Run 2D Cellular Automata grain evolution simulation and compute texture index.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "nx": {"type": "integer"},
                                        "ny": {"type": "integer"},
                                        "dx_um": {"type": "number"},
                                        "cooling_rate_k_s": {"type": "number"}
                                    }
                                }
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                args = params.get("arguments", {})
                result = server.dispatch(tool_name, args)
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}}
            else:
                response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": f"Method not supported: {method}"}}

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
