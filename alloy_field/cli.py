"""
Command-line interface for alloy-field.
Provides fast evaluation of dendrite arm spacing, CET regimes, and 2D Cellular Automata solidification runs.
"""

import argparse
import json
import sys
from alloy_field.core.kgt_kinetics import KGTDendriteKinetics
from alloy_field.core.cet_solver import CETSolver
from alloy_field.core.cellular_automata import CellularAutomataSolidificationSolver
from alloy_field.core.texture import TextureAnalyzer


def main():
    parser = argparse.ArgumentParser(description="AlloyField: Solidification & Microstructure Cellular Automata Engine")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Arm Spacing Subcommand
    arm_parser = subparsers.add_parser("spacing", help="Calculate primary (lambda_1) and secondary (lambda_2) dendrite spacing")
    arm_parser.add_argument("--g", type=float, default=1e6, help="Thermal gradient G in K/m (default: 1e6)")
    arm_parser.add_argument("--r", type=float, default=0.05, help="Solidification velocity R in m/s (default: 0.05)")
    arm_parser.add_argument("--solute", type=float, default=5.0, help="Solute content in wt% (default: 5.0)")

    # 2. CET Prediction Subcommand
    cet_parser = subparsers.add_parser("cet", help="Predict Columnar-to-Equiaxed Transition (CET) regime")
    cet_parser.add_argument("--g", type=float, default=5e5, help="Thermal gradient G in K/m")
    cet_parser.add_argument("--v", type=float, default=0.01, help="Solidification velocity V in m/s")
    cet_parser.add_argument("--n0", type=float, default=1e12, help="Nucleant density in 1/m^3 (default: 1e12)")

    # 3. Cellular Automata Run
    ca_parser = subparsers.add_parser("simulate-ca", help="Run 2D Cellular Automata grain evolution simulation")
    ca_parser.add_argument("--nx", type=int, default=80, help="Grid size X (default: 80)")
    ca_parser.add_argument("--ny", type=int, default=80, help="Grid size Y (default: 80)")
    ca_parser.add_argument("--dx", type=float, default=1.0, help="Grid resolution in um (default: 1.0)")
    ca_parser.add_argument("--cooling-rate", type=float, default=1e5, help="Cooling rate in K/s (default: 1e5)")

    args = parser.parse_args()

    if args.command == "spacing":
        kgt = KGTDendriteKinetics()
        res = kgt.calculate_dendrite_spacings(
            thermal_gradient_k_m=args.g,
            solidification_velocity_m_s=args.r,
            solute_content_wt_pct=args.solute
        )
        print("=== KGT Dendrite Tip Kinetics & Spacing Result ===")
        print(f"Thermal Gradient (G):        {args.g:.2e} K/m")
        print(f"Solidification Velocity (R): {args.r:.4f} m/s")
        print(f"Cooling Rate (G*R):          {args.g * args.r:.2e} K/s")
        print(f"Total Undercooling:          {res.undercooling_k:.2f} K")
        print(f"Dendrite Tip Velocity:       {res.tip_velocity_m_s:.4f} m/s")
        print(f"Primary Arm Spacing (λ₁):    {res.primary_arm_spacing_um:.2f} μm")
        print(f"Secondary Arm Spacing (λ₂):  {res.secondary_arm_spacing_um:.2f} μm")

    elif args.command == "cet":
        solver = CETSolver(nucleant_density_m3=args.n0)
        res = solver.predict_regime(
            thermal_gradient_k_m=args.g,
            solidification_velocity_m_s=args.v
        )
        print("=== Columnar-to-Equiaxed Transition (CET) Result ===")
        print(f"Thermal Gradient (G):        {args.g:.2e} K/m")
        print(f"Solidification Velocity (V): {args.v:.4f} m/s")
        print(f"Predicted CET Regime:        {res.regime.value.upper()}")
        print(f"Equiaxed Volume Fraction:    {res.equiaxed_volume_fraction * 100.0:.2f}%")
        print(f"Hunt Parameter (G / V^0.5):  {res.hunt_parameter:.2e} K·s^0.5/m^1.5")

    elif args.command == "simulate-ca":
        ca = CellularAutomataSolidificationSolver(nx=args.nx, ny=args.ny, dx_um=args.dx)
        res = ca.simulate(cooling_rate_k_s=args.cooling_rate)
        j_index = TextureAnalyzer.calculate_texture_index(res.euler_angles_deg)
        print("=== Cellular Automata Simulation Result ===")
        print(f"Grid Dimensions:             {args.nx} x {args.ny} voxels ({args.nx * args.dx} x {args.ny * args.dx} μm)")
        print(f"Total Grains Formed:         {res.num_grains}")
        print(f"Mean Equivalent Grain Size:  {res.mean_grain_size_um:.2f} μm")
        print(f"Texture Intensity Index (J): {j_index:.2f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
