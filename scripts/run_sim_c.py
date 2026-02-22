#!/usr/bin/env python3
"""
Ejecuta sim_C headless con temporización Webster (sin optimización dinámica).

sim_C usa las mismas rutas que sim_A/sim_B pero con tiempos de semáforo
calculados por la fórmula de Webster (fixed-time óptimo teórico).

Requiere que generate_webster_timing.py haya creado sim_C/ primero.

Uso:
    uv run python scripts/run_sim_c.py [--sim-steps N] [--seed N]
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
SIM_C_DIR = PROJECT_ROOT / "sim_C"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta sim_C con temporización Webster headless"
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=3600,
        help="Pasos de simulación (default: 3600 = fin de la simulación)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla aleatoria SUMO (default: SUMO default)",
    )
    args = parser.parse_args()

    config_file = SIM_C_DIR / "simulation.sumocfg"
    if not config_file.exists():
        print(f"ERROR: {config_file} no encontrado.")
        print("Ejecuta primero: uv run python scripts/generate_webster_timing.py")
        return 1

    out_dir = SIM_C_DIR / "logs" / "sumo_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ejecutando sim_C (Webster fixed-time)...")
    print(f"  Config:  {config_file}")
    print(f"  Output:  {out_dir}")
    print(f"  Steps:   {args.sim_steps}")
    if args.seed is not None:
        print(f"  Seed:    {args.seed}")

    cmd = [
        "sumo",
        "-c", str(config_file),
        "--no-step-log", "true",
        "--time-to-teleport", "-1",
        "--end", str(args.sim_steps),
        "--tripinfo-output", str(out_dir / "tripinfo.xml"),
        "--summary-output", str(out_dir / "summary.xml"),
        "--fcd-output", str(out_dir / "fcd.xml"),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    print(f"\n$ {' '.join(cmd[:4])} ...")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nERROR: SUMO terminó con código {result.returncode}")
        return 1

    # Check output
    tripinfo = out_dir / "tripinfo.xml"
    if tripinfo.exists():
        # Count completed trips
        import xml.etree.ElementTree as ET
        count = sum(1 for _, elem in ET.iterparse(str(tripinfo))
                    if elem.tag == "tripinfo")
        print(f"\nsim_C completado: {count} viajes en {tripinfo}")
        if count < 10:
            print("WARNING: Muy pocos viajes completados — verifica sim_C/traffic_lights.add.xml")
    else:
        print("\nWARNING: tripinfo.xml no fue creado")

    print(f"\nSiguiente paso:")
    print(f"  uv run python scripts/compare_three_runs.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
