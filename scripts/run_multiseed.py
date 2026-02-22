#!/usr/bin/env python3
"""
Experimento multi-seed: ejecuta sim_A (fixed-time) y sim_B (IA) con múltiples seeds.

Para cada seed, corre ambas simulaciones headless usando SUMO directamente
y guarda tripinfo.xml + summary.xml en:
    results/seed_{N}/sim_A/tripinfo.xml
    results/seed_{N}/sim_B/tripinfo.xml

Uso:
    uv run python scripts/run_multiseed.py --seeds 42 123 456 789 1337 --sim-steps 2000 --out results/

Notas:
    - sim_A: fixed-time (sin optimización dinámica)
    - sim_B: con optimización IA (requiere traffic-control corriendo vía Docker para efecto real;
      sin el servicio, actúa como fixed-time con las mismas rutas)
    - Las rutas están prefijadas en routes.rou.xml; el seed controla el comportamiento
      estocástico de los conductores (gap acceptance, cambio de carril, sigma=0.5).
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._sim_utils import require_sim_dir  # noqa: E402


def run_sumo_headless(
    config_file: Path,
    seed: int,
    out_dir: Path,
    sim_steps: int,
    enable_dynamic_optimization: bool = False,
) -> bool:
    """
    Ejecuta SUMO headless con un seed dado.

    Para sim_A (fixed-time) se usa sumo directamente.
    Para sim_B (IA) se usa TraCI via SimulationOrchestrator si enable_dynamic_optimization=True,
    pero como el servicio de traffic-control puede no estar disponible,
    también se ofrece el modo SUMO directo como fallback.

    Returns True si la ejecución fue exitosa.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if enable_dynamic_optimization:
        # Run via SimulationOrchestrator to enable dynamic TL optimization
        return _run_via_orchestrator(config_file, seed, out_dir, sim_steps)
    else:
        return _run_via_subprocess(config_file, seed, out_dir, sim_steps)


def _run_via_subprocess(
    config_file: Path,
    seed: int,
    out_dir: Path,
    sim_steps: int,
) -> bool:
    """Corre SUMO como subprocess directo (sin TraCI). Más rápido, sin optimización."""
    cmd = [
        "sumo",
        "-c", str(config_file),
        "--seed", str(seed),
        "--no-step-log", "true",
        "--time-to-teleport", "-1",
        "--end", str(sim_steps),
        "--tripinfo-output", str(out_dir / "tripinfo.xml"),
        "--summary-output", str(out_dir / "summary.xml"),
    ]
    print(f"    $ {' '.join(cmd[:6])} ... --end {sim_steps}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(config_file.parent))
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-300:]}")
        return False
    return True


def _run_via_orchestrator(
    config_file: Path,
    seed: int,
    out_dir: Path,
    sim_steps: int,
) -> bool:
    """Corre la simulación via SimulationOrchestrator (con TraCI y optimización dinámica)."""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from simulation_orchestrator import SimulationOrchestrator
    except ImportError as e:
        print(f"    ERROR importando SimulationOrchestrator: {e}")
        return False

    # Create a symlinked temp dir pointing to the sim files, with custom output
    import tempfile, os, shutil

    with tempfile.TemporaryDirectory(prefix="multiseed_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        sim_dir = config_file.parent

        # Symlink all simulation files into tmpdir
        for item in sim_dir.iterdir():
            if item.name in ("logs",):
                continue
            target = tmpdir_path / item.name
            try:
                os.symlink(item.resolve(), target)
            except Exception:
                shutil.copy2(str(item), str(target))

        # Ensure logs/sumo_output exists and points to out_dir
        logs_dir = tmpdir_path / "logs" / "sumo_output"
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Create symlinks for output files so orchestrator writes to out_dir
        for fname in ("tripinfo.xml", "summary.xml", "fcd.xml"):
            link = logs_dir / fname
            out_file = out_dir / fname
            try:
                os.symlink(out_file.resolve(), link)
            except Exception:
                pass

        try:
            orchestrator = SimulationOrchestrator(
                simulation_dir=str(tmpdir_path),
                sim_steps=sim_steps,
                enable_dynamic_optimization=True,
                seed=seed,
            )
            if not orchestrator.setup_simulation():
                print("    ERROR: setup_simulation() failed")
                return False
            orchestrator.run_simulation()
            return True
        except Exception as e:
            print(f"    ERROR en orchestrator: {e}")
            return False
        finally:
            try:
                import traci
                traci.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experimento multi-seed: ejecuta sim_A y sim_B con múltiples seeds SUMO"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456, 789, 1337],
        metavar="N",
        help="Seeds a ejecutar (default: 42 123 456 789 1337)",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=2000,
        metavar="N",
        help="Pasos de simulación por run (default: 2000)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results",
        metavar="DIR",
        help="Directorio de salida base (default: results/)",
    )
    parser.add_argument(
        "--dynamic-optimization",
        action="store_true",
        help="Habilitar optimización dinámica en sim_B (requiere traffic-control corriendo)",
    )
    args = parser.parse_args()

    sim_a_dir = require_sim_dir("A")
    sim_b_dir = require_sim_dir("B")
    sim_a_cfg = sim_a_dir / "simulation.sumocfg"
    sim_b_cfg = sim_b_dir / "simulation.sumocfg"

    if not sim_a_cfg.exists():
        print(f"ERROR: No se encontró {sim_a_cfg}")
        return 1
    if not sim_b_cfg.exists():
        print(f"ERROR: No se encontró {sim_b_cfg}")
        return 1

    print(f"Multi-seed experiment")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Sim steps:  {args.sim_steps}")
    print(f"  Output dir: {args.out}")
    print(f"  Dynamic opt (sim_B): {args.dynamic_optimization}")
    print()

    successes = 0
    total = len(args.seeds) * 2

    for seed in args.seeds:
        print(f"=== Seed {seed} ===")

        # sim_A: always fixed-time (subprocess)
        out_a = args.out / f"seed_{seed}" / "sim_A"
        print(f"  sim_A → {out_a}")
        ok_a = run_sumo_headless(
            sim_a_cfg, seed, out_a, args.sim_steps,
            enable_dynamic_optimization=False,
        )
        if ok_a:
            successes += 1
            print(f"  sim_A OK — {out_a / 'tripinfo.xml'}")
        else:
            print(f"  sim_A FAILED")

        # sim_B: with or without dynamic optimization
        out_b = args.out / f"seed_{seed}" / "sim_B"
        print(f"  sim_B → {out_b}")
        ok_b = run_sumo_headless(
            sim_b_cfg, seed, out_b, args.sim_steps,
            enable_dynamic_optimization=args.dynamic_optimization,
        )
        if ok_b:
            successes += 1
            print(f"  sim_B OK — {out_b / 'tripinfo.xml'}")
        else:
            print(f"  sim_B FAILED")

        print()

    print(f"Completado: {successes}/{total} runs exitosos")
    print(f"Resultados en: {args.out}")
    print()
    print("Siguiente paso:")
    print(f"  uv run python scripts/aggregate_seeds.py --results-dir {args.out}")

    return 0 if successes == total else 1


if __name__ == "__main__":
    sys.exit(main())
