#!/usr/bin/env python3
"""
Script principal para ejecutar la simulación de tráfico
Acepta un archivo ZIP con los archivos de simulación SUMO
"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Agregar el directorio raíz al path para importaciones
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger


def extract_simulation_zip(zip_path: str, extract_dir: str | None = None) -> str:
    """
    Extrae un archivo ZIP con archivos de simulación SUMO

    Args:
        zip_path: Ruta al archivo ZIP
        extract_dir: Directorio donde extraer (opcional)

    Returns:
        Ruta al directorio extraído

    Raises:
        ValueError: Si el ZIP no contiene los archivos requeridos
    """
    zip_file = Path(zip_path)

    if not zip_file.exists():
        raise FileNotFoundError(f"Archivo ZIP no encontrado: {zip_path}")

    if not zip_file.suffix.lower() == '.zip':
        raise ValueError(f"El archivo debe ser un ZIP: {zip_path}")

    if extract_dir is None:
        extract_dir = tempfile.mkdtemp(prefix="traffic_sim_")
    else:
        extract_dir_path = Path(extract_dir)
        extract_dir_path.mkdir(parents=True, exist_ok=True)
        extract_dir = str(extract_dir_path)

    print(f"Extrayendo {zip_path} a {extract_dir}...")

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    required_files = [
        "edges.edg.xml",
        "nodes.nod.xml",
        "routes.rou.xml",
        "simulation.sumocfg"
    ]

    missing_files = [f for f in required_files if not (Path(extract_dir) / f).exists()]
    if missing_files:
        raise ValueError(f"Archivos faltantes en el ZIP: {missing_files}")

    print("Archivos extraídos correctamente")
    return str(extract_dir)


def run_simulation_mode(
    simulation_dir: str,
    *,
    gui: bool = False,
    green_time: float | None = None,
    cycle_time: float | None = None,
    sim_steps: int | None = None,
    enable_dynamic_optimization: bool = False,
    seed: int | None = None,
) -> bool:
    """
    Ejecuta la simulación (headless o GUI) vía SimulationOrchestrator.

    Returns:
        True si la simulación se ejecutó correctamente, False en caso contrario
    """
    orchestrator = None
    try:
        from simulation_orchestrator import SimulationOrchestrator

        orchestrator = SimulationOrchestrator(
            simulation_dir,
            green_time=green_time,
            cycle_time=cycle_time,
            sim_steps=sim_steps,
            enable_dynamic_optimization=enable_dynamic_optimization,
            seed=seed,
            gui=gui,
        )

        if enable_dynamic_optimization:
            print("Optimización dinámica de clusters HABILITADA (API real)")

        if not orchestrator.setup_simulation():
            print("Error configurando simulación")
            return False

        print("Iniciando simulación...")
        orchestrator.run_simulation()  # limpia sus recursos en su propio finally

        stats = orchestrator.get_simulation_stats()
        print()
        print("=== Estadísticas Finales ===")
        print(f"Tiempo total de simulación: {stats.get('current_time', 0):.0f} segundos")
        print(f"Vehículos en el sistema: {stats.get('vehicle_count', 0)}")
        print(f"Cuellos de botella detectados: {stats.get('bottleneck_detections', 0)}")

        if stats.get('detection_history'):
            print("\nHistorial de detecciones:")
            for detection in stats['detection_history']:
                print(f"  - {detection['intersection_id']}: {detection['severity']} (t={detection['timestamp']:.0f}s)")

        # Generar visualizaciones si existen salidas de SUMO
        try:
            from visualization import generate_visualizations
            out_dir = str(Path(simulation_dir) / "logs" / "visualizations")
            viz_dir = generate_visualizations(simulation_dir, out_dir=out_dir)
            print(f"Visualizaciones guardadas en: {viz_dir}")
        except Exception as e:
            print(f"No se pudieron generar visualizaciones: {e}")

        print()
        print("Simulación completada exitosamente")
        return True

    except KeyboardInterrupt:
        print()
        print("Simulación interrumpida por el usuario")
        return True
    except Exception as e:
        print(f"Error durante la simulación: {e}")
        if orchestrator is not None:
            try:
                orchestrator._cleanup()
            except Exception as cleanup_error:
                print(f"Error en limpieza: {cleanup_error}")
        return False


def _resolve_compare_with(compare_with: str | None, extract_dir: str | None) -> str | None:
    """Resuelve el directorio A para comparación implícita (<base>_B → <base>_A)."""
    if compare_with:
        return compare_with
    if not extract_dir:
        return None
    p = Path(extract_dir)
    if p.name.endswith('_B'):
        candidate = p.with_name(p.name[:-2] + '_A')
        if candidate.exists():
            return str(candidate)
    return None


def main():
    """Función principal para ejecutar la simulación"""
    parser = argparse.ArgumentParser(
        description="Traffic-Sim: Simulador de Tráfico Inteligente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python run_simulation.py simulation.zip
  python run_simulation.py simulation.zip --gui
  python run_simulation.py simulation.zip --extract-dir ./mi_simulacion
        """
    )

    parser.add_argument(
        "zip_file",
        help="Ruta al archivo ZIP con archivos de simulación SUMO"
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Ejecutar con SUMO-GUI (interfaz gráfica)"
    )

    parser.add_argument(
        "--green-time",
        type=float,
        default=None,
        help="Tiempo de verde en segundos a aplicar a todos los semáforos"
    )

    parser.add_argument(
        "--cycle-time",
        type=float,
        default=None,
        help="Tiempo de ciclo en segundos (verde+rojo) para ajustar semáforos"
    )

    parser.add_argument(
        "--sim-steps",
        type=int,
        default=300,
        help="Número de pasos de simulación antes de detenerse automáticamente (default: 300)"
    )

    parser.add_argument(
        "--extract-dir",
        help="Directorio donde extraer los archivos (por defecto: temporal)"
    )

    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Mantener archivos extraídos después de la simulación"
    )

    parser.add_argument(
        "--compare-with",
        help="Ruta a otra ejecución (extract-dir) para comparar (A). Si no se especifica, y --extract-dir termina en _B, buscará sibling *_A.",
        default=None,
    )

    parser.add_argument(
        "--dynamic-optimization",
        action="store_true",
        help="Habilitar optimización dinámica de clusters de semáforos durante la simulación"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla aleatoria para SUMO (controla comportamiento estocástico de conductores)"
    )

    args = parser.parse_args()

    setup_logger("main")

    try:
        simulation_dir = extract_simulation_zip(args.zip_file, args.extract_dir)
    except Exception as e:
        print(f"Error extrayendo ZIP: {e}")
        return False

    success = run_simulation_mode(
        simulation_dir,
        gui=args.gui,
        green_time=args.green_time,
        cycle_time=args.cycle_time,
        sim_steps=args.sim_steps,
        enable_dynamic_optimization=args.dynamic_optimization,
        seed=args.seed,
    )

    # Comparación A/B (explícita o implícita) ANTES de limpiar archivos
    compare_with = _resolve_compare_with(args.compare_with, args.extract_dir)
    if compare_with:
        try:
            from visualization import generate_ab_test
            out_dir_ab = str(Path(simulation_dir) / 'logs' / 'visualizations' / 'ab_test')
            print(f"Running A/B comparison: A={compare_with}, B={simulation_dir}")
            report = generate_ab_test(compare_with, simulation_dir, out_dir=out_dir_ab, labels=('A', 'B'))
            print(f"A/B report written to: {report.get('csv')}, figures in: {report.get('out_dir')}")
        except Exception as e:
            print(f"Error running A/B comparison: {e}")

    # Limpiar archivos temporales
    if not args.keep_files and args.extract_dir is None:
        try:
            shutil.rmtree(simulation_dir)
        except Exception as e:
            print(f"Error limpiando archivos temporales: {e}")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
