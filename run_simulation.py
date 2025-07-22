#!/usr/bin/env python3
"""
Script principal para ejecutar la simulación de tráfico
Acepta un archivo ZIP con los archivos de simulación SUMO
"""

import sys
import os
import zipfile
import tempfile
import shutil
import argparse
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
    
    # Crear directorio temporal si no se especifica
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp(prefix="traffic_sim_")
    else:
        extract_dir_path = Path(extract_dir)
        extract_dir_path.mkdir(parents=True, exist_ok=True)
        extract_dir = str(extract_dir_path)
    
    print(f"Extrayendo {zip_path} a {extract_dir}...")
    
    # Extraer ZIP
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Verificar archivos requeridos
    required_files = [
        "edges.edg.xml",
        "nodes.nod.xml", 
        "routes.rou.xml",
        "simulation.sumocfg"
    ]
    
    missing_files = []
    for file in required_files:
        if not (Path(extract_dir) / file).exists():
            missing_files.append(file)
    
    if missing_files:
        raise ValueError(f"Archivos faltantes en el ZIP: {missing_files}")
    
    print("✅ Archivos extraídos correctamente")
    return str(extract_dir)

def run_with_sumo_gui(simulation_dir: str) -> bool:
    """
    Ejecuta la simulación con sumo-gui (interfaz gráfica)
    
    Args:
        simulation_dir: Directorio con archivos de simulación
        
    Returns:
        True si la simulación se ejecutó correctamente, False en caso contrario
    """
    try:
        import traci
        import time
        import os
        import subprocess
        from detectors.bottleneck_detector import BottleneckDetector
        from services.traffic_control_client import TrafficControlClient
        traffic_control_client = TrafficControlClient()
        
        # Generar red si no existe
        network_file = os.path.join(simulation_dir, "network.net.xml")
        if not os.path.exists(network_file):
            print("Generando red con netconvert...")
            netconvert_cmd = [
                "netconvert",
                "--node-files", "nodes.nod.xml",
                "--edge-files", "edges.edg.xml",
                "--output-file", "network.net.xml",
                "--no-turnarounds",
                "--tls.guess", "true"
            ]
            result = subprocess.run(netconvert_cmd, cwd=simulation_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Error generando red: {result.stderr}")
                return False
            print("✅ Red generada exitosamente")
        
        config_file = os.path.join(simulation_dir, "simulation.sumocfg")
        print("🖥️  Iniciando SUMO-GUI con control TraCI desde Python...")
        traci.start([
            "sumo-gui",
            "-c", config_file,
            "--no-step-log", "true",
            "--time-to-teleport", "-1"
        ])
        print("✅ SUMO-GUI iniciado y conectado via TraCI")
        print("🎉 NO pulses play en la GUI, el script controla el avance.")
        print("📝 El debug output aparecerá en esta consola")
        print("🛑 Presiona Ctrl+C para detener")

        detector = BottleneckDetector()
        print("✅ Detector de cuellos de botella configurado")
        last_detection_time = 0
        detection_interval = 10  # segundos
        step = 0
        try:
            # type: ignore
            while int(traci.simulation.getMinExpectedNumber()) > 0:
                traci.simulationStep()
                # type: ignore
                current_time = float(traci.simulation.getTime())
                # type: ignore
                vehicle_count = int(traci.vehicle.getIDCount())

                # Detectar cuellos de botella periódicamente
                if (current_time - last_detection_time) >= detection_interval:
                    print(f"\n🔍 DETECCIÓN EN TIEMPO {current_time:.0f}s")
                    detections = detector.detect_bottlenecks()
                    if detections:
                        print(f"🚨 Se detectaron {len(detections)} cuellos de botella")
                        for detection in detections:
                            print(f"📍 {detection.intersection_id}: {detection.severity}")
                            # Imprimir el payload que se enviaría a traffic-control
                            controlled_edges = detector.intersection_edges.get(detection.intersection_id, [])
                            payload = traffic_control_client.create_traffic_payload(
                                traffic_light_id=detection.traffic_light_id,
                                controlled_edges=controlled_edges,
                                vehicle_count=int(detection.metrics['vehicle_count']),
                                average_speed=float(detection.metrics['average_speed']),
                                density=float(detection.metrics['density']),
                                queue_length=int(detection.metrics['queue_length']),
                                timestamp=None
                            )
                            import json as _json
                            print("\n========== PAYLOAD A TRAFFIC-CONTROL ==========")
                            print(_json.dumps(payload.to_dict(), indent=2, ensure_ascii=False))
                            print("==============================================\n")
                    else:
                        print("✅ No se detectaron cuellos de botella")
                    last_detection_time = current_time

                # Log de progreso cada 10 segundos
                if int(current_time) % 10 == 0 and current_time > 0:
                    print(f"⏰ Tiempo: {current_time:.0f}s | Vehículos: {vehicle_count}")

                step += 1
                time.sleep(0.05)  # 50ms para no saturar
        except KeyboardInterrupt:
            print("\n⚠️  Simulación interrumpida por el usuario")
        finally:
            traci.close()
            print("✅ Simulación finalizada.")
        return True
    except Exception as e:
        print(f"❌ Error ejecutando SUMO-GUI: {e}")
        return False

def run_with_sumo_headless(simulation_dir: str) -> bool:
    """
    Ejecuta la simulación con sumo (modo headless) usando el orquestador
    
    Args:
        simulation_dir: Directorio con archivos de simulación
        
    Returns:
        True si la simulación se ejecutó correctamente, False en caso contrario
    """
    try:
        from simulation_orchestrator import SimulationOrchestrator
        
        # Crear orquestador
        orchestrator = SimulationOrchestrator(simulation_dir)
        print("✅ Orquestador creado exitosamente")
        
        # Configurar simulación
        print("Configurando simulación...")
        if not orchestrator.setup_simulation():
            print("❌ Error configurando simulación")
            return False
        print("✅ Simulación configurada exitosamente")
        
        print()
        print("Iniciando simulación en modo headless...")
        print("Presiona Ctrl+C para detener")
        print()
        
        # Ejecutar simulación
        orchestrator.run_simulation()
        
        # Mostrar estadísticas finales
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
        
        print()
        print("✅ Simulación completada exitosamente")
        return True
        
    except KeyboardInterrupt:
        print()
        print("⚠️  Simulación interrumpida por el usuario")
        return True
    except Exception as e:
        print(f"❌ Error durante la simulación: {e}")
        return False
    finally:
        try:
            orchestrator._cleanup()
        except Exception as e:
            print(f"⚠️  Error en limpieza: {e}")

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
        "--extract-dir",
        help="Directorio donde extraer los archivos (por defecto: temporal)"
    )
    
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Mantener archivos extraídos después de la simulación"
    )
    
    args = parser.parse_args()
    
    logger = setup_logger("main")
    
    print("=== Traffic-Sim: Simulador de Tráfico Inteligente ===")
    print()
    
    # Extraer archivo ZIP
    try:
        simulation_dir = extract_simulation_zip(args.zip_file, args.extract_dir)
        print(f"Directorio de simulación: {simulation_dir}")
        print()
    except Exception as e:
        print(f"❌ Error extrayendo ZIP: {e}")
        return False
    
    # Ejecutar según el modo seleccionado
    if args.gui:
        print("🖥️  Modo GUI seleccionado")
        print("✅ Tendrás GUI Y debug output en consola")
        success = run_with_sumo_gui(simulation_dir)
    else:
        print("🤖 Modo headless seleccionado")
        print("✅ Este modo mostrará debug output de detección de cuellos de botella")
        success = run_with_sumo_headless(simulation_dir)
    
    # Limpiar archivos temporales
    if not args.keep_files and args.extract_dir is None:
        try:
            print(f"Limpiando archivos temporales: {simulation_dir}")
            shutil.rmtree(simulation_dir)
        except Exception as e:
            print(f"⚠️  Error limpiando archivos temporales: {e}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 