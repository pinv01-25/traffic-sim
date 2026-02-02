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
from datetime import datetime, timezone
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
    
    print("Archivos extraídos correctamente")
    return str(extract_dir)

def run_with_sumo_gui(simulation_dir: str, green_time: float | None = None, cycle_time: float | None = None, sim_steps: int | None = None) -> bool:
    """
    Ejecuta la simulación con sumo-gui (interfaz gráfica)
    
    Args:
        simulation_dir: Directorio con archivos de simulación
        
    Returns:
        True si la simulación se ejecutó correctamente, False en caso contrario
    """
    try:
        import os
        import subprocess
        import threading
        import time
        from queue import Empty, Queue

        import traci
        from detectors.bottleneck_detector import BottleneckDetector
        from services.traffic_control_client import TrafficControlClient
        from utils.descriptive_names import descriptive_names
        
        # Configuración para threading
        traffic_control_client = TrafficControlClient()
        request_queue = Queue()
        worker_running = True
        worker_thread = None
        
        def http_worker():
            """Worker thread que procesa peticiones HTTP en segundo plano"""
            while worker_running:
                try:
                    # Obtener petición de la cola (con timeout para poder salir)
                    try:
                        batch_payload = request_queue.get(timeout=1.0)
                    except Empty:
                        continue  # Continuar el loop si no hay peticiones
                    
                    try:
                        # Enviar petición HTTP (esto es lo que antes bloqueaba)
                        response = traffic_control_client.send_traffic_data_batch(batch_payload)
                        
                        # Procesar respuesta
                        if response and response.get("status") == "success":
                            print("✅ Payload enviado exitosamente a traffic-control")
                            print(f"Respuesta: {response.get('message', 'Sin mensaje')}")
                        else:
                            print("❌ Error enviando payload a traffic-control")
                            print(f"Respuesta: {response}")
                    
                    except Exception as e:
                        print(f"❌ Error enviando a traffic-control: {e}")
                    
                    finally:
                        # Marcar la tarea como completada
                        request_queue.task_done()
                        
                except Exception as e:
                    print(f"Error en worker thread: {e}")
                    # Continuar procesando otras peticiones
        
        # Iniciar worker thread
        worker_thread = threading.Thread(
            target=http_worker,
            name="HTTP-Worker-GUI",
            daemon=True
        )
        worker_thread.start()
        print("Worker thread para peticiones HTTP iniciado")
        
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
                print(f"Error generando red: {result.stderr}")
                return False
            print("Red generada exitosamente")
        
        config_file = os.path.join(simulation_dir, "simulation.sumocfg")
        print("Iniciando simulación...")
        # Ensure SUMO output directory exists and instruct SUMO to write outputs
        sim_output_dir = os.path.join(simulation_dir, "logs", "sumo_output")
        os.makedirs(sim_output_dir, exist_ok=True)
        traci_cmd = [
            "sumo-gui",
            "-c", config_file,
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--tripinfo-output", os.path.join(sim_output_dir, "tripinfo.xml"),
            "--summary-output", os.path.join(sim_output_dir, "summary.xml"),
            "--fcd-output", os.path.join(sim_output_dir, "fcd.xml")
        ]
        traci.start(traci_cmd)
        # Apply signal timings if requested
        try:
            if green_time is not None and cycle_time is not None:
                # safeguard
                gt = float(green_time)
                ct = float(cycle_time)
                if gt < 0 or ct <= 0:
                    print("Invalid green or cycle time; skipping signal adjustment")
                else:
                    red_time = max(ct - gt, 0.0)
                    tls_list = traci.trafficlight.getIDList()
                    print(f"Applying signal timings: green={gt}s cycle={ct}s red={red_time}s to {len(tls_list)} traffic lights")
                    try:
                        from utils.signal_utils import apply_timings_to_all_tls
                        viz_dir = os.path.join(simulation_dir, 'logs', 'visualizations')
                        os.makedirs(viz_dir, exist_ok=True)
                        out_csv = os.path.join(viz_dir, 'tls_assigned_durations.csv')
                        apply_timings_to_all_tls(gt, ct, out_csv=out_csv)
                        print(f"TLS assigned durations written to: {out_csv}")
                    except Exception as e:
                        print(f"Error applying signal timings (utils): {e}")
        except Exception as e:
            print(f"Error applying signal timings: {e}")
        print("Simulación iniciada")

        detector = BottleneckDetector()
        last_detection_step = 0
        from config import BOTTLENECK_CONFIG
        detection_interval_steps = BOTTLENECK_CONFIG["detection_interval"]  # pasos entre detecciones
        step = 0
        try:
            # type: ignore
            while int(traci.simulation.getMinExpectedNumber()) > 0:
                traci.simulationStep()
                # type: ignore
                current_time = float(traci.simulation.getTime())
                # type: ignore
                vehicle_count = int(traci.vehicle.getIDCount())

                # Detectar cuellos de botella cada N pasos
                if (step - last_detection_step) >= detection_interval_steps:
                    detections = detector.detect_bottlenecks()
                    if detections:
                        print("\nCUELLO DE BOTELLA DETECTADO")
                        print(f"Paso: {step} | Tiempo: {current_time:.0f}s")
                        batch_timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                        sensors = []
                        for detection in detections:
                            intersection_name = descriptive_names.get_intersection_name(detection.intersection_id)
                            controlled_streets = detector.intersection_edges.get(detection.intersection_id, [])
                            print(f"Intersección: {intersection_name}")
                            print(f"Calles: {', '.join([descriptive_names.get_edge_name(edge) for edge in controlled_streets])}")
                            print(f"Severidad: {detection.severity.upper()}")
                            print("Métricas:")
                            print(f"   • Vehículos: {detection.metrics.get('vehicle_count', 0)}")
                            print(f"   • Velocidad promedio: {detection.metrics.get('average_speed', 0.0):.1f} m/s")
                            print(f"   • Densidad: {detection.metrics.get('density', 0.0):.2f} veh/km")
                            print(f"   • Cola: {detection.metrics.get('queue_length', 0)} vehículos")
                            print(f"Tiempo: {current_time:.0f}s")
                            print("=" * 50)
                            
                            controlled_edges = detector.intersection_edges.get(detection.intersection_id, [])
                            # Crear métricas para el payload
                            vehicle_count = int(detection.metrics.get('vehicle_count', 0))
                            average_speed = float(detection.metrics.get('average_speed', 0.0))
                            density = float(detection.metrics.get('density', 0.0))
                            
                            # Calcular vehicles_per_minute correctamente (vehículos por minuto)
                            vehicles_per_minute = int(vehicle_count * 60 / 60)  # Corregido: dividir por 60, no 3600
                            
                            # Obtener avg_circulation_time_sec del detector
                            avg_circulation_time_sec = float(detection.metrics.get('avg_circulation_time_sec', 30.0))
                            
                            metrics = {
                                'vehicles_per_minute': vehicles_per_minute,
                                'avg_speed_kmh': average_speed,
                                'avg_circulation_time_sec': avg_circulation_time_sec,
                                'density': density,
                                'vehicle_stats': detection.metrics.get('vehicle_stats', {
                                    'motorcycle': 0,
                                    'car': vehicle_count,
                                    'bus': 0,
                                    'truck': 0
                                })
                            }
                            
                            # Crear el sensor individual directamente
                            # Normalizar densidad a rango 0-1 (traffic-control espera valores entre 0 y 1)
                            normalized_density = min(metrics['density'] / 100.0, 1.0) if metrics['density'] > 1.0 else metrics['density']
                            
                            sensor_data = {
                                "traffic_light_id": detection.traffic_light_id,
                                "controlled_edges": controlled_edges,
                                "metrics": {
                                    "vehicles_per_minute": metrics['vehicles_per_minute'],
                                    "avg_speed_kmh": metrics['avg_speed_kmh'],
                                    "avg_circulation_time_sec": metrics['avg_circulation_time_sec'],
                                    "density": normalized_density
                                },
                                "vehicle_stats": {
                                    "motorcycle": metrics['vehicle_stats'].get('motorcycle', 0),
                                    "car": metrics['vehicle_stats'].get('car', 0),
                                    "bus": metrics['vehicle_stats'].get('bus', 0),
                                    "truck": metrics['vehicle_stats'].get('truck', 0)
                                }
                            }
                            sensors.append(sensor_data)
                        # Construir batch
                        batch_payload = {
                            "version": "2.0",
                            "type": "data",
                            "timestamp": batch_timestamp,
                            "traffic_light_id": detections[0].traffic_light_id if detections else "",
                            "sensors": sensors
                        }
                        import json as _json
                        print("\n========== BATCH PAYLOAD A TRAFFIC-CONTROL ==========")
                        print(_json.dumps(batch_payload, indent=2, ensure_ascii=False))
                        print("====================================================\n")
                        
                        # ENVIAR A TRAFFIC-CONTROL (versión no bloqueante)
                        try:
                            print("Enviando payload a traffic-control...")
                            # Agregar a la cola para procesamiento asíncrono
                            request_queue.put(batch_payload)
                            print("Petición agregada a cola para procesamiento asíncrono")
                        except Exception as e:
                            print(f"❌ Error agregando petición a cola: {e}")
                    else:
                        # Mensaje pequeño cada 15 pasos cuando no hay detecciones
                        print(f"Paso {step} | Tiempo {current_time:.0f}s | Vehículos {vehicle_count} | Sin cuellos de botella")
                    last_detection_step = step

                step += 1
                # Auto-stop after configured number of steps
                if sim_steps is not None and step >= int(sim_steps):
                    print(f"Reached sim_steps={sim_steps}, stopping simulation")
                    break
                time.sleep(0.05)  # 50ms para no saturar
        except KeyboardInterrupt:
            print("\nSimulación interrumpida por el usuario")
        finally:
            # Detener worker thread
            worker_running = False
            
            # Esperar a que el worker thread termine (máximo 5 segundos)
            if worker_thread and worker_thread.is_alive():
                worker_thread.join(timeout=5.0)
                if worker_thread.is_alive():
                    print("Worker thread no terminó en el tiempo esperado")
                else:
                    print("Worker thread terminado correctamente")
            
            traci.close()
            print("Simulación finalizada.")
            # Generar visualizaciones aunque el usuario haya interrumpido
            try:
                from visualization import generate_visualizations
                out_dir = str(Path(simulation_dir) / "logs" / "visualizations")
                viz_dir = generate_visualizations(simulation_dir, out_dir=out_dir)
                print(f"Visualizaciones guardadas en: {viz_dir}")
            except Exception as e:
                print(f"No se pudieron generar visualizaciones: {e}")
        return True
    except Exception as e:
        print(f"Error ejecutando SUMO-GUI: {e}")
        return False

def run_with_sumo_headless(simulation_dir: str, green_time: float | None = None, cycle_time: float | None = None, sim_steps: int | None = None) -> bool:
    """
    Ejecuta la simulación con sumo (modo headless)
    
    Args:
        simulation_dir: Directorio con archivos de simulación
        
    Returns:
        True si la simulación se ejecutó correctamente, False en caso contrario
    """
    try:
        from simulation_orchestrator import SimulationOrchestrator

        # Crear orquestador (pasar tiempos de semáforo y sim_steps if present)
        orchestrator = SimulationOrchestrator(simulation_dir, green_time=green_time, cycle_time=cycle_time, sim_steps=sim_steps)
        
        # Configurar simulación
        if not orchestrator.setup_simulation():
            print("Error configurando simulación")
            return False
        
        print("Iniciando simulación...")
        
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
        
        # intentar generar visualizaciones si existen salidas de SUMO
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
        return False
    finally:
        try:
            orchestrator._cleanup()
        except Exception as e:
            print(f"Error en limpieza: {e}")



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
    
    args = parser.parse_args()
    
    setup_logger("main")
    
    # Extraer archivo ZIP
    try:
        simulation_dir = extract_simulation_zip(args.zip_file, args.extract_dir)
    except Exception as e:
        print(f"Error extrayendo ZIP: {e}")
        return False
    
    # Ejecutar según el modo seleccionado
    if args.gui:
        success = run_with_sumo_gui(
            simulation_dir,
            green_time=args.green_time,
            cycle_time=args.cycle_time,
            sim_steps=args.sim_steps,
        )
    else:
        success = run_with_sumo_headless(
            simulation_dir,
            green_time=args.green_time,
            cycle_time=args.cycle_time,
            sim_steps=args.sim_steps,
        )
    
    # Limpiar archivos temporales
    if not args.keep_files and args.extract_dir is None:
        try:
            shutil.rmtree(simulation_dir)
        except Exception as e:
            print(f"Error limpiando archivos temporales: {e}")

    # If user requested an A/B comparison, or implicit sibling exists, run comparison
    try:
        compare_with = args.compare_with
        if compare_with is None and args.extract_dir:
            # implicit detection: if extract dir endswith _B, look for sibling *_A
            try:
                p = Path(args.extract_dir)
                name = p.name
                if name.endswith('_B'):
                    candidate = str(p.with_name(name[:-2] + '_A'))
                    from pathlib import Path as _P
                    if _P(candidate).exists():
                        compare_with = candidate
            except Exception:
                compare_with = None

        if compare_with:
            try:
                from visualization import generate_ab_test
                out_dir_ab = str(Path(simulation_dir) / 'logs' / 'visualizations' / 'ab_test')
                print(f"Running A/B comparison: A={compare_with}, B={simulation_dir}")
                report = generate_ab_test(compare_with, simulation_dir, out_dir=out_dir_ab, labels=('A', 'B'))
                print(f"A/B report written to: {report.get('csv')}, figures in: {report.get('out_dir')}")
            except Exception as e:
                print(f"Error running A/B comparison: {e}")
    except Exception:
        pass
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 