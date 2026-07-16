"""
Orquestador principal de la simulación de tráfico
Coordina detección de cuellos de botella, comunicación con traffic-control y actualización de semáforos

Modos:
- Fixed-time (default): sin comunicación con traffic-control — baseline limpio.
- Estático: green_time/cycle_time aplicados al inicio vía signal_utils.
- Dinámico (enable_dynamic_optimization): detección → cluster → /ingest → aplicación.
"""

import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import traci
from config import BOTTLENECK_CONFIG, SIMULATION_CONFIG
from controllers.traffic_light_controller import TrafficLightController
from detectors.bottleneck_detector import BottleneckDetection, BottleneckDetector
from services.traffic_control_client import (
    ClusterOptimizationResponse,
    RawSimulationPayload,
    TrafficControlClient,
)
from utils.descriptive_names import descriptive_names
from utils.logger import Colors, get_simulation_logger


class SimulationOrchestrator:
    """
    Orquestador principal de la simulación de tráfico
    Coordina todos los componentes del sistema
    """

    def __init__(
        self,
        simulation_dir: str = "simulation",
        green_time: float | None = None,
        cycle_time: float | None = None,
        sim_steps: int | None = None,
        enable_dynamic_optimization: bool = False,
        seed: int | None = None,
        gui: bool = False,
    ):
        self.simulation_dir = Path(simulation_dir)
        # optional signal timing overrides
        self.green_time = green_time
        self.cycle_time = cycle_time
        # optional auto-stop after number of simulation steps
        self.sim_steps = int(sim_steps) if sim_steps is not None else None
        # optional SUMO random seed
        self.seed = seed
        self.gui = gui
        self.logger = get_simulation_logger()

        # Configuración de optimización dinámica de clusters
        self.enable_dynamic_optimization = enable_dynamic_optimization

        # Componentes del sistema
        self.bottleneck_detector = None
        self.traffic_control_client = None
        self.traffic_light_controller = None

        # Estado de la simulación
        self.last_detection_step = 0
        self.current_step = 0
        self.bottleneck_history = []

        # Configuración
        self.end_time = SIMULATION_CONFIG["end_time"]
        self.detection_interval = BOTTLENECK_CONFIG["detection_interval"]

        # La API puede leer el historial desde otro thread
        self._history_lock = threading.Lock()

        self.logger.info("Orquestador de simulación inicializado")

    def setup_simulation(self) -> bool:
        """
        Configura la simulación SUMO

        Returns:
            True si la configuración fue exitosa
        """
        try:
            self.logger.info("Configurando simulación SUMO...")

            if not self._check_sumo_installation():
                self.logger.error("SUMO no está instalado o no está en el PATH")
                return False

            if not self._generate_network():
                self.logger.error("Error generando la red de simulación")
                return False

            if not self._start_traci_connection():
                self.logger.error("Error iniciando conexión traci")
                return False

            self._initialize_components()

            self.logger.info("Simulación configurada exitosamente")
            return True

        except Exception as e:
            self.logger.error(f"Error configurando simulación: {e}")
            return False

    def _sumo_binary(self) -> str:
        return "sumo-gui" if self.gui else "sumo"

    def _check_sumo_installation(self) -> bool:
        """Verifica que SUMO esté instalado"""
        try:
            result = subprocess.run([self._sumo_binary(), "--version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _generate_network(self) -> bool:
        """Genera la red de simulación usando netconvert"""
        try:
            network_file = self.simulation_dir / "network.net.xml"

            if network_file.exists():
                self.logger.info("Red de simulación ya existe")
                return True

            cmd = [
                "netconvert",
                "--node-files", str(self.simulation_dir / "nodes.nod.xml"),
                "--edge-files", str(self.simulation_dir / "edges.edg.xml"),
                "--output-file", str(network_file),
                "--no-turnarounds",
                "--tls.guess", "true"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info("Red de simulación generada exitosamente")
                return True
            else:
                self.logger.error(f"Error generando red: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error en generación de red: {e}")
            return False

    def _start_traci_connection(self) -> bool:
        """Inicia la conexión TraCI con SUMO"""
        try:
            config_file = self.simulation_dir / "simulation.sumocfg"

            if not config_file.exists():
                self.logger.error(f"Archivo de configuración no encontrado: {config_file}")
                return False

            # Iniciar SUMO con TraCI y asegurar que produce salidas útiles
            output_dir = str(self.simulation_dir / "logs" / "sumo_output")
            os.makedirs(output_dir, exist_ok=True)
            traci_cmd = [
                self._sumo_binary(),
                "-c", str(config_file),
                "--no-step-log", "true",
                "--time-to-teleport", "-1",
                "--tripinfo-output", os.path.join(output_dir, "tripinfo.xml"),
                "--summary-output", os.path.join(output_dir, "summary.xml"),
                "--fcd-output", os.path.join(output_dir, "fcd.xml"),
            ]
            if self.seed is not None:
                traci_cmd.extend(["--seed", str(self.seed)])
            traci.start(traci_cmd)

            self._apply_static_timings()

            self.logger.info("Conexión TraCI establecida")
            return True

        except Exception as e:
            self.logger.error(f"Error iniciando TraCI: {e}")
            return False

    def _apply_static_timings(self):
        """Aplica green_time/cycle_time a todos los semáforos si fueron pedidos"""
        if self.green_time is None or self.cycle_time is None:
            return
        try:
            gt = float(self.green_time)
            ct = float(self.cycle_time)
            if gt < 0 or ct <= 0:
                self.logger.warning("Invalid green or cycle time; skipping signal adjustment")
                return

            from utils.signal_utils import apply_timings_to_all_tls
            tls_count = len(traci.trafficlight.getIDList())
            self.logger.info(
                f"Applying signal timings: green={gt}s cycle={ct}s to {tls_count} traffic lights"
            )
            viz_dir = self.simulation_dir / 'logs' / 'visualizations'
            viz_dir.mkdir(parents=True, exist_ok=True)
            out_csv = str(viz_dir / 'tls_assigned_durations.csv')
            apply_timings_to_all_tls(gt, ct, out_csv=out_csv)
            self.logger.info(f"TLS assigned durations written to: {out_csv}")
        except Exception as e:
            self.logger.error(f"Error applying signal timings: {e}")

    def _initialize_components(self):
        """Inicializa los componentes del sistema"""
        try:
            self.bottleneck_detector = BottleneckDetector()

            # Solo el modo dinámico habla con traffic-control y toca semáforos:
            # el baseline queda aislado por construcción.
            if self.enable_dynamic_optimization:
                self.traffic_control_client = TrafficControlClient()
                self.traffic_light_controller = TrafficLightController()

            self.logger.info("Componentes inicializados")

        except Exception as e:
            self.logger.error(f"Error inicializando componentes: {e}")
            raise

    def run_simulation(self):
        """Ejecuta la simulación principal"""
        try:
            self.logger.info("Iniciando simulación...")

            while self._should_continue_simulation():
                traci.simulationStep()
                self.current_step += 1

                current_time = float(traci.simulation.getTime())

                if self._should_detect_bottlenecks():
                    self._handle_bottleneck_detection(current_time)

                if self.gui:
                    time.sleep(0.05)  # ritmo visible para el usuario

        except KeyboardInterrupt:
            self.logger.info("Simulación interrumpida por el usuario")
        except Exception as e:
            self.logger.error(f"Error durante la simulación: {e}")
        finally:
            self._cleanup()

    def _should_continue_simulation(self) -> bool:
        """Determina si la simulación debe continuar"""
        try:
            # Vehículos esperados (incluye los que aún no se insertaron)
            expected_vehicles = int(traci.simulation.getMinExpectedNumber())
            current_time = float(traci.simulation.getTime())

            if expected_vehicles <= 0 or current_time >= self.end_time:
                return False
            if self.sim_steps is not None and self.current_step >= self.sim_steps:
                return False
            return True

        except Exception as e:
            self.logger.error(f"Error verificando estado de simulación: {e}")
            return False

    def _should_detect_bottlenecks(self) -> bool:
        """Determina si es momento de detectar cuellos de botella"""
        return (self.current_step - self.last_detection_step) >= self.detection_interval

    def _handle_bottleneck_detection(self, current_time: float):
        """Maneja la detección de cuellos de botella"""
        try:
            detections = self.bottleneck_detector.detect_bottlenecks()

            if detections:
                self.logger.info(f"Se detectaron {len(detections)} cuellos de botella")

                for detection in detections:
                    self._process_bottleneck_detection(detection, current_time)

                    intersection_name = descriptive_names.get_intersection_name(detection.intersection_id)
                    controlled_streets = [
                        descriptive_names.get_edge_name(edge)
                        for edge in self.bottleneck_detector.intersection_edges.get(detection.traffic_light_id, [])
                    ]

                    print(f"\n{Colors.RED}{Colors.BOLD}CUELLO DE BOTELLA DETECTADO{Colors.END}")
                    print(f"{Colors.RED}Paso: {self.current_step} | Tiempo: {current_time:.0f}s{Colors.END}")
                    print(f"{Colors.RED}Intersección: {intersection_name}{Colors.END}")
                    print(f"{Colors.RED}Calles: {', '.join(controlled_streets)}{Colors.END}")
                    print(f"{Colors.RED}Severidad: {detection.severity.upper()}{Colors.END}")
                    print(f"{Colors.RED}Métricas:{Colors.END}")
                    print(f"{Colors.RED}   • Vehículos: {detection.metrics.get('vehicle_count', 0)}{Colors.END}")
                    print(f"{Colors.RED}   • Velocidad promedio: {detection.metrics.get('average_speed', 0.0):.1f} m/s{Colors.END}")
                    print(f"{Colors.RED}   • Densidad: {detection.metrics.get('density', 0.0):.2f} veh/km{Colors.END}")
                    print(f"{Colors.RED}   • Cola: {detection.metrics.get('queue_length', 0)} vehículos{Colors.END}")
                    print(f"{Colors.RED}{'='*50}{Colors.END}")

            self.last_detection_step = self.current_step

        except Exception as e:
            self.logger.error(f"Error en detección de cuellos de botella: {e}")

    def _process_bottleneck_detection(self, detection: BottleneckDetection, current_time: float):
        """Registra la detección y, en modo dinámico, dispara la optimización de cluster"""
        try:
            if self.enable_dynamic_optimization:
                self._process_cluster_optimization(detection, current_time)
            else:
                # Baseline / estático: solo registrar, sin comunicación externa
                with self._history_lock:
                    self.bottleneck_history.append({
                        "timestamp": current_time,
                        "intersection_id": detection.intersection_id,
                        "severity": detection.severity,
                        "metrics": detection.metrics,
                    })

        except Exception as e:
            self.logger.error(f"Error procesando detección: {e}")

    def _process_cluster_optimization(self, detection: BottleneckDetection, current_time: float):
        """
        Procesa optimización de cluster: obtiene semáforos cercanos,
        envía datos crudos a traffic-control /ingest y aplica las
        optimizaciones recibidas.
        """
        try:
            primary_tl_id = detection.traffic_light_id

            nearby_tls = self._get_nearby_traffic_lights(primary_tl_id)
            self.logger.info(f"Cluster de {len(nearby_tls)} semáforos para optimización")

            raw_payload = self._create_raw_cluster_payload(primary_tl_id, nearby_tls)

            print(f"\n{Colors.CYAN}{Colors.BOLD}PAYLOAD DE CLUSTER A TRAFFIC-CONTROL (/ingest){Colors.END}")
            print(f"{Colors.CYAN}Semáforo primario: {primary_tl_id}{Colors.END}")
            print(f"{Colors.CYAN}Semáforos en cluster: {nearby_tls}{Colors.END}")
            print(f"{Colors.CYAN}{'='*50}{Colors.END}")
            print(json.dumps(raw_payload.to_dict(), indent=2, ensure_ascii=False))
            print(f"{Colors.CYAN}{'='*50}{Colors.END}\n")

            response = self.traffic_control_client.send_raw_data(raw_payload)

            self._apply_cluster_optimization(response)

            with self._history_lock:
                self.bottleneck_history.append({
                    "timestamp": current_time,
                    "intersection_id": detection.intersection_id,
                    "severity": detection.severity,
                    "metrics": detection.metrics,
                    "cluster_size": len(nearby_tls),
                    "optimizations_applied": len(response.optimizations) if response.status == "success" else 0
                })

        except Exception as e:
            self.logger.error(f"Error en optimización de cluster: {e}")

    def _get_nearby_traffic_lights(self, primary_tl_id: str, max_distance: float = 200.0) -> list:
        """
        Obtiene los semáforos cercanos al semáforo primario.

        Args:
            primary_tl_id: ID del semáforo con cuello de botella
            max_distance: Distancia máxima en metros para considerar "cercano"

        Returns:
            Lista de IDs de semáforos cercanos (incluyendo el primario)
        """
        try:
            all_tls = list(traci.trafficlight.getIDList())

            # Si solo hay pocos semáforos, devolver todos
            if len(all_tls) <= 4:
                return all_tls

            try:
                primary_pos = traci.junction.getPosition(primary_tl_id)
            except traci.exceptions.TraCIException:
                return [primary_tl_id]

            nearby = [primary_tl_id]

            for tl_id in all_tls:
                if tl_id == primary_tl_id:
                    continue
                try:
                    tl_pos = traci.junction.getPosition(tl_id)
                    distance = ((primary_pos[0] - tl_pos[0]) ** 2 +
                               (primary_pos[1] - tl_pos[1]) ** 2) ** 0.5
                    if distance <= max_distance:
                        nearby.append(tl_id)
                except traci.exceptions.TraCIException:
                    continue

            self.logger.debug(f"Semáforos cercanos a {primary_tl_id}: {nearby}")
            return nearby

        except Exception as e:
            self.logger.error(f"Error obteniendo semáforos cercanos: {e}")
            return [primary_tl_id]

    def _create_raw_cluster_payload(
        self,
        primary_tl_id: str,
        nearby_tls: list,
    ) -> RawSimulationPayload:
        """
        Crea un payload crudo con datos de todos los semáforos del cluster.

        No realiza ninguna normalización — traffic-control se encarga de:
        - Normalización de IDs (extraer dígitos de IDs SUMO)
        - Normalización de densidad (veh/km → 0-1)
        - Relleno de vehicle_stats (asegurar 4 claves)
        - Asignación de versión
        """
        sensors = []

        for tl_id in nearby_tls:
            intersection_data = self.bottleneck_detector.get_intersection_data(tl_id)

            if intersection_data:
                sensors.append({
                    "traffic_light_id": tl_id,  # ID SUMO crudo, sin normalizar
                    "controlled_edges": self.bottleneck_detector.intersection_edges.get(tl_id, []),
                    "metrics": {
                        "vehicles_per_minute": intersection_data.vehicle_count,
                        "avg_speed_kmh": intersection_data.average_speed,
                        "avg_circulation_time_sec": intersection_data.avg_circulation_time,
                        "density": intersection_data.density,  # veh/km crudo
                    },
                    "vehicle_stats": intersection_data.vehicle_stats or None,
                })

        return RawSimulationPayload(
            timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            source_id=primary_tl_id,
            sensors=sensors,
        )

    def _apply_cluster_optimization(self, response: ClusterOptimizationResponse):
        """
        Aplica las optimizaciones recibidas de traffic-sync (vía traffic-control)
        a los semáforos del cluster en SUMO.

        Cada optimización agrupa semáforos por cluster_sensors; se aplica el
        mismo green/red a todos los semáforos de cada cluster.
        """
        if response.status != "success":
            self.logger.warning(f"Optimización de cluster fallida: {response.message}")
            return

        # Mapear IDs normalizados (solo dígitos) a IDs originales de SUMO
        normalized_to_sumo = {}
        for tl_id in traci.trafficlight.getIDList():
            match = re.search(r"(\d+)", str(tl_id))
            normalized_to_sumo[match.group(1) if match else tl_id] = tl_id

        applied_count = 0

        for opt in response.optimizations:
            target_tl_ids = opt.cluster_sensors if opt.cluster_sensors else [opt.traffic_light_id]

            for normalized_id in target_tl_ids:
                sumo_id = normalized_to_sumo.get(normalized_id, normalized_id)
                try:
                    success = self.traffic_light_controller.update_traffic_light(
                        sumo_id,
                        {
                            "optimization": {
                                "green_time_sec": opt.green_time_sec,
                                "red_time_sec": opt.red_time_sec,
                            }
                        },
                    )
                    if success:
                        applied_count += 1
                        self.logger.info(
                            f"Optimización aplicada a {sumo_id} (norm:{normalized_id}): "
                            f"green={opt.green_time_sec}s, red={opt.red_time_sec}s "
                            f"[cluster: {opt.cluster_sensors}]"
                        )
                except Exception as e:
                    self.logger.error(f"Error aplicando optimización a {sumo_id}: {e}")

        if applied_count > 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}OPTIMIZACIÓN DE CLUSTER APLICADA{Colors.END}")
            print(f"{Colors.GREEN}Semáforos actualizados: {applied_count}{Colors.END}")
            for opt in response.optimizations:
                print(
                    f"{Colors.GREEN}  Cluster {opt.cluster_sensors}: "
                    f"green={opt.green_time_sec}s, red={opt.red_time_sec}s "
                    f"({opt.original_category} → {opt.optimized_category}){Colors.END}"
                )
            print(f"{Colors.GREEN}{'='*50}{Colors.END}\n")

    def _cleanup(self):
        """Libera recursos de la simulación"""
        try:
            # Snapshot de stats finales antes de cerrar (tras close no hay red)
            try:
                self._final_time = float(traci.simulation.getTime())
                self._final_vehicle_count = int(traci.vehicle.getIDCount())
            except Exception:
                pass
            try:
                traci.close()
            except Exception:
                # Conexión ya cerrada o inexistente
                pass
            self.logger.info("Recursos de simulación liberados")
        except Exception as e:
            self.logger.error(f"Error en limpieza: {e}")

    def get_simulation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la simulación"""
        try:
            try:
                current_time = float(traci.simulation.getTime())
                vehicle_count = traci.vehicle.getIDCount()
            except Exception:
                # Conexión cerrada: usar el snapshot tomado en _cleanup
                current_time = getattr(self, '_final_time', 0.0)
                vehicle_count = getattr(self, '_final_vehicle_count', 0)

            with self._history_lock:
                history_len = len(self.bottleneck_history)
                history_copy = list(self.bottleneck_history)

            return {
                "current_time": current_time,
                "vehicle_count": vehicle_count,
                "bottleneck_detections": history_len,
                "detection_history": history_copy,
            }
        except Exception as e:
            self.logger.error(f"Error obteniendo estadísticas: {e}")
            return {}


def main():
    """Función principal para testing"""
    orchestrator = SimulationOrchestrator()

    if orchestrator.setup_simulation():
        orchestrator.run_simulation()
    else:
        print("Error configurando simulación")


if __name__ == "__main__":
    main()
