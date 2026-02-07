"""
Orquestador principal de la simulación de tráfico
Coordina detección de cuellos de botella, comunicación con traffic-control y actualización de semáforos
"""

import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Optional

import traci
from config import BOTTLENECK_CONFIG, SIMULATION_CONFIG
from controllers.traffic_light_controller import TrafficLightController
from detectors.bottleneck_detector import BottleneckDetection, BottleneckDetector
from services.traffic_control_client import (
    ClusterDataPayload,
    ClusterOptimizationResponse,
    TrafficControlClient,
    TrafficDataPayload,
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
        use_mock_api: bool = True
    ):
        self.simulation_dir = Path(simulation_dir)
        # optional signal timing overrides
        self.green_time = green_time
        self.cycle_time = cycle_time
        # optional auto-stop after number of simulation steps
        self.sim_steps = int(sim_steps) if sim_steps is not None else None
        self.logger = get_simulation_logger()

        # Configuración de optimización dinámica de clusters
        self.enable_dynamic_optimization = enable_dynamic_optimization
        self.use_mock_api = use_mock_api  # True = usar mock, False = usar API real
        
        # Componentes del sistema
        self.bottleneck_detector = None
        self.traffic_control_client = None
        self.traffic_light_controller = None
        
        # Estado de la simulación
        self.simulation_running = False
        self.last_detection_step = 0
        self.current_step = 0
        self.bottleneck_history = []
        
        # Configuración
        self.begin_time = SIMULATION_CONFIG["begin_time"]
        self.end_time = SIMULATION_CONFIG["end_time"]
        self.step_length = SIMULATION_CONFIG["step_length"]
        self.detection_interval = BOTTLENECK_CONFIG["detection_interval"]
        
        # Atributos para threading
        self.request_queue = Queue()
        self.response_queue = Queue()
        self.worker_thread = None
        self.worker_running = False
        self._history_lock = threading.Lock()  # Lock para acceso thread-safe al historial

        self.logger.info("Orquestador de simulación inicializado")
    
    def setup_simulation(self) -> bool:
        """
        Configura la simulación SUMO
        
        Returns:
            True si la configuración fue exitosa
        """
        try:
            self.logger.info("Configurando simulación SUMO...")
            
            # Verificar que SUMO esté instalado
            if not self._check_sumo_installation():
                self.logger.error("SUMO no está instalado o no está en el PATH")
                return False
            
            # Generar red si no existe
            if not self._generate_network():
                self.logger.error("Error generando la red de simulación")
                return False
            
            # Iniciar conexión traci
            if not self._start_traci_connection():
                self.logger.error("Error iniciando conexión traci")
                return False
            
            # Inicializar componentes
            self._initialize_components()
            
            self.logger.info("Simulación configurada exitosamente")
            return True
            
        except Exception as e:
            self.logger.error(f"Error configurando simulación: {e}")
            return False
    
    def _check_sumo_installation(self) -> bool:
        """Verifica que SUMO esté instalado"""
        try:
            result = subprocess.run(["sumo", "--version"], capture_output=True, text=True)
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
            
            # Comando netconvert
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
            # Configurar archivo de configuración
            config_file = self.simulation_dir / "simulation.sumocfg"
            
            if not config_file.exists():
                self.logger.error(f"Archivo de configuración no encontrado: {config_file}")
                return False
            
            # Iniciar SUMO con TraCI y asegurar que produce salidas útiles
            output_dir = str(self.simulation_dir / "logs" / "sumo_output")
            import os
            os.makedirs(output_dir, exist_ok=True)
            traci_cmd = [
                "sumo",
                "-c", str(config_file),
                "--no-step-log", "true",
                "--time-to-teleport", "-1",
                "--tripinfo-output", os.path.join(output_dir, "tripinfo.xml"),
                "--summary-output", os.path.join(output_dir, "summary.xml"),
                "--fcd-output", os.path.join(output_dir, "fcd.xml"),
            ]
            traci.start(traci_cmd)
            # Apply signal timings if requested
            try:
                if hasattr(self, 'green_time') and self.green_time is not None and self.cycle_time is not None:
                    gt = float(self.green_time)
                    ct = float(self.cycle_time)
                    if gt < 0 or ct <= 0:
                        self.logger.warning("Invalid green or cycle time; skipping signal adjustment")
                    else:
                        red_time = max(ct - gt, 0.0)
                        tls_list = traci.trafficlight.getIDList()
                        self.logger.info(f"Applying signal timings: green={gt}s cycle={ct}s red={red_time}s to {len(tls_list)} traffic lights")
                        try:
                            from utils.signal_utils import apply_timings_to_all_tls
                            viz_dir = str(self.simulation_dir / 'logs' / 'visualizations')
                            import os as _os
                            _os.makedirs(viz_dir, exist_ok=True)
                            out_csv = _os.path.join(viz_dir, 'tls_assigned_durations.csv')
                            apply_timings_to_all_tls(gt, ct, out_csv=out_csv)
                            self.logger.info(f"TLS assigned durations written to: {out_csv}")
                        except Exception as e:
                            self.logger.error(f"Error applying signal timings (utils): {e}")
            except Exception as e:
                self.logger.error(f"Error applying signal timings: {e}")
            
            self.logger.info("Conexión TraCI establecida")
            return True
            
        except Exception as e:
            self.logger.error(f"Error iniciando TraCI: {e}")
            return False
    
    def _start_worker_thread(self):
        """Inicia el thread worker para manejar peticiones HTTP"""
        try:
            self.worker_running = True
            self.worker_thread = threading.Thread(
                target=self._http_worker,
                name="HTTP-Worker",
                daemon=True  # Se cerrará automáticamente cuando el programa principal termine
            )
            self.worker_thread.start()
            self.logger.info("Worker thread para peticiones HTTP iniciado")
        except Exception as e:
            self.logger.error(f"Error iniciando worker thread: {e}")
            raise
    
    def _http_worker(self):
        """Worker thread que procesa peticiones HTTP en segundo plano"""
        while self.worker_running:
            try:
                # Obtener petición de la cola (con timeout para poder salir)
                try:
                    request_data = self.request_queue.get(timeout=1.0)
                except Empty:
                    continue  # Continuar el loop si no hay peticiones
                
                # Procesar la petición
                payload, detection, current_time = request_data
                
                try:
                    # Enviar petición HTTP (esto es lo que antes bloqueaba)
                    response = self.traffic_control_client.send_traffic_data(payload)
                    
                    # Procesar respuesta
                    if response and response.get("status") == "success":
                        self.logger.info("Datos enviados exitosamente a traffic-control")
                        
                        # Procesar optimización si está disponible
                        optimization_data = self._extract_optimization_data(response)
                        if optimization_data:
                            # Aplicar optimización en el thread principal
                            self._apply_traffic_optimization(optimization_data)
                    else:
                        self.logger.warning("Error enviando datos a traffic-control")
                    
                    # Guardar en historial (thread-safe)
                    with self._history_lock:
                        self.bottleneck_history.append({
                            "timestamp": current_time,
                            "intersection_id": detection.intersection_id,
                            "severity": detection.severity,
                            "metrics": detection.metrics
                        })
                    
                except Exception as e:
                    self.logger.error(f"Error procesando petición HTTP: {e}")
                
                finally:
                    # Marcar la tarea como completada
                    self.request_queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"Error en worker thread: {e}")
                # Continuar procesando otras peticiones
    
    def _initialize_components(self):
        """Inicializa todos los componentes del sistema"""
        try:
            # Detector de cuellos de botella
            self.bottleneck_detector = BottleneckDetector()
            
            # Cliente de traffic-control
            self.traffic_control_client = TrafficControlClient()
            
            # Controlador de semáforos
            self.traffic_light_controller = TrafficLightController()
            
            # Iniciar worker thread para peticiones HTTP
            self._start_worker_thread()
            
            self.logger.info("Componentes inicializados")
            
        except Exception as e:
            self.logger.error(f"Error inicializando componentes: {e}")
            raise
    
    def run_simulation(self):
        """Ejecuta la simulación principal"""
        try:
            self.simulation_running = True
            self.logger.info("Iniciando simulación...")
            
            while self._should_stop_simulation():
                # Avanzar simulación
                traci.simulationStep()
                self.current_step += 1
                
                current_time = float(traci.simulation.getTime())
                
                # Detectar cuellos de botella
                if self._should_detect_bottlenecks():
                    self._handle_bottleneck_detection(current_time)
                
                # Pausa para no saturar
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            self.logger.info("Simulación interrumpida por el usuario")
        except Exception as e:
            self.logger.error(f"Error durante la simulación: {e}")
        finally:
            self._cleanup()
    
    def _should_stop_simulation(self) -> bool:
        """Determina si la simulación debe continuar (retorna True para continuar)"""
        try:
            # Verificar si hay vehículos esperados (incluyendo los que aún no se han insertado)
            # Esto es más confiable que getIDCount() que solo cuenta vehículos ya insertados
            expected_vehicles = int(traci.simulation.getMinExpectedNumber())
            
            # Verificar tiempo de simulación
            current_time = float(traci.simulation.getTime())
            # Continue if vehicles expected, not reached time limit, and not reached sim_steps
            cond = expected_vehicles > 0 and current_time < self.end_time
            if not cond:
                return False
            if hasattr(self, 'sim_steps') and self.sim_steps is not None:
                if self.current_step >= self.sim_steps:
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
            # Detectar cuellos de botella
            detections = self.bottleneck_detector.detect_bottlenecks()
            
            if detections:
                self.logger.info(f"Se detectaron {len(detections)} cuellos de botella")
                
                # Procesar cada detección
                for detection in detections:
                    self._process_bottleneck_detection(detection, current_time)
                    
                    # Obtener nombres descriptivos
                    intersection_name = descriptive_names.get_intersection_name(detection.intersection_id)
                    controlled_streets = [descriptive_names.get_edge_name(edge) for edge in self.bottleneck_detector.intersection_edges.get(detection.traffic_light_id, [])]
                    
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
                    print(f"{Colors.RED}Tiempo: {current_time:.0f}s{Colors.END}")
                    print(f"{Colors.RED}{'='*50}{Colors.END}")
            
            # Actualizar paso de detección
            self.last_detection_step = self.current_step
            
        except Exception as e:
            self.logger.error(f"Error en detección de cuellos de botella: {e}")
    
    def _process_bottleneck_detection(self, detection: BottleneckDetection, current_time: float):
        """Procesa una detección de cuello de botella (versión no bloqueante)"""
        try:
            if self.enable_dynamic_optimization:
                # Modo cluster: obtener semáforos cercanos y optimizar juntos
                self._process_cluster_optimization(detection, current_time)
            else:
                # Modo legacy: enviar solo datos del semáforo con cuello de botella
                payload = self._create_traffic_payload(detection, current_time)
                self.request_queue.put((payload, detection, current_time))
                self.logger.info(f"Petición agregada a cola para semáforo {detection.traffic_light_id}")

        except Exception as e:
            self.logger.error(f"Error procesando detección: {e}")

    def _process_cluster_optimization(self, detection: BottleneckDetection, current_time: float):
        """
        Procesa optimización de cluster: obtiene semáforos cercanos,
        envía datos a la API y aplica optimizaciones inmediatamente.
        """
        try:
            primary_tl_id = detection.traffic_light_id

            # Obtener semáforos cercanos
            nearby_tls = self._get_nearby_traffic_lights(primary_tl_id)
            self.logger.info(f"Cluster de {len(nearby_tls)} semáforos para optimización")

            # Crear payload con datos de todo el cluster
            cluster_payload = self._create_cluster_payload(primary_tl_id, nearby_tls, current_time)

            # Mostrar payload en consola
            print(f"\n{Colors.CYAN}{Colors.BOLD}PAYLOAD DE CLUSTER A TRAFFIC-CONTROL{Colors.END}")
            print(f"{Colors.CYAN}Semáforo primario: {primary_tl_id}{Colors.END}")
            print(f"{Colors.CYAN}Semáforos en cluster: {nearby_tls}{Colors.END}")
            print(f"{Colors.CYAN}{'='*50}{Colors.END}")
            print(json.dumps(cluster_payload.to_dict(), indent=2, ensure_ascii=False))
            print(f"{Colors.CYAN}{'='*50}{Colors.END}\n")

            # Enviar a la API (o mock)
            response = self.traffic_control_client.send_cluster_data(
                cluster_payload,
                use_mock=self.use_mock_api
            )

            # Aplicar optimizaciones inmediatamente
            self._apply_cluster_optimization(response)

            # Guardar en historial
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
    
    def _create_traffic_payload(self, detection: BottleneckDetection, current_time: float) -> TrafficDataPayload:
        """Crea el payload para traffic-control"""
        try:
            # Obtener edges controlados
            controlled_edges = self.bottleneck_detector.intersection_edges.get(detection.traffic_light_id, [])

            # Crear métricas (la normalización de densidad se hace en TrafficDataPayload.normalize())
            metrics = {
                'vehicles_per_minute': int(detection.metrics.get('vehicle_count', 0)),
                'avg_speed_kmh': float(detection.metrics.get('average_speed', 0.0)),
                'avg_circulation_time_sec': float(detection.metrics.get('avg_circulation_time_sec', 30.0)),
                'density': float(detection.metrics.get('density', 0.0)),
                'vehicle_stats': detection.metrics.get('vehicle_stats', {
                    'motorcycle': 0,
                    'car': int(detection.metrics.get('vehicle_count', 0)),
                    'bus': 0,
                    'truck': 0
                })
            }
            
            # Crear payload
            payload = self.traffic_control_client.create_traffic_payload(
                traffic_light_id=detection.traffic_light_id,
                controlled_edges=controlled_edges,
                metrics=metrics,
                timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            )
            
            # Imprimir en consola el payload que se enviará
            print(f"\n{Colors.CYAN}{Colors.BOLD}PAYLOAD A TRAFFIC-CONTROL{Colors.END}")
            print(f"{Colors.CYAN}{'='*50}{Colors.END}")
            print(json.dumps(payload.to_dict(), indent=2, ensure_ascii=False))
            print(f"{Colors.CYAN}{'='*50}{Colors.END}\n")
            
            return payload
            
        except Exception as e:
            self.logger.error(f"Error creando payload: {e}")
            raise
    
    def _extract_optimization_data(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extrae datos de optimización de la respuesta"""
        try:
            if "optimization" in response:
                return response["optimization"]
            return None
        except Exception as e:
            self.logger.error(f"Error extrayendo datos de optimización: {e}")
            return None
    
    def _apply_traffic_optimization(self, optimization_data: Dict[str, Any]):
        """Aplica optimización de tráfico para un solo semáforo (legacy)"""
        try:
            green_time = optimization_data.get("green_time_sec")
            red_time = optimization_data.get("red_time_sec")

            if green_time and red_time:
                # Aplicar usando TrafficLightController
                # (requiere conocer el tl_id, que debería venir en optimization_data)
                tl_id = optimization_data.get("traffic_light_id")
                if tl_id:
                    success = self.traffic_light_controller.update_traffic_light(
                        tl_id, {"optimization": optimization_data}
                    )
                    if success:
                        self.logger.info(f"Optimización aplicada a {tl_id}: green={green_time}s, red={red_time}s")
                    else:
                        self.logger.warning(f"No se pudo aplicar optimización a {tl_id}")
        except Exception as e:
            self.logger.error(f"Error aplicando optimización: {e}")

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

            # Obtener posición del semáforo primario
            try:
                primary_pos = traci.junction.getPosition(primary_tl_id)
            except traci.exceptions.TraCIException:
                # Si no se puede obtener posición, devolver solo el primario
                return [primary_tl_id]

            nearby = [primary_tl_id]

            for tl_id in all_tls:
                if tl_id == primary_tl_id:
                    continue
                try:
                    tl_pos = traci.junction.getPosition(tl_id)
                    # Calcular distancia euclidiana
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

    def _create_cluster_payload(
        self,
        primary_tl_id: str,
        nearby_tls: list,
        current_time: float
    ) -> ClusterDataPayload:
        """
        Crea un payload con datos de todos los semáforos del cluster.

        Args:
            primary_tl_id: ID del semáforo que detectó el cuello de botella
            nearby_tls: Lista de IDs de semáforos cercanos
            current_time: Tiempo actual de simulación

        Returns:
            ClusterDataPayload con métricas de todos los semáforos
        """
        sensors = []

        for tl_id in nearby_tls:
            # Obtener datos de intersección
            intersection_data = self.bottleneck_detector.get_intersection_data(tl_id)

            if intersection_data:
                # Normalizar densidad a 0-1
                raw_density = intersection_data.density
                normalized_density = min(raw_density / 100.0, 1.0)

                sensor_data = {
                    "traffic_light_id": tl_id,
                    "controlled_edges": self.bottleneck_detector.intersection_edges.get(tl_id, []),
                    "metrics": {
                        "vehicles_per_minute": intersection_data.vehicle_count,
                        "avg_speed_kmh": intersection_data.average_speed,
                        "avg_circulation_time_sec": intersection_data.avg_circulation_time,
                        "density": normalized_density
                    },
                    "vehicle_stats": intersection_data.vehicle_stats
                }
                sensors.append(sensor_data)

        return ClusterDataPayload(
            timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            primary_traffic_light_id=primary_tl_id,
            sensors=sensors
        )

    def _apply_cluster_optimization(self, response: ClusterOptimizationResponse):
        """
        Aplica las optimizaciones recibidas para todo el cluster de semáforos.

        Args:
            response: Respuesta con optimizaciones para cada semáforo
        """
        if response.status != "success":
            self.logger.warning(f"Optimización de cluster fallida: {response.message}")
            return

        applied_count = 0

        for opt in response.optimizations:
            if not opt.apply_immediately:
                continue

            try:
                # Usar TrafficLightController para aplicar cambios
                success = self.traffic_light_controller.update_traffic_light(
                    opt.traffic_light_id,
                    {
                        "optimization": {
                            "green_time_sec": opt.green_time_sec,
                            "red_time_sec": opt.red_time_sec
                        }
                    }
                )

                if success:
                    applied_count += 1
                    self.logger.info(
                        f"Optimización aplicada a {opt.traffic_light_id}: "
                        f"green={opt.green_time_sec}s, red={opt.red_time_sec}s"
                    )

            except Exception as e:
                self.logger.error(f"Error aplicando optimización a {opt.traffic_light_id}: {e}")

        if applied_count > 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}OPTIMIZACIÓN DE CLUSTER APLICADA{Colors.END}")
            print(f"{Colors.GREEN}Semáforos actualizados: {applied_count}/{len(response.optimizations)}{Colors.END}")
            if response.cluster_id:
                print(f"{Colors.GREEN}Cluster ID: {response.cluster_id}{Colors.END}")
            print(f"{Colors.GREEN}{'='*50}{Colors.END}\n")
    
    def _pause_simulation(self):
        """Pausa la simulación"""
        self.simulation_running = False
        self.logger.info("Simulación pausada")
    
    def _resume_simulation(self):
        """Reanuda la simulación"""
        self.simulation_running = True
        self.logger.info("Simulación reanudada")
    
    def _cleanup(self):
        """Limpia recursos de la simulación"""
        try:
            # Detener worker thread
            if hasattr(self, 'worker_running'):
                self.worker_running = False
                
                # Esperar a que el worker thread termine (máximo 5 segundos)
                if self.worker_thread and self.worker_thread.is_alive():
                    self.worker_thread.join(timeout=5.0)
                    if self.worker_thread.is_alive():
                        self.logger.warning("Worker thread no terminó en el tiempo esperado")
                    else:
                        self.logger.info("Worker thread terminado correctamente")
            
            # Verificar si traci está disponible y conectado
            if 'traci' in globals():
                try:
                    # Intentar cerrar la conexión
                    traci.close()
                except Exception:
                    # Si falla, la conexión ya estaba cerrada o no existe
                    pass
            self.logger.info("Recursos de simulación liberados")
        except Exception as e:
            self.logger.error(f"Error en limpieza: {e}")
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la simulación"""
        try:
            # Verificar conexión intentando obtener el tiempo
            try:
                current_time = float(traci.simulation.getTime())
                vehicle_count = traci.vehicle.getIDCount()
            except (AttributeError, RuntimeError, traci.exceptions.FatalTraCIError):
                # No conectado o error de conexión
                current_time = 0.0
                vehicle_count = 0
            
            # Obtener estadísticas de la cola de peticiones
            queue_size = self.request_queue.qsize() if hasattr(self, 'request_queue') else 0

            # Acceso thread-safe al historial
            with self._history_lock:
                history_len = len(self.bottleneck_history)
                history_copy = list(self.bottleneck_history)

            return {
                "current_time": current_time,
                "vehicle_count": vehicle_count,
                "bottleneck_detections": history_len,
                "detection_history": history_copy,
                "pending_requests": queue_size,
                "worker_thread_alive": self.worker_thread.is_alive() if hasattr(self, 'worker_thread') and self.worker_thread else False
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