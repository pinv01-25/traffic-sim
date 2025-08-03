"""
Orquestador principal de la simulación de tráfico
Coordina detección de cuellos de botella, comunicación con traffic-control y actualización de semáforos
"""

import os
import sys
import subprocess
import time
import traci
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

from utils.logger import get_simulation_logger, Colors
from utils.descriptive_names import descriptive_names
from config import SIMULATION_CONFIG, BOTTLENECK_CONFIG
from detectors.bottleneck_detector import BottleneckDetector, BottleneckDetection
from services.traffic_control_client import TrafficControlClient, TrafficDataPayload
from controllers.traffic_light_controller import TrafficLightController

class SimulationOrchestrator:
    """
    Orquestador principal de la simulación de tráfico
    Coordina todos los componentes del sistema
    """
    
    def __init__(self, simulation_dir: str = "simulation"):
        self.simulation_dir = Path(simulation_dir)
        self.logger = get_simulation_logger()
        
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
            
            # Iniciar SUMO con TraCI
            traci.start([
                "sumo",
                "-c", str(config_file),
                "--no-step-log", "true",
                "--time-to-teleport", "-1"
            ])
            
            self.logger.info("Conexión TraCI establecida")
            return True
            
        except Exception as e:
            self.logger.error(f"Error iniciando TraCI: {e}")
            return False
    
    def _initialize_components(self):
        """Inicializa todos los componentes del sistema"""
        try:
            # Detector de cuellos de botella
            self.bottleneck_detector = BottleneckDetector()
            
            # Cliente de traffic-control
            self.traffic_control_client = TrafficControlClient()
            
            # Controlador de semáforos
            self.traffic_light_controller = TrafficLightController()
            
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
        """Determina si la simulación debe continuar"""
        try:
            # Verificar si hay vehículos en el sistema
            vehicle_count = traci.vehicle.getIDCount()
            
            # Verificar tiempo de simulación
            current_time = float(traci.simulation.getTime())
            
            # Continuar si hay vehículos y no se ha alcanzado el tiempo límite
            return vehicle_count > 0 and current_time < self.end_time
            
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
        """Procesa una detección de cuello de botella"""
        try:
            # Crear payload para traffic-control
            payload = self._create_traffic_payload(detection, current_time)
            
            # Enviar a traffic-control
            response = self.traffic_control_client.send_traffic_data(payload)
            
            if response and response.get("status") == "success":
                self.logger.info("Datos enviados exitosamente a traffic-control")
                
                # Procesar optimización si está disponible
                optimization_data = self._extract_optimization_data(response)
                if optimization_data:
                    self._apply_traffic_optimization(optimization_data)
            else:
                self.logger.warning("Error enviando datos a traffic-control")
            
            # Guardar en historial
            self.bottleneck_history.append({
                "timestamp": current_time,
                "intersection_id": detection.intersection_id,
                "severity": detection.severity,
                "metrics": detection.metrics
            })
            
        except Exception as e:
            self.logger.error(f"Error procesando detección: {e}")
    
    def _create_traffic_payload(self, detection: BottleneckDetection, current_time: float) -> TrafficDataPayload:
        """Crea el payload para traffic-control"""
        try:
            # Obtener edges controlados
            controlled_edges = self.bottleneck_detector.intersection_edges.get(detection.traffic_light_id, [])
            
            # Crear métricas
            metrics = {
                'vehicles_per_minute': int(detection.metrics.get('vehicle_count', 0)),
                'avg_speed_kmh': float(detection.metrics.get('average_speed', 0.0)),
                'avg_circulation_time_sec': float(detection.metrics.get('avg_circulation_time_sec', 30.0)),
                'density': float(detection.metrics.get('density', 0.0)),
                'vehicle_stats': {
                    'motorcycle': 0,
                    'car': int(detection.metrics.get('vehicle_count', 0)),
                    'bus': 0,
                    'truck': 0
                }
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
        """Aplica optimización de tráfico"""
        try:
            # Implementar lógica de optimización aquí
            self.logger.info("Aplicando optimización de tráfico")
            # TODO: Implementar optimización real
        except Exception as e:
            self.logger.error(f"Error aplicando optimización: {e}")
    
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
            if traci.isConnected():
                traci.close()
            self.logger.info("Recursos de simulación liberados")
        except Exception as e:
            self.logger.error(f"Error en limpieza: {e}")
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la simulación"""
        try:
            current_time = float(traci.simulation.getTime()) if traci.isConnected() else 0.0
            vehicle_count = traci.vehicle.getIDCount() if traci.isConnected() else 0
            
            return {
                "current_time": current_time,
                "vehicle_count": vehicle_count,
                "bottleneck_detections": len(self.bottleneck_history),
                "detection_history": self.bottleneck_history
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