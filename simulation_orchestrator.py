"""
Orquestador principal de la simulación de tráfico
Coordina detección de cuellos de botella, comunicación con traffic-control y actualización de semáforos
"""

import os
import sys
import subprocess
import time
import traci
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from utils.logger import get_simulation_logger
from config import SIMULATION_CONFIG, BOTTLENECK_CONFIG
from detectors.bottleneck_detector import BottleneckDetector, BottleneckDetection
from services.traffic_control_client import TrafficControlClient, TrafficDataPayload
from controllers.traffic_light_controller import TrafficLightController

class SimulationOrchestrator:
    """
    Orquestador principal de la simulación de tráfico
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
        self.last_detection_time = 0
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
            result = subprocess.run(["sumo", "--version"], 
                                  capture_output=True, text=True, check=True)
            self.logger.info(f"SUMO encontrado: {result.stdout.split()[1]}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _generate_network(self) -> bool:
        """Genera la red de simulación usando netconvert"""
        try:
            network_file = self.simulation_dir / "network.net.xml"
            
            if network_file.exists():
                self.logger.info("Red de simulación ya existe")
                return True
            
            self.logger.info("Generando red con netconvert...")
            
            netconvert_cmd = [
                "netconvert",
                "--node-files", "nodes.nod.xml",
                "--edge-files", "edges.edg.xml",
                "--output-file", "network.net.xml",
                "--no-turnarounds",
                "--tls.guess", "true"
            ]
            
            result = subprocess.run(netconvert_cmd, 
                                  cwd=self.simulation_dir, 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Error generando red: {result.stderr}")
                return False
            
            self.logger.info("Red generada exitosamente")
            return True
            
        except Exception as e:
            self.logger.error(f"Error en generación de red: {e}")
            return False
    
    def _start_traci_connection(self) -> bool:
        """Inicia la conexión traci con SUMO"""
        try:
            # Configurar archivo de configuración
            config_file = self.simulation_dir / "simulation.sumocfg"
            
            if not config_file.exists():
                self.logger.error(f"Archivo de configuración no encontrado: {config_file}")
                return False
            
            # Iniciar SUMO con traci
            sumo_cmd = [
                "sumo",
                "-c", str(config_file),
                "--no-step-log", "true",
                "--no-warnings", "true",
                "--time-to-teleport", "-1"
            ]
            
            # Iniciar proceso SUMO
            self.sumo_process = subprocess.Popen(
                sumo_cmd,
                cwd=self.simulation_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Esperar un momento para que SUMO inicie
            time.sleep(2)
            
            # Conectar traci
            traci.connect(port=8813)
            
            self.logger.info("Conexión traci establecida")
            return True
            
        except Exception as e:
            self.logger.error(f"Error iniciando conexión traci: {e}")
            return False
    
    def _initialize_components(self):
        """Inicializa todos los componentes del sistema"""
        try:
            # Inicializar detector de cuellos de botella
            self.bottleneck_detector = BottleneckDetector()
            self.logger.info("Detector de cuellos de botella inicializado")
            
            # Inicializar cliente de traffic-control
            self.traffic_control_client = TrafficControlClient()
            
            # Verificar conectividad con traffic-control
            if self.traffic_control_client.health_check():
                self.logger.info("Conexión con traffic-control establecida")
            else:
                self.logger.warning("No se pudo conectar con traffic-control")
            
            # Inicializar controlador de semáforos
            self.traffic_light_controller = TrafficLightController()
            self.logger.info("Controlador de semáforos inicializado")
            
        except Exception as e:
            self.logger.error(f"Error inicializando componentes: {e}")
            raise
    
    def run_simulation(self):
        """Ejecuta la simulación completa"""
        try:
            self.logger.info("Iniciando simulación de tráfico...")
            print("🚀 INICIANDO SIMULACIÓN - Debug output activado")
            self.simulation_running = True
            
            # Bucle principal de simulación
            while self.simulation_running:
                # Verificar condiciones de terminación
                if self._should_stop_simulation():
                    break
                
                # Ejecutar paso de simulación
                traci.simulationStep()
                current_time = float(traci.simulation.getTime())
                
                # Detectar cuellos de botella periódicamente
                if self._should_detect_bottlenecks(current_time):
                    print(f"\n🔍 DETECCIÓN PROGRAMADA EN TIEMPO {current_time:.0f}s")
                    self._handle_bottleneck_detection(current_time)
                
                # Log de progreso más frecuente para debug
                if int(current_time) % 10 == 0 and current_time > 0:  # Cada 10 segundos
                    vehicle_count = traci.vehicle.getIDCount()
                    print(f"⏰ Tiempo: {current_time:.0f}s | Vehículos: {vehicle_count}")
                    self.logger.info(f"Tiempo de simulación: {current_time:.0f}s")
            
            self.logger.info("Simulación completada")
            print("✅ SIMULACIÓN COMPLETADA")
            
        except Exception as e:
            self.logger.error(f"Error en simulación: {e}")
        finally:
            self._cleanup()
    
    def _should_stop_simulation(self) -> bool:
        """Determina si la simulación debe detenerse"""
        try:
            current_time = float(traci.simulation.getTime())
            
            # Verificar tiempo límite
            if current_time >= self.end_time:
                self.logger.info(f"Simulación terminada por tiempo límite: {current_time}s")
                return True
            
            # Verificar si no hay vehículos
            vehicle_count = traci.vehicle.getIDCount()
            if vehicle_count == 0:
                self.logger.info("Simulación terminada: no hay vehículos en el sistema")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error verificando condiciones de parada: {e}")
            return True
    
    def _should_detect_bottlenecks(self, current_time: float) -> bool:
        """Determina si es momento de detectar cuellos de botella"""
        return (current_time - self.last_detection_time) >= self.detection_interval
    
    def _handle_bottleneck_detection(self, current_time: float):
        """Maneja la detección de cuellos de botella"""
        try:
            self.logger.info(f"Detectando cuellos de botella en tiempo {current_time}")
            
            # Detectar cuellos de botella
            detections = self.bottleneck_detector.detect_bottlenecks()
            
            if detections:
                self.logger.info(f"Se detectaron {len(detections)} cuellos de botella")
                
                # Imprimir información detallada de cada detección
                for detection in detections:
                    print(f"\n🚨 CUELLO DE BOTELLA DETECTADO 🚨")
                    print(f"📍 Intersección: {detection.intersection_id}")
                    print(f"🚦 Semáforo: {detection.traffic_light_id}")
                    print(f"⚠️  Severidad: {detection.severity.upper()}")
                    print(f"📊 Métricas:")
                    print(f"   • Vehículos: {detection.metrics['vehicle_count']}")
                    print(f"   • Velocidad promedio: {detection.metrics['average_speed']:.1f} m/s")
                    print(f"   • Densidad: {detection.metrics['density']:.1f} veh/km")
                    print(f"   • Cola: {detection.metrics['queue_length']} vehículos")
                    print(f"⏰ Tiempo: {current_time:.0f}s")
                    print("=" * 50)
                
                # Procesar cada detección
                for detection in detections:
                    self._process_bottleneck_detection(detection, current_time)
            else:
                # Mostrar información de debug para entender por qué no se detectan
                if current_time % 60 == 0:  # Cada minuto
                    print(f"\n🔍 DEBUG - Tiempo: {current_time:.0f}s")
                    for intersection_id in self.bottleneck_detector.intersection_edges:
                        status = self.bottleneck_detector.get_intersection_status(intersection_id)
                        if "error" not in status:
                            print(f"📍 {intersection_id}: {status['vehicle_count']} veh, "
                                  f"{status['average_speed']:.1f} m/s, "
                                  f"{status['density']:.1f} veh/km, "
                                  f"{status['queue_length']} en cola")
                    print("=" * 30)
            
            self.last_detection_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error en detección de cuellos de botella: {e}")
    
    def _process_bottleneck_detection(self, detection: BottleneckDetection, current_time: float):
        """Procesa una detección de cuello de botella"""
        try:
            self.logger.info(f"Procesando cuello de botella en {detection.intersection_id}")
            
            # Pausar simulación
            self._pause_simulation()
            
            # Obtener datos de la intersección
            intersection_data = self.bottleneck_detector.get_intersection_data(detection.intersection_id)
            
            if intersection_data is None:
                self.logger.error(f"No se pudieron obtener datos de {detection.intersection_id}")
                self._resume_simulation()
                return
            
            # Crear payload para traffic-control
            payload = self.traffic_control_client.create_traffic_payload(
                traffic_light_id=detection.traffic_light_id,
                controlled_edges=intersection_data.edges,
                vehicle_count=intersection_data.vehicle_count,
                average_speed=intersection_data.average_speed,
                density=intersection_data.density,
                queue_length=intersection_data.queue_length,
                timestamp=datetime.now().isoformat()
            )
            
            # Enviar a traffic-control
            self.logger.info(f"Enviando datos a traffic-control para {detection.traffic_light_id}")
            response = self.traffic_control_client.send_traffic_data(payload)
            
            # Actualizar semáforo con datos optimizados
            if response and response.get("status") == "success":
                self.logger.info("Datos procesados exitosamente, actualizando semáforo...")
                
                # Extraer datos de optimización de la respuesta
                # Nota: La respuesta real dependerá de cómo traffic-control devuelve los datos optimizados
                optimization_data = self._extract_optimization_data(response)
                
                if optimization_data:
                    success = self.traffic_light_controller.update_traffic_light(
                        detection.traffic_light_id, 
                        optimization_data
                    )
                    
                    if success:
                        self.logger.info(f"Semáforo {detection.traffic_light_id} actualizado exitosamente")
                    else:
                        self.logger.error(f"Error actualizando semáforo {detection.traffic_light_id}")
                else:
                    self.logger.warning("No se encontraron datos de optimización en la respuesta")
            else:
                self.logger.error("Error procesando datos en traffic-control")
            
            # Reanudar simulación
            self._resume_simulation()
            
            # Registrar detección
            self.bottleneck_history.append({
                "timestamp": current_time,
                "intersection_id": detection.intersection_id,
                "severity": detection.severity,
                "processed": True
            })
            
        except Exception as e:
            self.logger.error(f"Error procesando detección: {e}")
            self._resume_simulation()
    
    def _extract_optimization_data(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extrae datos de optimización de la respuesta de traffic-control
        
        Args:
            response: Respuesta de traffic-control
            
        Returns:
            Datos de optimización o None si no se encuentran
        """
        try:
            # La estructura exacta dependerá de cómo traffic-control devuelve los datos
            # Por ahora, asumimos una estructura básica
            if "optimization" in response:
                return response["optimization"]
            elif "data" in response and "optimization" in response["data"]:
                return response["data"]["optimization"]
            else:
                # Crear datos de optimización por defecto basados en la severidad
                return {
                    "optimization": {
                        "green_time_sec": 45,
                        "red_time_sec": 45
                    },
                    "impact": {
                        "original_congestion": 0,
                        "optimized_congestion": 0,
                        "original_category": "medium",
                        "optimized_category": "low"
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error extrayendo datos de optimización: {e}")
            return None
    
    def _pause_simulation(self):
        """Pausa la simulación momentáneamente"""
        self.logger.info("Pausando simulación para procesamiento...")
        # En una implementación real, aquí se pausaría SUMO
        # Por ahora, solo registramos la pausa
    
    def _resume_simulation(self):
        """Reanuda la simulación"""
        self.logger.info("Reanudando simulación...")
        # En una implementación real, aquí se reanudaría SUMO
        # Por ahora, solo registramos la reanudación
    
    def _cleanup(self):
        """Limpia recursos al finalizar la simulación"""
        try:
            self.logger.info("Limpiando recursos...")
            
            # Cerrar conexión traci
            if traci.isConnected():
                traci.close()
            
            # Terminar proceso SUMO
            if hasattr(self, 'sumo_process'):
                self.sumo_process.terminate()
                self.sumo_process.wait()
            
            self.simulation_running = False
            self.logger.info("Limpieza completada")
            
        except Exception as e:
            self.logger.error(f"Error en limpieza: {e}")
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la simulación"""
        try:
            current_time = float(traci.simulation.getTime())
            vehicle_count = traci.vehicle.getIDCount()
            
            return {
                "current_time": current_time,
                "vehicle_count": vehicle_count,
                "bottleneck_detections": len(self.bottleneck_history),
                "simulation_running": self.simulation_running,
                "detection_history": self.bottleneck_history
            }
            
        except Exception as e:
            self.logger.error(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}

def main():
    """Función principal para ejecutar la simulación"""
    # Configurar directorio de simulación
    simulation_dir = "simulation"
    
    # Crear orquestador
    orchestrator = SimulationOrchestrator(simulation_dir)
    
    try:
        # Configurar simulación
        if not orchestrator.setup_simulation():
            print("Error configurando simulación")
            return
        
        # Ejecutar simulación
        orchestrator.run_simulation()
        
        # Mostrar estadísticas finales
        stats = orchestrator.get_simulation_stats()
        print(f"\nEstadísticas de simulación:")
        print(f"Tiempo total: {stats.get('current_time', 0):.0f}s")
        print(f"Detectores de cuellos de botella: {stats.get('bottleneck_detections', 0)}")
        
    except KeyboardInterrupt:
        print("\nSimulación interrumpida por el usuario")
    except Exception as e:
        print(f"Error en simulación: {e}")
    finally:
        orchestrator._cleanup()

if __name__ == "__main__":
    main() 