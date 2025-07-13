"""
Detector de cuellos de botella para intersecciones con semáforos
"""

import traci
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from utils.logger import setup_logger
from config import BOTTLENECK_CONFIG
import os
from datetime import datetime

logger = setup_logger(__name__)

# Configuración para logging a archivo
LOG_FILE = "logs/bottleneck_detection.log"
os.makedirs("logs", exist_ok=True)

def log_to_file(message: str):
    """Escribe mensaje al archivo de log con timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "w") as f:  # 'w' para sobrescribir cada vez
        f.write(f"[{timestamp}] {message}\n")

@dataclass
class IntersectionData:
    intersection_id: str
    edges: List[str]
    traffic_light_id: str
    vehicle_count: int
    average_speed: float
    density: float  # vehículos por kilómetro
    queue_length: int
    timestamp: float

@dataclass
class BottleneckDetection:
    intersection_id: str
    traffic_light_id: str
    severity: str  # 'low', 'medium', 'high'
    metrics: Dict[str, float]
    timestamp: float

class BottleneckDetector:
    def __init__(self):
        self.detection_history: Dict[str, List[BottleneckDetection]] = {}
        self.intersection_edges: Dict[str, List[str]] = {}
        self.visible_range = 100.0  # metros visibles para detección
        self._initialize_intersections()

    def _initialize_intersections(self):
        try:
            traffic_lights = traci.trafficlight.getIDList()
            all_edges = set(traci.edge.getIDList())
            log_to_file(f"Semáforos encontrados: {traffic_lights}")

            for tl_id in traffic_lights:
                try:
                    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                    controlled_edges = set()
                    for lane in controlled_lanes:
                        # Extraer edge real del lane de forma robusta
                        if "_" in lane:
                            edge_candidate = lane.rsplit("_", 1)[0]
                            if edge_candidate in all_edges:
                                controlled_edges.add(edge_candidate)
                    if controlled_edges:
                        self.intersection_edges[tl_id] = list(controlled_edges)
                        log_to_file(f"Semáforo {tl_id} controla edges: {list(controlled_edges)}")
                    else:
                        log_to_file(f"Semáforo {tl_id} no controla edges válidos")
                except Exception as e:
                    log_to_file(f"Error obteniendo edges para semáforo {tl_id}: {e}")
        except Exception as e:
            log_to_file(f"Error inicializando intersecciones: {e}")

    def _get_intersection_for_traffic_light(self, tl_id: str) -> str:
        return tl_id

    def get_intersection_data(self, intersection_id: str) -> Optional[IntersectionData]:
        if intersection_id not in self.intersection_edges:
            return None
        edges = self.intersection_edges[intersection_id]
        traffic_light_id = intersection_id
        vehicles = []
        total_speed = 0.0
        total_length = 0.0
        
        # Debug: imprimir información de cada edge
        # log_to_file(f"\n🔍 DEBUG - Intersección {intersection_id}:")
        
        for edge_id in edges:
            try:
                # Obtener vehículos visibles en el rango especificado
                visible_vehicles = self._get_visible_vehicles(edge_id)
                vehicles.extend(visible_vehicles)
                
                if visible_vehicles:
                    edge_speed = traci.edge.getLastStepMeanSpeed(edge_id)
                    # Robustecer: asegurar que edge_speed sea float y no tupla/lista
                    if isinstance(edge_speed, (list, tuple)):
                        edge_speed = float(edge_speed[0]) if edge_speed else 0.0
                    else:
                        try:
                            edge_speed = float(edge_speed)
                        except Exception:
                            edge_speed = 0.0
                    
                    # getLength: sumar la longitud de todos los lanes del edge
                    try:
                        lane_ids = traci.edge.getLaneIDs(edge_id)
                        if not isinstance(lane_ids, (list, tuple)):
                            lane_ids = [lane_ids]
                        edge_length = 0.0
                        for lane in lane_ids:
                            lane_length = traci.lane.getLength(lane)
                            if isinstance(lane_length, (list, tuple)):
                                lane_length = float(lane_length[0]) if lane_length else 0.0
                            else:
                                try:
                                    lane_length = float(lane_length)
                                except Exception:
                                    lane_length = 0.0
                            edge_length += lane_length
                        if edge_length == 0.0:
                            edge_length = 100.0  # valor por defecto si no hay lanes
                    except Exception:
                        edge_length = 100.0  # valor por defecto si no está disponible
                    
                    if edge_length > 0:
                        total_speed += edge_speed * edge_length
                        total_length += edge_length
                
                # Debug: información de cada edge (eliminado)
                # log_to_file(f"   🛣️  {edge_id}: {len(edge_vehicles)} veh, {edge_speed:.1f} m/s, {edge_length:.0f}m")
                
            except Exception as e:
                logger.warning(f"Error obteniendo datos de edge {edge_id}: {e}")
        
        current_time = float(traci.simulation.getTime())
        if not vehicles or total_length == 0.0:
            log_to_file(f"   ❌ Sin vehículos o longitud total = 0")
            return IntersectionData(
                intersection_id=intersection_id,
                edges=edges,
                traffic_light_id=traffic_light_id,
                vehicle_count=0,
                average_speed=0.0,
                density=0.0,
                queue_length=0,
                timestamp=current_time
            )
        
        vehicle_count = len(vehicles)
        
        # Robustecer: asegurar que los valores sean float
        try:
            if isinstance(total_speed, (list, tuple)):
                total_speed = float(total_speed[0]) if total_speed else 0.0
            else:
                total_speed = float(total_speed)
        except Exception:
            total_speed = 0.0
        try:
            if isinstance(total_length, (list, tuple)):
                total_length = float(total_length[0]) if total_length else 0.0
            else:
                total_length = float(total_length)
        except Exception:
            total_length = 0.0
        try:
            vehicle_count = int(vehicle_count)
        except Exception:
            vehicle_count = 0
        
        # Calcular densidad de manera más precisa usando rango visible
        visible_length_km = min(self.visible_range, total_length) / 1000.0
        average_speed = total_speed / total_length if total_length > 0 else 0.0
        
        # Densidad: vehículos por kilómetro en el rango visible
        density = vehicle_count / visible_length_km if visible_length_km > 0 else 0.0
        
        # Calcular cola
        queue_length = self._calculate_queue_length(edges)
        
        # Debug: información final
        log_to_file(f"   📊 Total: {vehicle_count} veh, {average_speed:.1f} m/s, {visible_length_km:.3f} km")
        log_to_file(f"   🚗 Densidad: {density:.1f} veh/km (umbral: {BOTTLENECK_CONFIG['density_threshold']})")
        log_to_file(f"   🚦 Cola: {queue_length} veh (umbral: {BOTTLENECK_CONFIG['queue_length_threshold']})")
        log_to_file(f"   ⚡ Velocidad: {average_speed:.1f} m/s (umbral: {BOTTLENECK_CONFIG['speed_threshold']})")
        
        return IntersectionData(
            intersection_id=intersection_id,
            edges=edges,
            traffic_light_id=traffic_light_id,
            vehicle_count=vehicle_count,
            average_speed=average_speed,
            density=density,
            queue_length=queue_length,
            timestamp=current_time
        )

    def _get_visible_vehicles(self, edge_id: str) -> List[str]:
        """Obtiene solo los vehículos visibles en el rango especificado"""
        visible_vehicles = []
        try:
            # Obtener el primer carril como referencia
            lane_id = edge_id + "_0"
            lane_length = traci.lane.getLength(lane_id)
            if isinstance(lane_length, (list, tuple)):
                lane_length = float(lane_length[0]) if lane_length else 0.0
            else:
                try:
                    lane_length = float(lane_length)
                except Exception:
                    lane_length = 0.0
            for v_id in traci.edge.getLastStepVehicleIDs(edge_id):
                try:
                    v_lane = traci.vehicle.getLaneID(v_id)
                    if isinstance(v_lane, (list, tuple)):
                        v_lane = str(v_lane[0]) if v_lane else ""
                    if isinstance(edge_id, (list, tuple)):
                        edge_id_str = str(edge_id[0]) if edge_id else ""
                    else:
                        edge_id_str = str(edge_id)
                    if v_lane.startswith(edge_id_str):  # Asegurarse de que el vehículo está en la misma edge
                        pos = traci.vehicle.getLanePosition(v_id)
                        if isinstance(pos, (list, tuple)):
                            pos = float(pos[0]) if pos else 0.0
                        else:
                            try:
                                pos = float(pos)
                            except Exception:
                                pos = 0.0
                        # Comprobar si el vehículo está dentro del rango visible
                        try:
                            if (lane_length - pos) <= self.visible_range:
                                visible_vehicles.append(v_id)
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            # Si hay error, usar todos los vehículos como fallback
            try:
                vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
                if isinstance(vehicles, (list, tuple)):
                    visible_vehicles = list(vehicles)
                else:
                    visible_vehicles = [vehicles]
            except Exception:
                visible_vehicles = []
        return visible_vehicles

    def _calculate_queue_length(self, edges: List[str]) -> int:
        queue_length = 0
        for edge_id in edges:
            try:
                vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
                for vehicle_id in vehicles:
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    # Robustecer: asegurar que speed sea float
                    if isinstance(speed, (list, tuple)):
                        speed = float(speed[0]) if speed else 0.0
                    else:
                        try:
                            speed = float(speed)
                        except Exception:
                            speed = 0.0
                    if speed < 1.0:
                        queue_length += 1
            except Exception as e:
                logger.warning(f"Error calculando cola en edge {edge_id}: {e}")
        return queue_length

    def detect_bottlenecks(self) -> List[BottleneckDetection]:
        detections = []
        for intersection_id in self.intersection_edges:
            intersection_data = self.get_intersection_data(intersection_id)
            if intersection_data is None:
                continue
            bottleneck = self._analyze_bottleneck(intersection_data)
            if bottleneck:
                detections.append(bottleneck)
                if intersection_id not in self.detection_history:
                    self.detection_history[intersection_id] = []
                self.detection_history[intersection_id].append(bottleneck)
                log_to_file(f"Cuello de botella detectado en {intersection_id}: {bottleneck.severity}")
        return detections

    def _analyze_bottleneck(self, data: IntersectionData) -> Optional[BottleneckDetection]:
        density_threshold = BOTTLENECK_CONFIG["density_threshold"]
        speed_threshold = BOTTLENECK_CONFIG["speed_threshold"]
        queue_threshold = BOTTLENECK_CONFIG["queue_length_threshold"]
        
        # Verificar que realmente hay vehículos antes de analizar
        if data.vehicle_count == 0:
            return None
        
        # Lógica mejorada basada en el enfoque de referencia
        density_ok = data.density > density_threshold
        speed_ok = data.average_speed < speed_threshold
        queue_ok = data.queue_length > queue_threshold
        
        # Debug: mostrar qué criterios se cumplen
        log_to_file(f"   🔍 Análisis para {data.intersection_id}:")
        log_to_file(f"      • Vehículos: {data.vehicle_count}")
        log_to_file(f"      • Densidad: {data.density:.1f} > {density_threshold} = {density_ok}")
        log_to_file(f"      • Velocidad: {data.average_speed:.1f} < {speed_threshold} = {speed_ok}")
        log_to_file(f"      • Cola: {data.queue_length} > {queue_threshold} = {queue_ok}")
        
        # Detectar si hay congestión REAL (al menos 2 criterios Y vehículos suficientes)
        if data.vehicle_count >= 3:  # Mínimo 3 vehículos para considerar congestión
            if density_ok and (speed_ok or queue_ok):
                severity = "high" if data.density > density_threshold * 1.5 else "medium"
            elif density_ok or (speed_ok and queue_ok):
                severity = "low"
            else:
                log_to_file(f"   ❌ No se cumple ningún criterio de detección")
                return None
        else:
            log_to_file(f"   ❌ Muy pocos vehículos ({data.vehicle_count}) para considerar congestión")
            return None
        
        log_to_file(f"   ✅ Detección: {severity.upper()}")
        
        if not self._check_detection_duration(data.intersection_id, severity):
            log_to_file(f"   ⏰ Duración insuficiente para {severity}")
            return None
        
        metrics = {
            "vehicle_count": data.vehicle_count,
            "average_speed": data.average_speed,
            "density": data.density,
            "queue_length": data.queue_length
        }
        return BottleneckDetection(
            intersection_id=data.intersection_id,
            traffic_light_id=data.traffic_light_id,
            severity=severity,
            metrics=metrics,
            timestamp=data.timestamp
        )

    def _check_detection_duration(self, intersection_id: str, severity: str) -> bool:
        if intersection_id not in self.detection_history:
            return True
        history = self.detection_history[intersection_id]
        current_time = float(traci.simulation.getTime())
        min_duration = BOTTLENECK_CONFIG["min_detection_duration"]
        for detection in reversed(history):
            if detection.severity == severity:
                time_diff = current_time - float(detection.timestamp)
                return time_diff >= min_duration
        return True

    def get_intersection_status(self, intersection_id: str) -> Dict:
        data = self.get_intersection_data(intersection_id)
        if data is None:
            return {"error": "Intersección no encontrada"}
        return {
            "intersection_id": data.intersection_id,
            "traffic_light_id": data.traffic_light_id,
            "vehicle_count": data.vehicle_count,
            "average_speed": data.average_speed,
            "density": data.density,
            "queue_length": data.queue_length,
            "timestamp": data.timestamp
        } 