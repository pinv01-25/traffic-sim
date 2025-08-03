"""
Detector de cuellos de botella para intersecciones con semáforos
"""

import traci
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from utils.logger import setup_logger
from utils.descriptive_names import descriptive_names
from config import BOTTLENECK_CONFIG
from services.metrics_calculator import MetricsCalculator, TrafficMetrics
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
    avg_circulation_time: float  # tiempo promedio de circulación en segundos
    vehicle_stats: Dict[str, int]  # Clasificación por tipo de vehículo
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
        self.metrics_calculator = MetricsCalculator()
        self._initialize_intersections()

    def _initialize_intersections(self):
        try:
            traffic_lights = traci.trafficlight.getIDList()
            all_edges = set(traci.edge.getIDList())
            logger.info(f"Semáforos encontrados: {traffic_lights}")

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
                        logger.info(f"Semáforo {tl_id} controla edges: {list(controlled_edges)}")
                    else:
                        logger.warning(f"Semáforo {tl_id} no controla edges válidos")
                except Exception as e:
                    logger.error(f"Error obteniendo edges para semáforo {tl_id}: {e}")
        except Exception as e:
            logger.error(f"Error inicializando intersecciones: {e}")

    def _get_intersection_for_traffic_light(self, tl_id: str) -> str:
        return tl_id

    def get_intersection_data(self, intersection_id: str) -> Optional[IntersectionData]:
        if intersection_id not in self.intersection_edges:
            return None
        
        edges = self.intersection_edges[intersection_id]
        traffic_light_id = intersection_id
        current_time = float(traci.simulation.getTime())
        
        # Limpiar tracking antiguo
        self.metrics_calculator.cleanup_old_tracking(current_time)
        
        # Calcular métricas agregadas para toda la intersección
        total_vehicles = 0
        total_speed = 0.0
        total_waiting_time = 0.0
        total_vehicles_per_minute = 0
        total_circulation_time = 0.0
        all_visible_vehicles = []
        vehicle_stats: Dict[str, int] = {}
        
        # Debug logging solo para archivo (no consola)
        intersection_name = descriptive_names.get_intersection_name(intersection_id)
        log_to_file(f"\nDEBUG - Intersección {intersection_name} ({intersection_id}):")
        
        for edge_id in edges:
            try:
                # Obtener nombre descriptivo al inicio para evitar errores de scope
                edge_name = descriptive_names.get_edge_name(edge_id)
                
                # Calcular métricas para este edge usando el nuevo calculador
                edge_metrics = self.metrics_calculator.calculate_metrics(edge_id, current_time)
                
                # Acumular métricas
                total_vehicles += edge_metrics.vehicle_count
                total_speed += edge_metrics.avg_speed_kmh * edge_metrics.vehicle_count  # Ponderado por vehículos
                total_waiting_time += edge_metrics.avg_circulation_time_sec * edge_metrics.vehicle_count
                total_vehicles_per_minute += edge_metrics.vehicles_per_minute
                total_circulation_time += edge_metrics.avg_circulation_time_sec * edge_metrics.vehicle_count
                
                # Obtener vehículos visibles para densidad
                visible_vehicles = self.metrics_calculator.get_visible_vehicles(edge_id)
                all_visible_vehicles.extend(visible_vehicles)
                
                # Acumular estadísticas de vehículos por tipo
                for vehicle_type, count in edge_metrics.vehicle_stats.items():
                    if vehicle_type in vehicle_stats:
                        vehicle_stats[vehicle_type] += count
                    else:
                        vehicle_stats[vehicle_type] = count
                
                log_to_file(f"   {edge_name} ({edge_id}): {edge_metrics.vehicle_count} veh, {edge_metrics.avg_speed_kmh:.1f} km/h, "
                            f"density: {edge_metrics.density:.2f} veh/km, "
                            f"circulation: {edge_metrics.avg_circulation_time_sec:.1f}s")
                
            except Exception as e:
                edge_name = descriptive_names.get_edge_name(edge_id)
                logger.warning(f"Error obteniendo datos de {edge_name} ({edge_id}): {e}")
        
        if total_vehicles == 0:
            log_to_file(f"   Sin vehículos en la intersección")
            return IntersectionData(
                intersection_id=intersection_id,
                edges=edges,
                traffic_light_id=traffic_light_id,
                vehicle_count=0,
                average_speed=0.0,
                density=0.0,
                queue_length=0,
                avg_circulation_time=0.0,
                vehicle_stats={},
                timestamp=current_time
            )
        
        # Calcular métricas agregadas
        average_speed = total_speed / total_vehicles if total_vehicles > 0 else 0.0
        avg_circulation_time = total_circulation_time / total_vehicles if total_vehicles > 0 else 0.0
        
        # Calcular densidad agregada (promedio ponderado de todos los edges)
        if edges and all_visible_vehicles:
            total_density = 0.0
            total_weight = 0.0
            
            for edge_id in edges:
                edge_vehicles = [v for v in all_visible_vehicles if v in self.metrics_calculator.get_visible_vehicles(edge_id)]
                if edge_vehicles:
                    edge_density = self.metrics_calculator.calculate_density(edge_vehicles, edge_id)
                    # Ponderar por número de vehículos en este edge
                    weight = len(edge_vehicles)
                    total_density += edge_density * weight
                    total_weight += weight
            
            density = total_density / total_weight if total_weight > 0 else 0.0
            log_to_file(f"   DENSIDAD CALCULADA: {density:.2f} veh/km (promedio ponderado de {len(edges)} edges, total vehicles: {len(all_visible_vehicles)})")
        else:
            density = 0.0
            log_to_file(f"   DENSIDAD: 0.0 (no edges or vehicles)")
        
        # Calcular cola (solo para detección interna)
        queue_length = self._calculate_queue_length(edges)
        
        # Debug: información final
        log_to_file(f"   Total: {total_vehicles} veh, {average_speed:.1f} km/h")
        log_to_file(f"   Densidad: {density:.1f} veh/km (umbral: {BOTTLENECK_CONFIG['density_threshold']})")
        log_to_file(f"   Cola: {queue_length} veh (umbral: {BOTTLENECK_CONFIG['queue_length_threshold']})")
        log_to_file(f"   Velocidad: {average_speed:.1f} km/h (umbral: {BOTTLENECK_CONFIG['speed_threshold'] * 3.6})")
        
        return IntersectionData(
            intersection_id=intersection_id,
            edges=edges,
            traffic_light_id=traffic_light_id,
            vehicle_count=total_vehicles,
            average_speed=average_speed,
            density=density,
            queue_length=queue_length,
            avg_circulation_time=avg_circulation_time,
            vehicle_stats=vehicle_stats,
            timestamp=current_time
        )



    def _calculate_queue_length(self, edges: List[str]) -> int:
        queue_length = 0
        for edge_id in edges:
            try:
                # Obtener nombre descriptivo al inicio para evitar errores de scope
                edge_name = descriptive_names.get_edge_name(edge_id)
                
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
                logger.warning(f"Error calculando cola en {edge_name} ({edge_id}): {e}")
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
                intersection_name = descriptive_names.get_intersection_name(intersection_id)
                logger.info(f"Cuello de botella detectado en {intersection_name} ({intersection_id}): {bottleneck.severity}")
        return detections

    def _analyze_bottleneck(self, data: IntersectionData) -> Optional[BottleneckDetection]:
        density_threshold = BOTTLENECK_CONFIG["density_threshold"]
        speed_threshold = BOTTLENECK_CONFIG["speed_threshold"]
        queue_threshold = BOTTLENECK_CONFIG["queue_length_threshold"]
        
        # Verificar que realmente hay vehículos antes de analizar
        if data.vehicle_count == 0:
            return None
        
        # CORRECCIÓN: Lógica mejorada de detección
        density_ok = data.density > density_threshold
        speed_ok = data.average_speed < speed_threshold
        queue_ok = data.queue_length > queue_threshold
        
        # CORRECCIÓN: Lógica de detección más sofisticada
        # Detectar si hay congestión REAL con criterios más estrictos
        if data.vehicle_count >= 4:  # Reducido de 6 a 4 vehículos mínimos
            # Criterio 1: Alta densidad Y baja velocidad (congestión severa)
            if density_ok and speed_ok:
                severity = "high"
            # Criterio 2: Alta densidad Y cola significativa
            elif density_ok and queue_ok:
                severity = "medium"
            # Criterio 3: Baja velocidad Y cola significativa
            elif speed_ok and queue_ok:
                severity = "medium"
            # Criterio 4: Solo alta densidad (congestión leve)
            elif density_ok:
                severity = "low"
            # Criterio 5: Solo baja velocidad (posible congestión)
            elif speed_ok and data.average_speed < 5.0:  # Muy baja velocidad
                severity = "low"
            else:
                return None
        else:
            return None
        
        if not self._check_detection_duration(data.intersection_id, severity):
            return None
        
        # Solo log cuando hay detección real
        intersection_name = descriptive_names.get_intersection_name(data.intersection_id)
        logger.info(f"Cuello de botella detectado en {intersection_name} ({data.intersection_id}): {severity}")
        
        metrics = {
            "vehicle_count": data.vehicle_count,
            "average_speed": data.average_speed,
            "density": data.density,
            "queue_length": data.queue_length,
            "avg_circulation_time_sec": data.avg_circulation_time,
            "vehicle_stats": data.vehicle_stats
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