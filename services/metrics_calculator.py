"""
Calculador de métricas de tráfico para traffic-sim
Implementa cálculos precisos de métricas críticas para traffic-sync
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import traci
from config import BOTTLENECK_CONFIG
from utils.descriptive_names import descriptive_names
from utils.logger import setup_logger
from utils.metrics_validator import metrics_validator
from utils.traci_helpers import (
    get_edge_length_from_lanes,
    get_edge_vehicles,
    get_vehicle_lane_id,
    get_vehicle_lane_position,
    get_vehicle_speed,
    get_vehicle_type,
    get_vehicle_waiting_time,
    safe_float,
)

logger = setup_logger(__name__)

@dataclass
class VehicleMetrics:
    """Métricas de un vehículo individual"""
    vehicle_id: str
    speed: float  # m/s
    waiting_time: float  # segundos
    position: float  # metros desde inicio del edge
    timestamp: float  # tiempo de simulación

@dataclass
class TrafficMetrics:
    """Métricas de tráfico calculadas"""
    vehicles_per_minute: int
    avg_speed_kmh: float
    avg_circulation_time_sec: float
    density: float  # vehículos por kilómetro
    vehicle_count: int
    vehicle_stats: Dict[str, int]  # Clasificación por tipo de vehículo
    timestamp: float

class MetricsCalculator:
    """
    Calculador de métricas de tráfico con tracking temporal
    """
    
    def __init__(self):
        self.vehicle_history: Dict[str, deque] = {}  # vehicle_id -> deque de timestamps
        self.edge_vehicle_tracking: Dict[str, Dict[str, float]] = {}  # edge_id -> {vehicle_id -> entry_time}
        self.last_calculation_time = 0.0
        self.visible_range = BOTTLENECK_CONFIG["visible_range"]
        
    def get_visible_vehicles(self, edge_id: str) -> List[str]:
        """
        Obtiene vehículos visibles en el rango especificado desde el semáforo.

        Args:
            edge_id: ID del edge a analizar

        Returns:
            Lista de IDs de vehículos visibles
        """
        visible_vehicles = []
        try:
            # Usar get_edge_length_from_lanes que usa traci.lane.getLength() correcto
            edge_length = get_edge_length_from_lanes(edge_id)

            # Obtener todos los vehículos en el edge
            vehicles = get_edge_vehicles(edge_id)

            for v_id in vehicles:
                # Verificar que el vehículo está en el edge correcto
                v_lane = get_vehicle_lane_id(v_id)
                if not v_lane.startswith(edge_id):
                    continue

                # Obtener posición del vehículo
                pos = get_vehicle_lane_position(v_id)

                # Verificar si está en el rango visible (desde el final del edge hacia atrás)
                if (edge_length - pos) <= self.visible_range:
                    visible_vehicles.append(v_id)

        except Exception as e:
            edge_name = descriptive_names.get_edge_name(edge_id)
            logger.error(f"Error obteniendo vehículos visibles en {edge_name} ({edge_id}): {e}")

        return visible_vehicles
    
    def track_vehicle_entry(self, edge_id: str, vehicle_id: str, current_time: float):
        """
        Registra cuando un vehículo entra al rango visible
        
        Args:
            edge_id: ID del edge
            vehicle_id: ID del vehículo
            current_time: Tiempo actual de simulación
        """
        if edge_id not in self.edge_vehicle_tracking:
            self.edge_vehicle_tracking[edge_id] = {}
        
        # Solo registrar si no está ya siendo trackeado
        if vehicle_id not in self.edge_vehicle_tracking[edge_id]:
            self.edge_vehicle_tracking[edge_id][vehicle_id] = current_time
            
            # Agregar al historial global
            if vehicle_id not in self.vehicle_history:
                self.vehicle_history[vehicle_id] = deque(maxlen=1000)
            self.vehicle_history[vehicle_id].append(current_time)
    
    def calculate_vehicles_per_minute(self, edge_id: str, current_time: float) -> int:
        """
        Calcula la tasa de vehículos por minuto basándose en entradas reales.

        Args:
            edge_id: ID del edge
            current_time: Tiempo actual de simulación

        Returns:
            Número de vehículos por minuto
        """
        if edge_id not in self.edge_vehicle_tracking:
            return 0

        # Contar vehículos que entraron en el último minuto (60 segundos)
        one_minute_ago = current_time - 60.0
        recent_entries = sum(
            1 for entry_time in self.edge_vehicle_tracking[edge_id].values()
            if entry_time >= one_minute_ago
        )

        # Si no hay entradas recientes, estimar basándose en vehículos visibles
        if recent_entries == 0:
            current_vehicles = len(self.get_visible_vehicles(edge_id))
            if current_vehicles > 0:
                # Factor de corrección: vehículos visibles representan ~1/3 del flujo
                return max(1, current_vehicles * 3)
            return 0

        # Limitar VPM a valores realistas (máximo 100 por carril)
        return min(recent_entries, 100)
    
    def calculate_avg_speed_kmh(self, visible_vehicles: List[str]) -> float:
        """
        Calcula la velocidad promedio en km/h de los vehículos visibles.

        Args:
            visible_vehicles: Lista de IDs de vehículos visibles

        Returns:
            Velocidad promedio en km/h
        """
        if not visible_vehicles:
            return 0.0

        total_speed = 0.0
        valid_vehicles = 0

        for v_id in visible_vehicles:
            speed = get_vehicle_speed(v_id)
            # Ignorar vehículos prácticamente detenidos (< 0.1 m/s)
            if speed >= 0.1:
                total_speed += speed
                valid_vehicles += 1

        if valid_vehicles == 0:
            return 0.0

        # Convertir de m/s a km/h
        avg_speed_kmh = (total_speed / valid_vehicles) * 3.6

        # Mínimo 5 km/h para vehículos en movimiento
        return round(max(avg_speed_kmh, 5.0), 2)
    
    def calculate_avg_circulation_time_sec(self, visible_vehicles: List[str]) -> float:
        """
        Calcula el tiempo promedio de circulación (espera) en segundos.

        Args:
            visible_vehicles: Lista de IDs de vehículos visibles

        Returns:
            Tiempo promedio de circulación en segundos
        """
        if not visible_vehicles:
            return 0.0

        total_waiting_time = sum(
            get_vehicle_waiting_time(v_id) for v_id in visible_vehicles
        )

        return round(total_waiting_time / len(visible_vehicles), 2)
    
    def calculate_density(self, visible_vehicles: List[str], edge_id: str) -> float:
        """
        Calcula la densidad de vehículos por kilómetro en el rango visible.

        Args:
            visible_vehicles: Lista de IDs de vehículos visibles
            edge_id: ID del edge (usado solo para logging)

        Returns:
            Densidad en vehículos por kilómetro
        """
        vehicle_count = len(visible_vehicles)
        if vehicle_count == 0:
            return 0.0

        # La densidad se calcula sobre el rango visible (60m = 0.06km)
        visible_length_km = self.visible_range / 1000.0
        if visible_length_km <= 0:
            return 0.0

        density = vehicle_count / visible_length_km

        # Limitar a máximo 200 veh/km (valor extremo para tráfico urbano)
        return round(min(density, 200.0), 2)
    
    def calculate_metrics(self, edge_id: str, current_time: float) -> TrafficMetrics:
        """
        Calcula todas las métricas para un edge específico.

        Args:
            edge_id: ID del edge
            current_time: Tiempo actual de simulación

        Returns:
            Objeto TrafficMetrics con todas las métricas calculadas
        """
        # Obtener vehículos visibles
        visible_vehicles = self.get_visible_vehicles(edge_id)

        # Trackear entradas de vehículos
        for v_id in visible_vehicles:
            self.track_vehicle_entry(edge_id, v_id, current_time)

        # Calcular métricas
        vehicle_count = len(visible_vehicles)
        metrics = TrafficMetrics(
            vehicles_per_minute=self.calculate_vehicles_per_minute(edge_id, current_time),
            avg_speed_kmh=self.calculate_avg_speed_kmh(visible_vehicles),
            avg_circulation_time_sec=self.calculate_avg_circulation_time_sec(visible_vehicles),
            density=self.calculate_density(visible_vehicles, edge_id),
            vehicle_count=vehicle_count,
            vehicle_stats=self.classify_vehicles_by_type(visible_vehicles),
            timestamp=current_time
        )

        # Validar métricas solo si hay vehículos o errores
        metrics_dict = {
            "vehicles_per_minute": metrics.vehicles_per_minute,
            "avg_speed_kmh": metrics.avg_speed_kmh,
            "avg_circulation_time_sec": metrics.avg_circulation_time_sec,
            "density": metrics.density,
            "vehicle_count": vehicle_count
        }
        validation_result = metrics_validator.validate_metrics(metrics_dict, edge_id)

        if vehicle_count > 0 or not validation_result.is_valid:
            metrics_validator.log_validation_result(validation_result, edge_id)

        return metrics
    
    def classify_vehicles_by_type(self, visible_vehicles: List[str]) -> Dict[str, int]:
        """
        Clasifica los vehículos visibles por tipo de vehículo.

        Args:
            visible_vehicles: Lista de IDs de vehículos visibles.

        Returns:
            Diccionario con tipo de vehículo como clave y cantidad como valor.
        """
        vehicle_stats: Dict[str, int] = {}

        for v_id in visible_vehicles:
            vehicle_type = get_vehicle_type(v_id)
            if vehicle_type:
                vehicle_stats[vehicle_type] = vehicle_stats.get(vehicle_type, 0) + 1

        return vehicle_stats
    
    def cleanup_old_tracking(self, current_time: float):
        """
        Limpia tracking de vehículos antiguos para evitar acumulación de memoria
        
        Args:
            current_time: Tiempo actual de simulación
        """
        # Limpiar tracking de edges (vehículos que salieron hace más de 5 minutos)
        five_minutes_ago = current_time - 300.0
        
        for edge_id in list(self.edge_vehicle_tracking.keys()):
            vehicles_to_remove = []
            for vehicle_id, entry_time in self.edge_vehicle_tracking[edge_id].items():
                if entry_time < five_minutes_ago:
                    vehicles_to_remove.append(vehicle_id)
            
            for vehicle_id in vehicles_to_remove:
                del self.edge_vehicle_tracking[edge_id][vehicle_id]
            
            # Eliminar edge si no tiene vehículos
            if not self.edge_vehicle_tracking[edge_id]:
                del self.edge_vehicle_tracking[edge_id] 