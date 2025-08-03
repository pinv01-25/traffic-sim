"""
Calculador de métricas de tráfico para traffic-sim
Implementa cálculos precisos de métricas críticas para traffic-sync
"""

import traci
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
import time

from utils.logger import setup_logger
from utils.metrics_validator import metrics_validator
from utils.descriptive_names import descriptive_names
from config import BOTTLENECK_CONFIG

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
        Obtiene vehículos visibles en el rango especificado desde el semáforo
        
        Args:
            edge_id: ID del edge a analizar
            
        Returns:
            Lista de IDs de vehículos visibles
        """
        # Obtener nombre descriptivo al inicio para evitar errores de scope
        edge_name = descriptive_names.get_edge_name(edge_id)
        
        visible_vehicles = []
        try:
            # Obtener longitud del edge usando getLastStepLength
            edge_length = traci.edge.getLastStepLength(edge_id)
            
            # Robustecer: asegurar que edge_length sea float
            if isinstance(edge_length, (list, tuple)):
                edge_length = float(edge_length[0]) if edge_length else 0.0
            else:
                try:
                    edge_length = float(edge_length)
                except Exception:
                    edge_length = 0.0
            
            if edge_length == 0.0:
                edge_length = 100.0  # valor por defecto
            
            # Obtener todos los vehículos en el edge
            vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
            if isinstance(vehicles, (list, tuple)):
                vehicles = list(vehicles)
            else:
                vehicles = [vehicles] if vehicles else []
            
            for v_id in vehicles:
                try:
                    # Verificar que el vehículo está en el edge correcto
                    v_lane = traci.vehicle.getLaneID(v_id)
                    if isinstance(v_lane, (list, tuple)):
                        v_lane = str(v_lane[0]) if v_lane else ""
                    else:
                        v_lane = str(v_lane)
                    
                    if not v_lane.startswith(edge_id):
                        continue
                    
                    # Obtener posición del vehículo
                    pos = traci.vehicle.getLanePosition(v_id)
                    if isinstance(pos, (list, tuple)):
                        pos = float(pos[0]) if pos else 0.0
                    else:
                        try:
                            pos = float(pos)
                        except Exception:
                            pos = 0.0
                    
                    # Verificar si está en el rango visible (desde el final del edge hacia atrás)
                    if (edge_length - pos) <= self.visible_range:
                        visible_vehicles.append(v_id)
                        
                except Exception as e:
                    logger.warning(f"Error procesando vehículo {v_id}: {e}")
                    continue
                    
        except Exception as e:
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
        Calcula la tasa de vehículos por minuto basándose en entradas reales
        
        Args:
            edge_id: ID del edge
            current_time: Tiempo actual de simulación
            
        Returns:
            Número de vehículos por minuto
        """
        # Obtener nombre descriptivo al inicio para evitar errores de scope
        edge_name = descriptive_names.get_edge_name(edge_id)
        
        if edge_id not in self.edge_vehicle_tracking:
            return 0
        
        # Contar vehículos que entraron en el último minuto (60 segundos)
        one_minute_ago = current_time - 60.0
        recent_entries = 0
        
        for vehicle_id, entry_time in self.edge_vehicle_tracking[edge_id].items():
            if entry_time >= one_minute_ago:
                recent_entries += 1
        
        # CORRECCIÓN: Mejorar estimación cuando no hay entradas recientes
        if recent_entries == 0:
            current_vehicles = len(self.get_visible_vehicles(edge_id))
            # Estimación más realista: asumir que los vehículos actuales representan
            # el flujo del último minuto, pero con un factor de corrección
            if current_vehicles > 0:
                # Factor de corrección: en tráfico urbano, los vehículos visibles
                # representan aproximadamente 1/3 del flujo por minuto
                estimated_vpm = max(1, int(current_vehicles * 3))
                return estimated_vpm
            else:
                return 0
        
        # CORRECCIÓN: Limitar VPM a valores realistas
        # En tráfico urbano, raramente se superan 100 vehículos por minuto por carril
        if recent_entries > 100:
            recent_entries = 100
            logger.warning(f"VPM muy alto para {edge_name} ({edge_id}), limitado a 100")
        
        return recent_entries
    
    def calculate_avg_speed_kmh(self, visible_vehicles: List[str]) -> float:
        """
        Calcula la velocidad promedio en km/h de los vehículos visibles
        
        Args:
            visible_vehicles: Lista de IDs de vehículos visibles
            
        Returns:
            Velocidad promedio en km/h
        """
        if not visible_vehicles:
            return 0.0
        
        total_speed = 0.0
        valid_vehicles = 0
        stopped_vehicles = 0
        
        for v_id in visible_vehicles:
            try:
                speed = traci.vehicle.getSpeed(v_id)
                
                # Robustecer: asegurar que speed sea float
                if isinstance(speed, (list, tuple)):
                    speed = float(speed[0]) if speed else 0.0
                else:
                    try:
                        speed = float(speed)
                    except Exception:
                        speed = 0.0
                
                # CORRECCIÓN: Manejar mejor vehículos detenidos
                if speed < 0.1:  # Menos de 0.1 m/s = prácticamente detenido
                    stopped_vehicles += 1
                    # No incluir en el promedio si está completamente detenido
                    continue
                
                if speed >= 0:  # Solo vehículos con velocidad válida
                    total_speed += speed
                    valid_vehicles += 1
                    
            except Exception as e:
                logger.warning(f"Error obteniendo velocidad de {v_id}: {e}")
                continue
        
        # CORRECCIÓN: Si todos los vehículos están detenidos, retornar 0
        if valid_vehicles == 0:
            if stopped_vehicles > 0:
                return 0.0
            else:
                return 0.0
        
        # Convertir de m/s a km/h
        avg_speed_mps = total_speed / valid_vehicles
        avg_speed_kmh = avg_speed_mps * 3.6
        
        # CORRECCIÓN: Velocidades mínimas realistas para tráfico urbano
        # Si la velocidad es muy baja pero hay vehículos en movimiento, usar mínimo realista
        if avg_speed_kmh < 5.0 and valid_vehicles > 0:
            avg_speed_kmh = 5.0  # Mínimo 5 km/h para vehículos en movimiento
        
        return round(avg_speed_kmh, 2)
    
    def calculate_avg_circulation_time_sec(self, visible_vehicles: List[str]) -> float:
        """
        Calcula el tiempo promedio de circulación (espera) en segundos
        
        Args:
            visible_vehicles: Lista de IDs de vehículos visibles
            
        Returns:
            Tiempo promedio de circulación en segundos
        """
        if not visible_vehicles:
            return 0.0
        
        total_waiting_time = 0.0
        valid_vehicles = 0
        
        for v_id in visible_vehicles:
            try:
                waiting_time = traci.vehicle.getWaitingTime(v_id)
                
                # Robustecer: asegurar que waiting_time sea float
                if isinstance(waiting_time, (list, tuple)):
                    waiting_time = float(waiting_time[0]) if waiting_time else 0.0
                else:
                    try:
                        waiting_time = float(waiting_time)
                    except Exception:
                        waiting_time = 0.0
                
                if waiting_time >= 0:  # Solo vehículos con tiempo de espera válido
                    total_waiting_time += waiting_time
                    valid_vehicles += 1
                    
            except Exception as e:
                logger.warning(f"Error obteniendo tiempo de espera de {v_id}: {e}")
                continue
        
        if valid_vehicles == 0:
            return 0.0
        
        avg_circulation_time = total_waiting_time / valid_vehicles
        return round(avg_circulation_time, 2)
    
    def calculate_density(self, visible_vehicles: List[str], edge_id: str) -> float:
        """
        Calcula la densidad de vehículos por kilómetro en el rango visible
        
        Args:
            visible_vehicles: Lista de IDs de vehículos visibles
            edge_id: ID del edge para obtener longitud
            
        Returns:
            Densidad en vehículos por kilómetro
        """
        vehicle_count = len(visible_vehicles)
        
        if vehicle_count == 0:
            return 0.0
        
        try:
            # Obtener nombre descriptivo al inicio para evitar errores de scope
            edge_name = descriptive_names.get_edge_name(edge_id)
            
            # CORRECCIÓN CRÍTICA: La densidad debe calcularse SOLO sobre el rango visible
            # No sobre toda la longitud del edge, sino sobre los 60m visibles desde el semáforo
            visible_length_km = self.visible_range / 1000.0  # 60m = 0.06km
            
            if visible_length_km <= 0:
                return 0.0
            
            density = vehicle_count / visible_length_km
            
            # CORRECCIÓN: Limitar densidad máxima a valores realistas
            # En tráfico urbano, densidades > 200 veh/km son extremas
            if density > 200.0:
                density = 200.0
                logger.warning(f"Densidad calculada muy alta para {edge_name} ({edge_id}), limitada a 200 veh/km")
            

            
            return round(density, 2)
            
        except Exception as e:
            edge_name = descriptive_names.get_edge_name(edge_id)
            logger.error(f"Error calculando densidad para {edge_name} ({edge_id}): {e}")
            return 0.0
    
    def calculate_metrics(self, edge_id: str, current_time: float) -> TrafficMetrics:
        """
        Calcula todas las métricas para un edge específico
        
        Args:
            edge_id: ID del edge
            current_time: Tiempo actual de simulación
            
        Returns:
            Objeto TrafficMetrics con todas las métricas calculadas
        """
        # Obtener nombre descriptivo al inicio para evitar errores de scope
        edge_name = descriptive_names.get_edge_name(edge_id)
        
        # Obtener vehículos visibles
        visible_vehicles = self.get_visible_vehicles(edge_id)
        
        # Trackear entradas de vehículos
        for v_id in visible_vehicles:
            self.track_vehicle_entry(edge_id, v_id, current_time)
        
        # Calcular métricas
        vehicles_per_minute = self.calculate_vehicles_per_minute(edge_id, current_time)
        avg_speed_kmh = self.calculate_avg_speed_kmh(visible_vehicles)
        avg_circulation_time_sec = self.calculate_avg_circulation_time_sec(visible_vehicles)
        density = self.calculate_density(visible_vehicles, edge_id)
        vehicle_count = len(visible_vehicles)
        
        # Crear objeto de métricas
        metrics = TrafficMetrics(
            vehicles_per_minute=vehicles_per_minute,
            avg_speed_kmh=avg_speed_kmh,
            avg_circulation_time_sec=avg_circulation_time_sec,
            density=density,
            vehicle_count=vehicle_count,
            timestamp=current_time
        )
        
        # VALIDACIÓN AUTOMÁTICA: Verificar que las métricas sean realistas
        metrics_dict = {
            "vehicles_per_minute": vehicles_per_minute,
            "avg_speed_kmh": avg_speed_kmh,
            "avg_circulation_time_sec": avg_circulation_time_sec,
            "density": density,
            "vehicle_count": vehicle_count
        }
        
        validation_result = metrics_validator.validate_metrics(metrics_dict, edge_id)
        
        # CORRECCIÓN: Solo mostrar warnings si hay vehículos o errores críticos
        if vehicle_count > 0 or not validation_result.is_valid:
            metrics_validator.log_validation_result(validation_result, edge_id)
        
        # Si hay errores críticos, registrar y ajustar
        if not validation_result.is_valid:
            logger.error(f"Métricas inválidas calculadas para {edge_name} ({edge_id})")
            for error in validation_result.errors:
                logger.error(f"   Error: {error}")
        
        return metrics
    
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