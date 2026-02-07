"""
Utilidades para manejo seguro de valores TRACI.

Proporciona funciones helper para conversión de tipos y obtención de
datos de la red de forma robusta, centralizando el manejo de errores.
"""

from typing import Any, Optional

import traci
from utils.logger import setup_logger

logger = setup_logger(__name__)


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convierte un valor TRACI a float de forma segura.

    Args:
        value: Valor a convertir (puede ser float, int, str, list, tuple)
        default: Valor por defecto si la conversión falla

    Returns:
        Valor convertido a float o default si falla
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Convierte un valor TRACI a int de forma segura.

    Args:
        value: Valor a convertir
        default: Valor por defecto si la conversión falla

    Returns:
        Valor convertido a int o default si falla
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_str(value: Any, default: str = "") -> str:
    """
    Convierte un valor TRACI a string de forma segura.

    Args:
        value: Valor a convertir
        default: Valor por defecto si la conversión falla

    Returns:
        Valor convertido a string o default si falla
    """
    if value is None:
        return default
    try:
        return str(value)
    except (ValueError, TypeError):
        return default


def get_lane_length(lane_id: str) -> float:
    """
    Obtiene la longitud de un carril usando la API TRACI correcta.

    Args:
        lane_id: ID del carril (ej: "edge_0_0")

    Returns:
        Longitud del carril en metros, o 100.0 como fallback
    """
    try:
        length = traci.lane.getLength(lane_id)
        return safe_float(length, 100.0)
    except traci.exceptions.TraCIException as e:
        logger.warning(f"Error obteniendo longitud del carril {lane_id}: {e}")
        return 100.0


def get_edge_length_from_lanes(edge_id: str) -> float:
    """
    Obtiene la longitud de un edge desde sus carriles.

    Usa traci.lane.getLength() que es el método correcto,
    NO traci.edge.getLastStepLength() que devuelve suma de vehículos.

    Args:
        edge_id: ID del edge

    Returns:
        Longitud del edge en metros, o 100.0 como fallback
    """
    try:
        # Obtener número de carriles del edge
        lane_count = traci.edge.getLaneNumber(edge_id)
        if lane_count > 0:
            # Usar el primer carril para obtener longitud
            lane_id = f"{edge_id}_0"
            return get_lane_length(lane_id)
        return 100.0
    except traci.exceptions.TraCIException as e:
        logger.warning(f"Error obteniendo longitud del edge {edge_id}: {e}")
        return 100.0


def get_vehicle_speed(vehicle_id: str) -> float:
    """
    Obtiene la velocidad de un vehículo de forma segura.

    Args:
        vehicle_id: ID del vehículo

    Returns:
        Velocidad en m/s, o 0.0 si hay error
    """
    try:
        speed = traci.vehicle.getSpeed(vehicle_id)
        return safe_float(speed, 0.0)
    except traci.exceptions.TraCIException:
        return 0.0


def get_vehicle_waiting_time(vehicle_id: str) -> float:
    """
    Obtiene el tiempo de espera de un vehículo de forma segura.

    Args:
        vehicle_id: ID del vehículo

    Returns:
        Tiempo de espera en segundos, o 0.0 si hay error
    """
    try:
        waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
        return safe_float(waiting_time, 0.0)
    except traci.exceptions.TraCIException:
        return 0.0


def get_vehicle_lane_position(vehicle_id: str) -> float:
    """
    Obtiene la posición de un vehículo en su carril de forma segura.

    Args:
        vehicle_id: ID del vehículo

    Returns:
        Posición en metros desde el inicio del carril, o 0.0 si hay error
    """
    try:
        pos = traci.vehicle.getLanePosition(vehicle_id)
        return safe_float(pos, 0.0)
    except traci.exceptions.TraCIException:
        return 0.0


def get_vehicle_lane_id(vehicle_id: str) -> str:
    """
    Obtiene el ID del carril donde está un vehículo de forma segura.

    Args:
        vehicle_id: ID del vehículo

    Returns:
        ID del carril, o string vacío si hay error
    """
    try:
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        return safe_str(lane_id, "")
    except traci.exceptions.TraCIException:
        return ""


def get_vehicle_type(vehicle_id: str) -> str:
    """
    Obtiene el tipo de un vehículo de forma segura.

    Args:
        vehicle_id: ID del vehículo

    Returns:
        Tipo del vehículo, o string vacío si hay error
    """
    try:
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        return safe_str(vehicle_type, "")
    except traci.exceptions.TraCIException:
        return ""


def get_edge_vehicles(edge_id: str) -> list:
    """
    Obtiene la lista de vehículos en un edge de forma segura.

    Args:
        edge_id: ID del edge

    Returns:
        Lista de IDs de vehículos, o lista vacía si hay error
    """
    try:
        vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
        return list(vehicles) if vehicles else []
    except traci.exceptions.TraCIException:
        return []
