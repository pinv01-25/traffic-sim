"""
Cliente HTTP para comunicación con traffic-control
"""

import requests
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from utils.logger import setup_logger
from config import TRAFFIC_CONTROL_CONFIG

logger = setup_logger(__name__)

@dataclass
class TrafficMetrics:
    """Métricas de tráfico para enviar a traffic-control"""
    vehicles_per_minute: int
    avg_speed_kmh: float
    avg_circulation_time_sec: float
    density: float

@dataclass
class VehicleStats:
    """Estadísticas de vehículos por tipo"""
    motorcycle: int
    car: int
    bus: int
    truck: int

@dataclass
class TrafficDataPayload:
    """Payload completo para enviar a traffic-control"""
    version: str = "1.0"
    type: str = "data"
    timestamp: str = ""
    traffic_light_id: str = ""
    controlled_edges: list = None
    metrics: TrafficMetrics = None
    vehicle_stats: VehicleStats = None

    def __post_init__(self):
        if self.controlled_edges is None:
            self.controlled_edges = []
        if self.metrics is None:
            self.metrics = TrafficMetrics(0, 0.0, 0.0, 0.0)
        if self.vehicle_stats is None:
            self.vehicle_stats = VehicleStats(0, 0, 0, 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el payload a diccionario para envío HTTP"""
        return {
            "version": self.version,
            "type": self.type,
            "timestamp": self.timestamp,
            "traffic_light_id": self.traffic_light_id,
            "controlled_edges": self.controlled_edges,
            "metrics": {
                "vehicles_per_minute": self.metrics.vehicles_per_minute,
                "avg_speed_kmh": self.metrics.avg_speed_kmh,
                "avg_circulation_time_sec": self.metrics.avg_circulation_time_sec,
                "density": self.metrics.density
            },
            "vehicle_stats": {
                "motorcycle": self.vehicle_stats.motorcycle,
                "car": self.vehicle_stats.car,
                "bus": self.vehicle_stats.bus,
                "truck": self.vehicle_stats.truck
            }
        }

class TrafficControlClient:
    """
    Cliente para comunicación síncrona con traffic-control
    """
    
    def __init__(self):
        self.base_url = TRAFFIC_CONTROL_CONFIG["base_url"]
        self.timeout = TRAFFIC_CONTROL_CONFIG["timeout"]
        self.retry_attempts = TRAFFIC_CONTROL_CONFIG["retry_attempts"]
        self.retry_delay = TRAFFIC_CONTROL_CONFIG["retry_delay"]
        self.session = requests.Session()
        
        # Configurar headers por defecto
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _make_request(self, method: str, endpoint: str, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Realiza una petición HTTP con reintentos automáticos
        
        Args:
            method: Método HTTP (GET, POST, etc.)
            endpoint: Endpoint específico (sin base_url)
            data: Datos a enviar (opcional)
            
        Returns:
            Respuesta JSON del servidor
            
        Raises:
            requests.RequestException: Si fallan todos los reintentos
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"Intento {attempt + 1}/{self.retry_attempts} - {method} {url}")
                
                if method.upper() == "GET":
                    response = self.session.get(url, timeout=self.timeout)
                elif method.upper() == "POST":
                    response = self.session.post(url, json=data, timeout=self.timeout)
                else:
                    raise ValueError(f"Método HTTP no soportado: {method}")
                
                response.raise_for_status()
                
                logger.info(f"Petición exitosa - Status: {response.status_code}")
                return response.json()
                
                response.raise_for_status()
                
                logger.info(f"Petición exitosa - Status: {response.status_code}")
                return response.json()
                
            except requests.RequestException as e:
                logger.warning(f"Intento {attempt + 1} falló: {e}")
                
                if attempt < self.retry_attempts - 1:
                    logger.info(f"Esperando {self.retry_delay} segundos antes del siguiente intento...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Todos los intentos fallaron para {method} {url}")
                    raise
    
    def health_check(self) -> bool:
        """
        Verifica que traffic-control esté disponible
        
        Returns:
            True si el servicio está saludable
        """
        try:
            response = self._make_request("GET", "/healthcheck")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Health check falló: {e}")
            return False
    
    def send_traffic_data(self, payload: TrafficDataPayload) -> Dict[str, Any]:
        """
        Envía datos de tráfico a traffic-control para procesamiento
        
        Args:
            payload: Datos de tráfico formateados
            
        Returns:
            Respuesta de traffic-control con datos optimizados
        """
        try:
            logger.info(f"Enviando datos de tráfico para semáforo {payload.traffic_light_id}")
            
            # Validar payload antes de enviar
            self._validate_payload(payload)
            
            # Convertir a diccionario
            data = payload.to_dict()
            
            # Enviar a traffic-control
            response = self._make_request("POST", "/process", data)
            
            logger.info(f"Datos procesados exitosamente para {payload.traffic_light_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error enviando datos de tráfico: {e}")
            raise
    
    def _validate_payload(self, payload: TrafficDataPayload):
        """
        Valida que el payload tenga todos los campos requeridos
        
        Args:
            payload: Payload a validar
            
        Raises:
            ValueError: Si el payload no es válido
        """
        if not payload.traffic_light_id:
            raise ValueError("traffic_light_id es requerido")
        
        if not payload.timestamp:
            raise ValueError("timestamp es requerido")
        
        if not payload.controlled_edges:
            raise ValueError("controlled_edges no puede estar vacío")
        
        if payload.metrics.vehicles_per_minute < 0:
            raise ValueError("vehicles_per_minute no puede ser negativo")
        
        if payload.metrics.avg_speed_kmh < 0:
            raise ValueError("avg_speed_kmh no puede ser negativo")
        
        if payload.metrics.density < 0:
            raise ValueError("density no puede ser negativa")
    
    def create_traffic_payload(
        self,
        traffic_light_id: str,
        controlled_edges: list,
        vehicle_count: int,
        average_speed: float,
        density: float,
        queue_length: int,
        timestamp: str | None = None
    ) -> TrafficDataPayload:
        """
        Crea un payload de datos de tráfico a partir de métricas de simulación
        
        Args:
            traffic_light_id: ID del semáforo
            controlled_edges: Lista de calles controladas
            vehicle_count: Número total de vehículos
            average_speed: Velocidad promedio en km/h
            density: Densidad de vehículos por km
            queue_length: Longitud de la cola
            timestamp: Timestamp ISO (si no se proporciona, se genera automáticamente)
            
        Returns:
            Payload formateado para traffic-control
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Calcular métricas adicionales
        vehicles_per_minute = int(vehicle_count * 60 / 3600)  # Estimación por minuto
        avg_circulation_time = 30.0  # Valor por defecto, se puede calcular más precisamente
        
        # Crear payload
        payload = TrafficDataPayload(
            traffic_light_id=traffic_light_id,
            controlled_edges=controlled_edges,
            timestamp=timestamp,
            metrics=TrafficMetrics(
                vehicles_per_minute=vehicles_per_minute,
                avg_speed_kmh=average_speed * 3.6,  # Convertir m/s a km/h
                avg_circulation_time_sec=avg_circulation_time,
                density=density
            ),
            vehicle_stats=VehicleStats(
                motorcycle=0,  # Por defecto, se puede calcular más precisamente
                car=vehicle_count,
                bus=0,
                truck=0
            )
        )
        
        return payload 