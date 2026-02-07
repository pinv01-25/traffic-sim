"""
Cliente HTTP para comunicación con traffic-control
"""

import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from config import TRAFFIC_CONTROL_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Estructuras de datos para respuesta de optimización de clusters
# ============================================================================

@dataclass
class TrafficLightOptimization:
    """Optimización para un semáforo individual"""
    traffic_light_id: str
    green_time_sec: float
    red_time_sec: float
    apply_immediately: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrafficLightOptimization":
        return cls(
            traffic_light_id=str(data.get("traffic_light_id", "")),
            green_time_sec=float(data.get("green_time_sec", 30)),
            red_time_sec=float(data.get("red_time_sec", 30)),
            apply_immediately=bool(data.get("apply_immediately", True))
        )


@dataclass
class ClusterOptimizationResponse:
    """Respuesta de optimización para un cluster de semáforos"""
    status: str
    optimizations: List[TrafficLightOptimization] = field(default_factory=list)
    cluster_id: Optional[str] = None
    coordinated: bool = False
    message: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterOptimizationResponse":
        optimizations = [
            TrafficLightOptimization.from_dict(opt)
            for opt in data.get("optimizations", [])
        ]
        cluster_info = data.get("cluster_info", {})
        return cls(
            status=data.get("status", "error"),
            optimizations=optimizations,
            cluster_id=cluster_info.get("cluster_id"),
            coordinated=cluster_info.get("coordinated", False),
            message=data.get("message")
        )


@dataclass
class ClusterDataPayload:
    """Payload con datos de múltiples semáforos para optimización de cluster"""
    version: str = "2.0"
    type: str = "cluster_data"
    timestamp: str = ""
    primary_traffic_light_id: str = ""  # El que detectó el cuello de botella
    sensors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "type": self.type,
            "timestamp": self.timestamp,
            "primary_traffic_light_id": self.primary_traffic_light_id,
            "sensors": self.sensors
        }

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

    def normalize(self):
        """Normaliza los campos para cumplir con el formato requerido por traffic-control (Pydantic)."""
        # traffic_light_id: solo números
        if self.traffic_light_id:
            match = re.search(r"(\d+)", str(self.traffic_light_id))
            if match:
                self.traffic_light_id = match.group(1)
        # density: normalizar de veh/km a rango 0-1 (traffic-control espera valores entre 0 y 1)
        # Dividir por 100 para convertir veh/km a escala 0-1 (100 veh/km = 1.0)
        if self.metrics and hasattr(self.metrics, 'density'):
            try:
                original_density = self.metrics.density
                # Normalizar: dividir por 100 y limitar a máximo 1.0
                normalized_density = min(self.metrics.density / 100.0, 1.0)
                self.metrics.density = round(normalized_density, 3)
                logger.debug(f"Densidad normalizada: {original_density} veh/km -> {self.metrics.density}")
            except Exception as e:
                logger.warning(f"Error normalizando densidad: {e}")
                pass
        # versión: forzar a 2.0
        self.version = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el payload a diccionario para envío HTTP"""
        self.normalize()
        return {
            "version": self.version,
            "type": self.type,
            "timestamp": self.timestamp,
            "traffic_light_id": self.traffic_light_id,
            "sensors": [
                {
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
            ]
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
        
        # Log la URL que se está usando para debug
        logger.info(f"TrafficControlClient inicializado con base_url: {self.base_url}")
        
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
                
            except requests.RequestException as e:
                # Intentar obtener detalles del error si es una respuesta HTTP
                error_detail = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_body = e.response.json()
                        error_detail = f"{e} - {error_body}"
                        logger.warning(f"Intento {attempt + 1} falló: {error_detail}")
                    except (ValueError, KeyError):
                        error_detail = f"{e} - Status: {e.response.status_code} - Body: {e.response.text[:200]}"
                        logger.warning(f"Intento {attempt + 1} falló: {error_detail}")
                else:
                    logger.warning(f"Intento {attempt + 1} falló: {error_detail}")
                
                if attempt < self.retry_attempts - 1:
                    logger.info(f"Esperando {self.retry_delay} segundos antes del siguiente intento...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Todos los intentos fallaron para {method} {url}: {error_detail}")
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
            
            # Normalizar antes de validar y enviar
            payload.normalize()
            
            # Validar payload antes de enviar
            self._validate_payload(payload)
            
            # Convertir a diccionario
            data = payload.to_dict()
            
            # Log del payload para debug (solo una vez, no en cada reintento)
            logger.debug(f"Payload a enviar: {data}")
            
            # Enviar a traffic-control
            response = self._make_request("POST", "/process", data)
            
            logger.info(f"Datos procesados exitosamente para {payload.traffic_light_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error enviando datos de tráfico: {e}")
            raise
    
    def send_traffic_data_batch(self, batch_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Envía datos de tráfico en formato batch a traffic-control para procesamiento
        
        Args:
            batch_payload: Datos de tráfico en formato batch
            
        Returns:
            Respuesta de traffic-control con datos optimizados
        """
        try:
            traffic_light_id = batch_payload.get("traffic_light_id", "unknown")
            logger.info(f"Enviando datos de tráfico batch para semáforo {traffic_light_id}")
            
            # Enviar a traffic-control
            response = self._make_request("POST", "/process", batch_payload)
            
            logger.info(f"Datos batch procesados exitosamente para {traffic_light_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error enviando datos de tráfico batch: {e}")
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
        metrics: dict,
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
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Crear payload usando las métricas calculadas
        payload = TrafficDataPayload(
            traffic_light_id=traffic_light_id,
            controlled_edges=controlled_edges,
            timestamp=timestamp,
            metrics=TrafficMetrics(
                vehicles_per_minute=metrics.get('vehicles_per_minute', 0),
                avg_speed_kmh=metrics.get('avg_speed_kmh', 0.0),
                avg_circulation_time_sec=metrics.get('avg_circulation_time_sec', 0.0),
                density=metrics.get('density', 0.0)
            ),
            vehicle_stats=VehicleStats(
                motorcycle=metrics.get('vehicle_stats', {}).get('motorcycle', 0),
                car=metrics.get('vehicle_stats', {}).get('car', 0),
                bus=metrics.get('vehicle_stats', {}).get('bus', 0),
                truck=metrics.get('vehicle_stats', {}).get('truck', 0)
            )
        )
        
        return payload

    def send_cluster_data(
        self,
        payload: ClusterDataPayload,
        use_mock: bool = False
    ) -> ClusterOptimizationResponse:
        """
        Envía datos de un cluster de semáforos para optimización coordinada.

        Args:
            payload: Datos del cluster con métricas de todos los semáforos
            use_mock: Si True, usa el mock en lugar de la API real

        Returns:
            Respuesta con optimizaciones para cada semáforo del cluster
        """
        if use_mock:
            return self._mock_cluster_optimization(payload)

        try:
            logger.info(f"Enviando datos de cluster (primario: {payload.primary_traffic_light_id})")
            data = payload.to_dict()

            # Enviar a traffic-control
            response = self._make_request("POST", "/process-cluster", data)

            return ClusterOptimizationResponse.from_dict(response)

        except Exception as e:
            logger.error(f"Error enviando datos de cluster: {e}")
            # Fallback al mock si la API falla
            logger.info("Usando mock como fallback")
            return self._mock_cluster_optimization(payload)

    def _mock_cluster_optimization(
        self,
        payload: ClusterDataPayload
    ) -> ClusterOptimizationResponse:
        """
        Mock que simula la respuesta de la API de optimización.

        Genera tiempos de semáforo basados en la densidad y velocidad
        de cada sensor, simulando una optimización inteligente.
        """
        optimizations = []

        for sensor in payload.sensors:
            tl_id = sensor.get("traffic_light_id", "")
            metrics = sensor.get("metrics", {})

            # Extraer métricas
            density = metrics.get("density", 0.5)  # Ya normalizada 0-1
            avg_speed = metrics.get("avg_speed_kmh", 30.0)
            vpm = metrics.get("vehicles_per_minute", 10)

            # Algoritmo de optimización simple:
            # - Alta densidad + baja velocidad = más tiempo de verde
            # - Baja densidad + alta velocidad = menos tiempo de verde
            congestion_score = density * 0.6 + (1 - min(avg_speed / 50.0, 1.0)) * 0.4

            # Calcular tiempos (ciclo base de 60s)
            base_cycle = 60.0
            if congestion_score > 0.7:
                # Alta congestión: más verde
                green_time = base_cycle * 0.6 + random.uniform(-2, 2)
            elif congestion_score > 0.4:
                # Congestión media: balance
                green_time = base_cycle * 0.5 + random.uniform(-2, 2)
            else:
                # Baja congestión: menos verde
                green_time = base_cycle * 0.4 + random.uniform(-2, 2)

            green_time = max(15, min(green_time, 45))  # Limitar entre 15-45s
            red_time = base_cycle - green_time

            optimizations.append(TrafficLightOptimization(
                traffic_light_id=tl_id,
                green_time_sec=round(green_time, 1),
                red_time_sec=round(red_time, 1),
                apply_immediately=True
            ))

            logger.debug(
                f"Mock optimization for {tl_id}: "
                f"congestion={congestion_score:.2f}, green={green_time:.1f}s"
            )

        return ClusterOptimizationResponse(
            status="success",
            optimizations=optimizations,
            cluster_id=f"cluster_{payload.primary_traffic_light_id}",
            coordinated=True,
            message="Mock optimization applied"
        ) 