"""
Cliente HTTP para comunicación con traffic-control
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from config import TRAFFIC_CONTROL_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TrafficLightOptimization:
    """Optimización para un semáforo individual dentro de un cluster"""
    traffic_light_id: str
    green_time_sec: float
    red_time_sec: float
    cluster_sensors: List[str] = field(default_factory=list)
    original_congestion: int = 0
    optimized_congestion: int = 0
    original_category: str = ""
    optimized_category: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrafficLightOptimization":
        optimization = data.get("optimization", {})
        impact = data.get("impact", {})
        return cls(
            traffic_light_id=str(data.get("traffic_light_id", "")),
            green_time_sec=float(optimization.get("green_time_sec", 30)),
            red_time_sec=float(optimization.get("red_time_sec", 30)),
            cluster_sensors=data.get("cluster_sensors", []),
            original_congestion=int(impact.get("original_congestion", 0)),
            optimized_congestion=int(impact.get("optimized_congestion", 0)),
            original_category=str(impact.get("original_category", "")),
            optimized_category=str(impact.get("optimized_category", "")),
        )


@dataclass
class ClusterOptimizationResponse:
    """Respuesta de optimización para un cluster de semáforos.

    Parsea la respuesta de traffic-control que incluye el OptimizationBatch
    generado por traffic-sync (fuzzy + clustering + PSO).
    """
    status: str
    optimizations: List[TrafficLightOptimization] = field(default_factory=list)
    cluster_id: Optional[str] = None
    message: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "ClusterOptimizationResponse":
        """Parsea la respuesta del endpoint de traffic-control.

        La respuesta incluye un campo 'optimization' con el OptimizationBatch
        que traffic-sync generó (contiene optimizations con cluster_sensors).
        """
        status = data.get("status", "error")
        optimization_batch = data.get("optimization", {})

        optimizations = [
            TrafficLightOptimization.from_dict(opt)
            for opt in optimization_batch.get("optimizations", [])
        ]

        return cls(
            status=status,
            optimizations=optimizations,
            cluster_id=optimization_batch.get("traffic_light_id"),
            message=data.get("message", ""),
        )


@dataclass
class RawSimulationPayload:
    """Raw payload for the /ingest endpoint. No normalization at all.

    traffic-control handles all formatting (ID normalization, density
    conversion, vehicle_stats padding, version assignment).
    """
    timestamp: str = ""
    source_id: str = ""  # Primary traffic light raw SUMO ID
    sensors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for HTTP transmission. No normalization."""
        return {
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "sensors": self.sensors,
        }


class TrafficControlClient:
    """Cliente para comunicación síncrona con traffic-control"""

    def __init__(self):
        self.base_url = TRAFFIC_CONTROL_CONFIG["base_url"]
        self.timeout = TRAFFIC_CONTROL_CONFIG["timeout"]
        self.retry_attempts = TRAFFIC_CONTROL_CONFIG["retry_attempts"]
        self.retry_delay = TRAFFIC_CONTROL_CONFIG["retry_delay"]
        self.session = requests.Session()

        logger.info(f"TrafficControlClient inicializado con base_url: {self.base_url}")

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
                error_detail = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_body = e.response.json()
                        error_detail = f"{e} - {error_body}"
                    except (ValueError, KeyError):
                        error_detail = f"{e} - Status: {e.response.status_code} - Body: {e.response.text[:200]}"
                logger.warning(f"Intento {attempt + 1} falló: {error_detail}")

                if attempt < self.retry_attempts - 1:
                    logger.info(f"Esperando {self.retry_delay} segundos antes del siguiente intento...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Todos los intentos fallaron para {method} {url}: {error_detail}")
                    raise

    def send_raw_data(self, payload: RawSimulationPayload) -> ClusterOptimizationResponse:
        """
        Envía datos crudos de simulación a traffic-control /ingest.

        No se realiza ninguna normalización — traffic-control se encarga de:
        - Normalización de IDs (extraer dígitos)
        - Normalización de densidad (veh/km → 0-1)
        - Relleno de vehicle_stats (asegurar 4 claves)
        - Asignación de versión

        Args:
            payload: Datos crudos de simulación SUMO

        Returns:
            Respuesta con optimizaciones para cada semáforo del cluster
        """
        try:
            logger.info(f"Enviando datos crudos a /ingest (fuente: {payload.source_id})")
            data = payload.to_dict()
            response = self._make_request("POST", "/ingest", data)
            return ClusterOptimizationResponse.from_api_response(response)
        except Exception as e:
            logger.error(f"Error enviando datos crudos: {e}")
            raise
