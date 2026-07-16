"""
Controlador de semáforos para actualización dinámica de tiempos
"""

from typing import Any, Dict

from config import TRAFFIC_LIGHT_CONFIG
from utils.logger import setup_logger
from utils.signal_utils import apply_durations_to_tls

logger = setup_logger(__name__)


class TrafficLightController:
    """
    Aplica tiempos de optimización a semáforos preservando la estructura
    de fases del programa existente (vía signal_utils).
    """

    def update_traffic_light(self, traffic_light_id: str, optimization_data: Dict[str, Any]) -> bool:
        """
        Actualiza un semáforo con nuevos tiempos de optimización.

        Args:
            traffic_light_id: ID del semáforo a actualizar
            optimization_data: {"optimization": {"green_time_sec": .., "red_time_sec": ..},
                                "priority_edge": <edge congestionado, opcional>}

        Returns:
            True si la actualización fue exitosa
        """
        try:
            optimization = optimization_data.get("optimization", {})
            green_time = optimization.get("green_time_sec", 30)
            red_time = optimization.get("red_time_sec", 30)

            min_dur = TRAFFIC_LIGHT_CONFIG["min_phase_duration"]
            max_dur = TRAFFIC_LIGHT_CONFIG["max_phase_duration"]
            green_time = max(min_dur, min(green_time, max_dur))
            red_time = max(min_dur, min(red_time, max_dur))

            success = apply_durations_to_tls(
                traffic_light_id, green_time, red_time,
                priority_edge=optimization_data.get("priority_edge"),
            )
            if success:
                logger.info(
                    f"Semáforo {traffic_light_id} actualizado - Verde: {green_time}s, Rojo: {red_time}s"
                )
            else:
                logger.warning(f"No se pudo aplicar programa a {traffic_light_id}")
            return success

        except Exception as e:
            logger.error(f"Error actualizando semáforo {traffic_light_id}: {e}")
            return False
