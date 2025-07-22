"""
Configuración del sistema de simulación de tráfico
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de la simulación
SIMULATION_CONFIG = {
    "begin_time": 0,
    "end_time": 3600,  # 1 hora
    "step_length": 1.0,  # segundos por paso
    "max_vehicles": 1000,
}

# Configuración de detección de cuellos de botella
BOTTLENECK_CONFIG = {
    "density_threshold": 8.0,   # vehículos por kilómetro (más sensible)
    "speed_threshold": 10.0,    # metros por segundo (36 km/h)
    "queue_length_threshold": 2, # número de vehículos en cola (muy sensible)
    "detection_interval": 10,   # segundos entre detecciones (muy frecuente)
    "min_detection_duration": 3, # segundos mínimos para confirmar cuello de botella
}

# Configuración de comunicación con traffic-control
TRAFFIC_CONTROL_CONFIG = {
    "base_url": os.getenv("TRAFFIC_CONTROL_URL", "http://localhost:8000"),
    "timeout": 30,  # segundos
    "retry_attempts": 3,
    "retry_delay": 5,  # segundos
}

# Configuración de semáforos
TRAFFIC_LIGHT_CONFIG = {
    "min_phase_duration": 10,  # segundos mínimos por fase
    "max_phase_duration": 120,  # segundos máximos por fase
    "default_cycle_length": 90,  # segundos por ciclo completo
    "yellow_time": 3,  # segundos de luz amarilla
    "all_red_time": 1,  # segundos de todas las luces rojas
}

# Configuración de logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "traffic_sim.log",
} 