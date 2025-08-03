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
    "density_threshold": 50.0,  # vehículos por kilómetro (reducido de 100.0 - más realista)
    "speed_threshold": 15.0,    # metros por segundo (54 km/h - más realista)
    "queue_length_threshold": 3, # número de vehículos en cola (aumentado de 2)
    "detection_interval": 15,   # segundos entre detecciones (15 pasos)
    "min_detection_duration": 3, # segundos mínimos para confirmar cuello de botella
    "visible_range": 60.0,      # metros visibles desde el semáforo (realista para urbano)
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
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "traffic_sim.log",
} 