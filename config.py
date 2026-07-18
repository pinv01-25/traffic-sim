"""
Configuración del sistema de simulación de tráfico
"""

import os

from dotenv import load_dotenv

# Cargar .env pero no sobrescribir variables de entorno existentes
# Esto permite que las variables de Docker Compose tengan prioridad
load_dotenv(override=False)

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
    "speed_threshold": 15.0,    # km/h (se compara contra avg_speed_kmh del detector)
    "queue_length_threshold": 3, # número de vehículos en cola (aumentado de 2)
    "detection_interval": 15,   # segundos entre detecciones (15 pasos)
    "min_detection_duration": 3, # segundos mínimos para confirmar cuello de botella
    "visible_range": 60.0,      # metros visibles desde el semáforo (realista para urbano)
}

# Configuración de comunicación con traffic-control
# Prioridad: CONTROL_URL > CONTROL_API_URL > TRAFFIC_CONTROL_URL > localhost
# Leer directamente de os.environ para evitar problemas con load_dotenv
control_url = os.environ.get("CONTROL_URL") or os.environ.get("CONTROL_API_URL") or os.environ.get("TRAFFIC_CONTROL_URL", "http://localhost:8003")
TRAFFIC_CONTROL_CONFIG = {
    "base_url": control_url,
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

# Configuración del controlador local Max-Pressure (Varaiya): reactivo por
# presión de carriles, reemplaza el reparto proporcional por ciclo.
MAX_PRESSURE_CONFIG = {
    "check_interval": 5.0,        # segundos entre decisiones de cambio de fase
    "hysteresis_factor": 1.3,    # margen mínimo para forzar un cambio (evita flapping)
    "starvation_bonus_per_sec": 0.05,  # presión artificial por segundo sin servicio
    "min_green_floor": 0.4,       # piso del verde mínimo con red vacía (adaptividad máxima)
    "min_green_floor_high": 0.4,  # piso con red cargada (~verde del plan fijo; menos conmutaciones)
    "occupancy_full_scale": 0.5,   # lane_queue_ratio medio al que el piso llega a su máximo
    "green_max_multiplier": 1.5,  # verde máximo duro = verde_base_equitativo * este factor
    "all_red_time": 1.0,          # despeje entre amarillo y verde entrante
    # Detector de estancamiento: si una fase recibe verde N veces seguidas
    # y su cola de entrada no baja, es que el bloqueo está aguas abajo (no
    # es un problema de reparto de verde) — excluirla un tiempo evita
    # seguir quemando ciclos en un movimiento que no puede drenar.
    "stall_limit": 3,             # servicios seguidos sin bajar la cola antes de excluir
    "stall_cooldown_sec": 60.0,   # cooldown base; se duplica en cada re-estancamiento
    "stall_cooldown_max_sec": 240.0,  # techo del backoff exponencial
    "downstream_discount_weight": 1.0,  # peso del término de cola de salida en la presión
    # Conmutación consciente de congestión: con la red llena, cada cambio de
    # fase quema yellow_time + all_red_time justo cuando cada segundo de
    # verde vale más caro (medido: demanda_media/alta pierden 4-15% contra
    # fixed-time por exceso de conmutación bajo saturación). Por encima de
    # este umbral de ocupación local se frena tanto la histéresis como el
    # verde mínimo, para conmutar menos seguido cuando más cuesta.
    "congestion_ratio_threshold": 99.0,   # lane_queue_ratio medio de entrada a partir del cual se considera "congestionado"
    "congested_hysteresis_boost": 1.25,  # multiplica hysteresis_factor bajo congestión (1.15 -> ~1.44)
    "congested_min_green_boost": 1.5,    # multiplica el verde mínimo efectivo bajo congestión
    # Gate duro anti-spillback: ocupación física (getLastStepOccupancy) de un
    # carril de SALIDA a partir de la cual una fase que lo alimenta pierde
    # siempre frente a alternativas que no. lane_queue_ratio (halting) no ve
    # un carril lleno pero fluyendo -> deadlock físico observado en
    # demanda_media/seed_43 (vehículos varados dentro del cruce). Ver
    # controllers/max_pressure.py:SPILLBACK_OCCUPANCY_GATE.
    "spillback_occupancy_gate": 0.85,
    # Velocidad media (m/s) de un carril de SALIDA por debajo de la cual se
    # considera realmente parado (no solo lleno). Sin este chequeo, el gate
    # de ocupación por sí solo frenaba carriles llenos-pero-drenando
    # (ocupación alta, velocidad normal) y colapsaba la inserción de SUMO
    # en demanda_media/42-43 y demanda_baja/44 — ver
    # controllers/max_pressure.py:SPILLBACK_SPEED_THRESHOLD.
    "spillback_speed_threshold": 0.5,
    # --- Supervisor de gridlock físico ---
    # Con --time-to-teleport -1 (sin teletransporte de vehículos varados),
    # Max-Pressure a veces empuja la red hacia un colapso irreversible:
    # vehículos detenidos dentro de los cruces que ya no pueden despejarse
    # solos. Cada ajuste fino de la presión mueve el escenario que colapsa
    # (whack-a-mole: demanda_media/42 vs hora_pico/43) en vez de eliminar el
    # riesgo. Esta capa detecta la formación temprana del atasco y devuelve
    # el control al plan fijo original de SUMO — que nunca se bloquea —
    # hasta que la red drena.
    "freeze_check_interval": 10.0,   # segundos entre muestras del detector de congelamiento
    "freeze_trigger_sec": 60.0,      # segundos seguidos por sobre freeze_halt_fraction para disparar recuperación
    "freeze_halt_fraction": 0.7,     # fracción de vehículos detenidos (v<0.1 m/s) que se considera "camino al gridlock"
    "freeze_min_vehicles": 10,       # con menos vehículos en red, la fracción detenida es ruido: no dispara
    "recovery_duration_sec": 240.0,  # tiempo que MP cede el control al plan fijo original antes de reintentar
}

# Configuración de logging
LOGGING_CONFIG = {
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "traffic_sim.log",
} 