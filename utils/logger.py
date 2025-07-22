"""
Sistema de logging para traffic-sim
"""

import logging
import os
from datetime import datetime
from config import LOGGING_CONFIG

def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """
    Configura un logger con el nombre especificado
    
    Args:
        name: Nombre del logger
        log_file: Archivo de log (opcional)
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Evitar duplicar handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, LOGGING_CONFIG["level"]))
    
    # Formato del log
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (si se especifica)
    if log_file:
        # Crear directorio de logs si no existe
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_simulation_logger() -> logging.Logger:
    """
    Obtiene el logger principal para la simulación
    
    Returns:
        Logger configurado para la simulación
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/simulation_{timestamp}.log"
    return setup_logger("traffic_sim", log_file) 