"""
Sistema de logging para traffic-sim con colores
"""

import logging
import os
import sys
from datetime import datetime
from config import LOGGING_CONFIG

# ANSI color codes for consistent logging
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

class ColoredFormatter(logging.Formatter):
    """Custom formatter with consistent colors."""
    
    COLORS = {
        'DEBUG': Colors.DIM + Colors.CYAN,
        'INFO': Colors.GREEN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.BOLD + Colors.RED,
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Colors.END}"
        
        # Add color to service name if present
        if hasattr(record, 'service_name'):
            record.service_name = f"{Colors.BLUE}{record.service_name}{Colors.END}"
        
        return super().format(record)

def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """
    Configura un logger con el nombre especificado y colores
    
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
    
    # Formato del log con colores para consola
    console_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - [traffic-sim] - %(name)s - %(message)s")
    
    # Handler para consola con colores
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (sin colores)
    if log_file:
        # Crear directorio de logs si no existe
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_formatter = logging.Formatter(LOGGING_CONFIG["format"])
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
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