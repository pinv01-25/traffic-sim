"""
Validador de métricas de tráfico para verificar que los valores sean realistas
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from utils.descriptive_names import descriptive_names

logger = logging.getLogger(__name__)

@dataclass
class MetricsValidationResult:
    """Resultado de validación de métricas"""
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    suggestions: List[str]

class MetricsValidator:
    """
    Validador de métricas de tráfico para detectar valores irreales
    """
    
    # Rangos realistas para tráfico urbano
    REALISTIC_RANGES = {
        "density": {
            "min": 0.0,
            "max": 200.0,  # Máximo realista en tráfico urbano (sobre 60m visibles)
            "unit": "veh/km"
        },
        "avg_speed_kmh": {
            "min": 0.0,
            "max": 80.0,  # Máximo en zonas urbanas
            "unit": "km/h"
        },
        "vehicles_per_minute": {
            "min": 0,
            "max": 100,  # Máximo por carril
            "unit": "veh/min"
        },
        "avg_circulation_time_sec": {
            "min": 0.0,
            "max": 300.0,  # Máximo 5 minutos de espera
            "unit": "segundos"
        },
        "vehicle_count": {
            "min": 0,
            "max": 50,  # Máximo vehículos visibles en 60m
            "unit": "vehículos"
        }
    }
    
    # Patrones sospechosos
    SUSPICIOUS_PATTERNS = {
        "zero_speed_with_vehicles": "Velocidad 0 km/h con vehículos presentes",
        "extreme_density": "Densidad > 150 veh/km (muy alta para 60m visibles)",
        "very_low_speed": "Velocidad < 5 km/h (muy baja)",
        "high_vpm": "Vehículos por minuto > 80 (muy alto)",
        "long_wait_time": "Tiempo de circulación > 180 segundos (muy largo)"
    }
    
    def validate_metrics(self, metrics: Dict[str, Any], edge_id: str = "unknown") -> MetricsValidationResult:
        """
        Valida que las métricas sean realistas
        
        Args:
            metrics: Diccionario con las métricas a validar
            edge_id: ID del edge para logging
            
        Returns:
            Resultado de validación con warnings y errores
        """
        warnings = []
        errors = []
        suggestions = []
        
        # Validar cada métrica
        for metric_name, value in metrics.items():
            if metric_name in self.REALISTIC_RANGES:
                validation = self._validate_single_metric(metric_name, value, edge_id)
                warnings.extend(validation["warnings"])
                errors.extend(validation["errors"])
                suggestions.extend(validation["suggestions"])
        
        # Validar patrones sospechosos
        pattern_validation = self._validate_suspicious_patterns(metrics, edge_id)
        warnings.extend(pattern_validation["warnings"])
        errors.extend(pattern_validation["errors"])
        
        # Determinar si las métricas son válidas
        is_valid = len(errors) == 0
        
        return MetricsValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            suggestions=suggestions
        )
    
    def _validate_single_metric(self, metric_name: str, value: Any, edge_id: str) -> Dict[str, List[str]]:
        """Valida una métrica individual"""
        warnings = []
        errors = []
        suggestions = []
        
        range_config = self.REALISTIC_RANGES[metric_name]
        
        try:
            value_float = float(value)
            
            # Verificar rango mínimo
            if value_float < range_config["min"]:
                errors.append(f"{metric_name}: {value_float} < {range_config['min']} {range_config['unit']}")
            
            # Verificar rango máximo
            if value_float > range_config["max"]:
                warnings.append(f"{metric_name}: {value_float} > {range_config['max']} {range_config['unit']} (muy alto)")
                suggestions.append(f"Verificar cálculo de {metric_name} para edge {edge_id}")
            
            # Verificar valores exactos sospechosos
            # CORRECCIÓN: VPM=0 es normal al inicio de simulación, no generar warning
            if value_float == 0.0 and metric_name == "avg_speed_kmh":
                # Velocidad 0 puede indicar error, pero no siempre
                pass  # Ya se maneja en patrones sospechosos
                
        except (ValueError, TypeError):
            errors.append(f"{metric_name}: valor '{value}' no es numérico")
        
        return {
            "warnings": warnings,
            "errors": errors,
            "suggestions": suggestions
        }
    
    def _validate_suspicious_patterns(self, metrics: Dict[str, Any], edge_id: str) -> Dict[str, List[str]]:
        """Valida patrones sospechosos en las métricas"""
        warnings = []
        errors = []
        
        # Patrón: velocidad 0 con vehículos presentes
        # CORRECCIÓN: No es sospechoso si hay pocos vehículos (normal en semáforos)
        if (metrics.get("avg_speed_kmh", 0) == 0.0 and 
            metrics.get("vehicle_count", 0) > 0):
            # Solo warning si hay muchos vehículos detenidos (posible congestión)
            if metrics.get("vehicle_count", 0) > 5:
                warnings.append(f"Patrón sospechoso: velocidad 0 km/h con {metrics['vehicle_count']} vehículos")
            else:
                # Vehículos detenidos en semáforo es normal
                pass
        
        # Patrón: densidad extrema
        if metrics.get("density", 0) > 150.0:
            warnings.append(f"Patrón sospechoso: densidad muy alta ({metrics['density']:.1f} veh/km)")
        
        # Patrón: velocidad muy baja (pero no 0)
        if 0 < metrics.get("avg_speed_kmh", 0) < 5.0:
            warnings.append(f"Patrón sospechoso: velocidad muy baja ({metrics['avg_speed_kmh']:.1f} km/h)")
        
        # Patrón: VPM muy alto
        if metrics.get("vehicles_per_minute", 0) > 80:
            warnings.append(f"Patrón sospechoso: VPM muy alto ({metrics['vehicles_per_minute']} veh/min)")
        
        # Patrón: tiempo de espera muy largo
        if metrics.get("avg_circulation_time_sec", 0) > 180.0:
            warnings.append(f"Patrón sospechoso: tiempo de espera muy largo ({metrics['avg_circulation_time_sec']:.1f}s)")
        
        return {
            "warnings": warnings,
            "errors": errors
        }
    
    def log_validation_result(self, result: MetricsValidationResult, edge_id: str):
        """Registra el resultado de validación"""
        edge_name = descriptive_names.get_edge_name(edge_id)
        
        if not result.is_valid:
            logger.error(f"Validación fallida para {edge_name} ({edge_id})")
            for error in result.errors:
                logger.error(f"   Error: {error}")

        if result.warnings:
            logger.warning(f"Advertencias para {edge_name} ({edge_id})")
            for warning in result.warnings:
                logger.warning(f"   Warning: {warning}")

        if result.suggestions:
            logger.info(f"Sugerencias para {edge_name} ({edge_id})")
            for suggestion in result.suggestions:
                logger.info(f"   Suggestion: {suggestion}")

        if result.is_valid and not result.warnings:
            logger.debug(f"Métricas válidas para {edge_name} ({edge_id})")

# Instancia global del validador
metrics_validator = MetricsValidator() 