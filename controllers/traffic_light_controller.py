"""
Controlador de semáforos para actualización dinámica de tiempos
"""

import traci
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from utils.logger import setup_logger
from config import TRAFFIC_LIGHT_CONFIG

logger = setup_logger(__name__)

@dataclass
class TrafficLightPhase:
    """Representa una fase de semáforo"""
    duration: int
    state: str  # String de estados (r=red, y=yellow, g=green)
    name: str = ""

@dataclass
class TrafficLightProgram:
    """Programa completo de un semáforo"""
    traffic_light_id: str
    phases: List[TrafficLightPhase]
    cycle_length: int
    offset: int = 0

class TrafficLightController:
    """
    Controlador para actualización dinámica de semáforos
    """
    
    def __init__(self):
        self.traffic_lights = {}
        self._initialize_traffic_lights()
    
    def _initialize_traffic_lights(self):
        """Inicializa el controlador con todos los semáforos disponibles"""
        try:
            traffic_light_ids = traci.trafficlight.getIDList()
            
            for tl_id in traffic_light_ids:
                # Obtener programa actual
                current_program = self._get_current_program(tl_id)
                self.traffic_lights[tl_id] = current_program
                
                logger.info(f"Semáforo {tl_id} inicializado con {len(current_program.phases)} fases")
                
        except Exception as e:
            logger.error(f"Error inicializando semáforos: {e}")
    
    def _get_current_program(self, traffic_light_id: str) -> TrafficLightProgram:
        """
        Obtiene el programa actual de un semáforo
        
        Args:
            traffic_light_id: ID del semáforo
            
        Returns:
            Programa actual del semáforo
        """
        try:
            # Obtener fases actuales
            phases = []
            phase_count = traci.trafficlight.getPhaseNumber(traffic_light_id)
            
            for i in range(phase_count):
                duration = traci.trafficlight.getPhaseDuration(traffic_light_id, i)
                state = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
                
                phase = TrafficLightPhase(
                    duration=int(duration),
                    state=state,
                    name=f"Phase_{i}"
                )
                phases.append(phase)
            
            # Calcular ciclo total
            cycle_length = sum(phase.duration for phase in phases)
            
            return TrafficLightProgram(
                traffic_light_id=traffic_light_id,
                phases=phases,
                cycle_length=cycle_length
            )
            
        except Exception as e:
            logger.error(f"Error obteniendo programa de {traffic_light_id}: {e}")
            # Retornar programa por defecto
            return self._create_default_program(traffic_light_id)
    
    def _create_default_program(self, traffic_light_id: str) -> TrafficLightProgram:
        """
        Crea un programa por defecto para un semáforo
        
        Args:
            traffic_light_id: ID del semáforo
            
        Returns:
            Programa por defecto
        """
        default_phases = [
            TrafficLightPhase(
                duration=TRAFFIC_LIGHT_CONFIG["default_cycle_length"] // 2,
                state="GGGgrrrrGGGgrrrr",  # Estado por defecto
                name="Green_Phase"
            ),
            TrafficLightPhase(
                duration=TRAFFIC_LIGHT_CONFIG["yellow_time"],
                state="yyyyrrrryyyyrrrr",
                name="Yellow_Phase"
            ),
            TrafficLightPhase(
                duration=TRAFFIC_LIGHT_CONFIG["default_cycle_length"] // 2,
                state="rrrrGGGGrrrrGGGG",
                name="Green_Phase_2"
            ),
            TrafficLightPhase(
                duration=TRAFFIC_LIGHT_CONFIG["yellow_time"],
                state="rrrryyyyrrrryyyy",
                name="Yellow_Phase_2"
            )
        ]
        
        return TrafficLightProgram(
            traffic_light_id=traffic_light_id,
            phases=default_phases,
            cycle_length=TRAFFIC_LIGHT_CONFIG["default_cycle_length"]
        )
    
    def update_traffic_light(self, traffic_light_id: str, optimization_data: Dict[str, Any]) -> bool:
        """
        Actualiza un semáforo con nuevos tiempos de optimización
        
        Args:
            traffic_light_id: ID del semáforo a actualizar
            optimization_data: Datos de optimización de traffic-control
            
        Returns:
            True si la actualización fue exitosa
        """
        try:
            logger.info(f"Actualizando semáforo {traffic_light_id} con datos de optimización")
            
            # Extraer tiempos de optimización
            green_time = optimization_data.get("optimization", {}).get("green_time_sec", 30)
            red_time = optimization_data.get("optimization", {}).get("red_time_sec", 30)
            
            # Validar tiempos
            green_time = max(TRAFFIC_LIGHT_CONFIG["min_phase_duration"], 
                           min(green_time, TRAFFIC_LIGHT_CONFIG["max_phase_duration"]))
            red_time = max(TRAFFIC_LIGHT_CONFIG["min_phase_duration"], 
                          min(red_time, TRAFFIC_LIGHT_CONFIG["max_phase_duration"]))
            
            # Crear nuevo programa
            new_program = self._create_optimized_program(traffic_light_id, green_time, red_time)
            
            # Aplicar nuevo programa
            self._apply_program(traffic_light_id, new_program)
            
            # Actualizar registro interno
            self.traffic_lights[traffic_light_id] = new_program
            
            logger.info(f"Semáforo {traffic_light_id} actualizado - Verde: {green_time}s, Rojo: {red_time}s")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando semáforo {traffic_light_id}: {e}")
            return False
    
    def _create_optimized_program(self, traffic_light_id: str, green_time: int, red_time: int) -> TrafficLightProgram:
        """
        Crea un programa optimizado basado en los nuevos tiempos
        
        Args:
            traffic_light_id: ID del semáforo
            green_time: Tiempo de luz verde en segundos
            red_time: Tiempo de luz roja en segundos
            
        Returns:
            Programa optimizado
        """
        # Obtener estados actuales para mantener la configuración
        current_states = self._get_traffic_light_states(traffic_light_id)
        
        if len(current_states) >= 2:
            # Usar estados existentes
            green_state = current_states[0]
            red_state = current_states[1]
        else:
            # Estados por defecto
            green_state = "GGGgrrrrGGGgrrrr"
            red_state = "rrrrGGGGrrrrGGGG"
        
        phases = [
            TrafficLightPhase(
                duration=green_time,
                state=green_state,
                name="Optimized_Green"
            ),
            TrafficLightPhase(
                duration=TRAFFIC_LIGHT_CONFIG["yellow_time"],
                state="yyyyrrrryyyyrrrr",
                name="Yellow"
            ),
            TrafficLightPhase(
                duration=red_time,
                state=red_state,
                name="Optimized_Red"
            ),
            TrafficLightPhase(
                duration=TRAFFIC_LIGHT_CONFIG["yellow_time"],
                state="rrrryyyyrrrryyyy",
                name="Yellow_2"
            )
        ]
        
        cycle_length = sum(phase.duration for phase in phases)
        
        return TrafficLightProgram(
            traffic_light_id=traffic_light_id,
            phases=phases,
            cycle_length=cycle_length
        )
    
    def _get_traffic_light_states(self, traffic_light_id: str) -> List[str]:
        """
        Obtiene los estados actuales de un semáforo
        
        Args:
            traffic_light_id: ID del semáforo
            
        Returns:
            Lista de estados
        """
        try:
            states = []
            phase_count = traci.trafficlight.getPhaseNumber(traffic_light_id)
            
            for i in range(phase_count):
                state = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
                if state not in states:
                    states.append(state)
            
            return states
            
        except Exception as e:
            logger.warning(f"Error obteniendo estados de {traffic_light_id}: {e}")
            return []
    
    def _apply_program(self, traffic_light_id: str, program: TrafficLightProgram):
        """
        Aplica un programa a un semáforo
        
        Args:
            traffic_light_id: ID del semáforo
            program: Programa a aplicar
        """
        try:
            # Crear definición de programa para traci
            program_definition = []
            
            for phase in program.phases:
                program_definition.append((phase.duration, phase.state))
            
            # Aplicar programa
            traci.trafficlight.setProgram(traffic_light_id, "optimized")
            traci.trafficlight.setPhaseDefinition(traffic_light_id, program_definition)
            
            logger.info(f"Programa aplicado a {traffic_light_id}: {len(program.phases)} fases")
            
        except Exception as e:
            logger.error(f"Error aplicando programa a {traffic_light_id}: {e}")
            # Fallback: usar método más simple
            self._apply_simple_program(traffic_light_id, program)
    
    def _apply_simple_program(self, traffic_light_id: str, program: TrafficLightProgram):
        """
        Método alternativo para aplicar programa (más simple)
        
        Args:
            traffic_light_id: ID del semáforo
            program: Programa a aplicar
        """
        try:
            # Usar método más básico de traci
            for i, phase in enumerate(program.phases):
                traci.trafficlight.setPhaseDuration(traffic_light_id, i, phase.duration)
            
            logger.info(f"Programa simple aplicado a {traffic_light_id}")
            
        except Exception as e:
            logger.error(f"Error aplicando programa simple a {traffic_light_id}: {e}")
    
    def get_traffic_light_status(self, traffic_light_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado actual de un semáforo
        
        Args:
            traffic_light_id: ID del semáforo
            
        Returns:
            Estado del semáforo
        """
        try:
            current_phase = traci.trafficlight.getPhase(traffic_light_id)
            current_state = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
            time_in_phase = traci.trafficlight.getNextSwitch(traffic_light_id) - traci.simulation.getTime()
            
            return {
                "traffic_light_id": traffic_light_id,
                "current_phase": current_phase,
                "current_state": current_state,
                "time_in_phase": float(time_in_phase),
                "program": self.traffic_lights.get(traffic_light_id, {}).__dict__ if traffic_light_id in self.traffic_lights else {}
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de {traffic_light_id}: {e}")
            return {"error": f"Error obteniendo estado: {e}"}
    
    def pause_simulation(self):
        """Pausa la simulación momentáneamente"""
        logger.info("Pausando simulación para actualización de semáforos...")
        # La pausa se maneja en el orquestador principal
    
    def resume_simulation(self):
        """Reanuda la simulación"""
        logger.info("Reanudando simulación después de actualización de semáforos...")
        # La reanudación se maneja en el orquestador principal 