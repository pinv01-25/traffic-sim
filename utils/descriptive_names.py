"""
Utilidad para nombres descriptivos de calles e intersecciones
Mapea IDs técnicos a nombres legibles para el logging
"""

import traci
import logging
from typing import Dict, List, Optional
from pathlib import Path
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class DescriptiveNames:
    """
    Manejador de nombres descriptivos para calles e intersecciones
    """

    def __init__(self, network_file: str = "simulation/network.net.xml"):
        self.edge_id_to_name: Dict[str, str] = {}
        self.traffic_light_to_intersection: Dict[str, str] = {}
        self.network_file = Path(network_file)

        # Cargar nombres desde el archivo de red
        self._load_network_names()

    def _load_network_names(self):
        """Carga nombres de calles e intersecciones desde el archivo de red SUMO"""
        try:
            if not self.network_file.exists():
                logger.warning(f"Archivo de red no encontrado: {self.network_file}")
                return

            tree = ET.parse(self.network_file)
            root = tree.getroot()

            # Cargar nombres de calles (edges)
            for edge in root.findall(".//edge"):
                edge_id = edge.get("id", "")
                edge_name = edge.get("name", "")

                if edge_name:
                    # Usar el nombre del archivo si está disponible
                    self.edge_id_to_name[edge_id] = edge_name
                else:
                    # Crear nombre descriptivo basado en el ID
                    self.edge_id_to_name[edge_id] = self._create_descriptive_edge_name(edge_id)

            # Cargar intersecciones con semáforos
            for junction in root.findall(".//junction[@type='traffic_light']"):
                junction_id = junction.get("id", "")
                if junction_id:
                    intersection_name = self._create_intersection_name(junction_id)
                    self.traffic_light_to_intersection[junction_id] = intersection_name

            logger.info(f"Cargados {len(self.edge_id_to_name)} nombres de calles y {len(self.traffic_light_to_intersection)} intersecciones")

        except Exception as e:
            logger.error(f"Error cargando nombres de red: {e}")

    def _create_descriptive_edge_name(self, edge_id: str) -> str:
        """Crea un nombre descriptivo para una calle basado en su ID"""
        # Si el ID ya es descriptivo, usarlo
        if "_" in edge_id and not edge_id.startswith("edge_"):
            # ID como "Calle_Principal_0" -> "Calle Principal"
            parts = edge_id.split("_")
            if len(parts) >= 2:
                # Remover el número al final si existe
                if parts[-1].isdigit():
                    parts = parts[:-1]
                return " ".join(parts).replace("_", " ")

        # Si es un ID técnico como "edge_0_1_2", crear nombre genérico
        if edge_id.startswith("edge_"):
            return f"Calle {edge_id.split('_')[1] if len(edge_id.split('_')) > 1 else 'Desconocida'}"

        # Para otros IDs, usar el ID directamente
        return edge_id.replace("_", " ")

    def _create_intersection_name(self, junction_id: str) -> str:
        """Crea un nombre descriptivo para una intersección"""
        try:
            # Obtener calles conectadas a esta intersección
            controlled_lanes = traci.trafficlight.getControlledLanes(junction_id)
            street_names = set()

            for lane in controlled_lanes:
                if "_" in lane:
                    edge_id = lane.rsplit("_", 1)[0]
                    street_name = self.get_edge_name(edge_id)
                    if street_name and street_name != edge_id:
                        street_names.add(street_name)

            if len(street_names) >= 2:
                # Crear nombre de intersección con las calles principales
                street_list = list(street_names)[:3]  # Máximo 3 calles
                return " y ".join(street_list)
            elif len(street_names) == 1:
                return f"Intersección {list(street_names)[0]}"
            else:
                return f"Intersección {junction_id}"

        except Exception as e:
            logger.warning(f"Error creando nombre de intersección para {junction_id}: {e}")
            return f"Intersección {junction_id}"

    def get_edge_name(self, edge_id: str) -> str:
        """Obtiene el nombre descriptivo de una calle"""
        return self.edge_id_to_name.get(edge_id, edge_id)

    def get_intersection_name(self, traffic_light_id: str) -> str:
        """Obtiene el nombre descriptivo de una intersección"""
        return self.traffic_light_to_intersection.get(traffic_light_id, traffic_light_id)

    def get_controlled_streets(self, traffic_light_id: str) -> List[str]:
        """Obtiene las calles controladas por un semáforo"""
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
            street_names = set()

            for lane in controlled_lanes:
                if "_" in lane:
                    edge_id = lane.rsplit("_", 1)[0]
                    street_name = self.get_edge_name(edge_id)
                    street_names.add(street_name)

            return list(street_names)

        except Exception as e:
            logger.warning(f"Error obteniendo calles controladas para {traffic_light_id}: {e}")
            return [traffic_light_id]

# Instancia global
descriptive_names = DescriptiveNames() 