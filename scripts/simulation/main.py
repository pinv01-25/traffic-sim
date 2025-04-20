#!/usr/bin/env python3
import os
import subprocess
from generate_trips import generate_trips
from config_simulation import create_config


def create_folders():
    """
    Creates the necessary folder structure for the simulation.

    This function ensures that the following directories exist:
    - "output/runs": Used to store simulation output files.
    - "data/net/restrictions": Used to store network restriction data.

    If the directories already exist, they will not be recreated.
    """
    os.makedirs("output/runs", exist_ok=True)
    os.makedirs("data/net/restrictions", exist_ok=True)


def convert_trips_to_routes(trips_file, route_file):
    """
    Converts trip definitions into route files using the duarouter tool.

    This function takes a SUMO trips file and converts it into a routes file
    by invoking the `duarouter` command-line tool. The generated routes file
    can then be used for traffic simulation.

    Args:
        trips_file (str): Path to the input trips file (e.g., .trips.xml) containing trip definitions.
        route_file (str): Path to the output routes file (e.g., .rou.xml) to be generated.

    Raises:
        subprocess.CalledProcessError: If the `duarouter` command fails to execute successfully.
    """
    """Convierte viajes a rutas usando duarouter"""
    cmd = [
        "duarouter",
        "-n",
        "data/net/san_lorenzo.net.xml",
        "-r",
        trips_file,
        "-o",
        route_file,
        "--ignore-errors",
        "--remove-loops",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # 1. Preparar entorno
    create_folders()

    # 2. Generar viajes con patrones específicos
    trips_file = "output/runs/trips.trips.xml"
    generate_trips(net_file="data/net/san_lorenzo.net.xml", output_file=trips_file)

    # 3. Convertir a rutas
    route_file = "output/runs/routes.rou.xml"
    convert_trips_to_routes(trips_file, route_file)

    # 4. Crear configuración SUMO con restricciones
    create_config(
        net_file="data/net/san_lorenzo.net.xml",
        route_file=route_file,
        restriction_file="data/net/restrictions/truck_restrictions.xml",
    )

    # 5. Ejecutar simulación
    print("Iniciando simulación en SUMO-GUI...")
    subprocess.run(["sumo-gui", "-c", "output/runs/scenario.sumocfg"], cwd="/home/majin/traffic-sim")
