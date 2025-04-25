#!/usr/bin/env python3
import os
import subprocess
import traci
from time import sleep

from generate_trips import generate_trips
from config_simulation import create_config

def create_folders():
    os.makedirs("output/runs", exist_ok=True)

def convert_trips_to_routes(trips_file, route_file):
    cmd = [
        "duarouter",
        "-n", "data/net/san_lorenzo.net.xml",
        "-r", trips_file,
        "-o", route_file,
        "--ignore-errors",
        "--remove-loops",
    ]
    subprocess.run(cmd, check=True)

def run_simulation_with_traci(config_file, gui=True):
    """
    Runs a SUMO simulation using TRACI and steps through it in near real-time.

    Args:
        config_file (str): Path to the SUMO configuration file.
        gui (bool): Whether to use the graphical interface (sumo-gui) or not.
    """
    sumo_binary = "sumo-gui" if gui else "sumo"
    traci.start([sumo_binary, "-c", config_file])
    print("Simulación iniciada con TRACI...")

    try:
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # Ejemplo de análisis en cada paso
            veh_ids = traci.vehicle.getIDList()
            print(f"Paso {step} — Vehículos en red: {len(veh_ids)}")

            sleep(0.3)

            # Aquí luego puedes agregar análisis de cuellos de botella o semáforos
            step += 1
    finally:
        traci.close()
        print("Simulación finalizada.")

if __name__ == "__main__":
    create_folders()

    trips_file = "output/runs/trips.trips.xml"
    generate_trips(net_file="data/net/san_lorenzo.net.xml", output_file=trips_file)

    route_file = "output/runs/routes.rou.xml"
    convert_trips_to_routes(trips_file, route_file)

    create_config(
        net_file="data/net/san_lorenzo.net.xml",
        route_file=route_file,
    )

    # Llamada principal: correr simulación con TRACI
    run_simulation_with_traci(config_file="output/runs/scenario.sumocfg", gui=True)
