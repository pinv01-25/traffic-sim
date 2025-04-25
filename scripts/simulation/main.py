#!/usr/bin/env python3
import os
import subprocess
import traci
from lxml import etree
from time import sleep

from generate_trips import generate_trips
from config_simulation import create_config

class NetInfo:
    def __init__(self, netfile):
        self.edge_id_to_name = {}
        self.tls_id_to_connected_street_names = {}

        tree = etree.parse(netfile)
        root = tree.getroot()

        for edge in root.xpath("//edge[@id and @from and @to]"):
            eid = edge.get("id")
            name = edge.get("name") or eid  # fallback 1

            # Buscar ref si existe como child <param key="ref" value="...">
            ref = None
            for param in edge.xpath("param[@key='ref']"):
                ref = param.get("value")
                break

            final_name = ref or name or eid  # ref > name > id
            self.edge_id_to_name[eid] = final_name

        for junction in root.xpath("//junction[@type='traffic_light']"):
            jid = junction.get("id")
            incoming_names = set()
            for lane in junction.get("incLanes", "").split():
                edge_id = lane.split("_")[0]
                incoming_names.add(self.edge_id_to_name.get(edge_id, edge_id))
            self.tls_id_to_connected_street_names[jid] = incoming_names

    def get_edge_name(self, edge_id):
        return self.edge_id_to_name.get(edge_id, edge_id)

    def get_tls_streets(self, tls_id):
        return self.tls_id_to_connected_street_names.get(tls_id, [])


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

def monitor_congestion_normalized(step, net_info, densidad_umbral=0.1):
    tls_ids = traci.trafficlight.getIDList()

    for tls_id in tls_ids:
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        edges = set(lane.split("_")[0] for lane in controlled_lanes)

        for edge_id in edges:
            try:
                vehs = traci.edge.getLastStepVehicleNumber(edge_id)
                length = traci.lane.getLength(edge_id + "_0")
                densidad = vehs / length if length > 0 else 0

                if densidad > densidad_umbral:
                    edge_name = net_info.get_edge_name(edge_id)
                    intersection = ", ".join(net_info.get_tls_streets(tls_id))
                    print(f"[PASO {step}] ⚠️ Alta densidad en '{edge_name}' cerca de la intersección con {intersection}")
            except traci.exceptions.TraCIException:
                continue



def run_simulation_with_traci(config_file, gui=True):
    """
    Runs a SUMO simulation using TRACI and steps through it in near real-time.

    Args:
        config_file (str): Path to the SUMO configuration file.
        gui (bool): Whether to use the graphical interface (sumo-gui) or not.
    """
    net_info = NetInfo("data/net/san_lorenzo.net.xml")
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

            # Monitoreo de congestión
            monitor_congestion_normalized(step, net_info, 0.1)

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
