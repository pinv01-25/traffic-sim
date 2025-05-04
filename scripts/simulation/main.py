import json
import os
import subprocess
import traci
from lxml import etree
from time import sleep
import requests
from datetime import datetime
from dotenv import load_dotenv
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
            name = edge.get("name") or eid 

            ref = None
            for param in edge.xpath("param[@key='ref']"):
                ref = param.get("value")
                break

            final_name = ref or name or eid
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

def monitor_congestion_normalized(step, net_info, last_alert_sent, intervalo=60, densidad_umbral=0.1):
    tls_ids = traci.trafficlight.getIDList()

    for tls_id in tls_ids:
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        edges = set(lane.split("_")[0] for lane in controlled_lanes)

        for edge_id in edges:
            key = (tls_id, edge_id)
            if key in last_alert_sent and step - last_alert_sent[key] < intervalo:
                continue
            try:
                vehs = traci.edge.getLastStepVehicleNumber(edge_id)
                length = traci.lane.getLength(edge_id + "_0")
                densidad = vehs / length if length > 0 else 0

                if densidad > densidad_umbral:
                    edge_name = net_info.get_edge_name(edge_id)
                    intersection = ", ".join(net_info.get_tls_streets(tls_id))
                    print(f"[PASO {step}] Alta densidad en '{edge_name}' cerca de la intersección con {intersection}")
                    
                    send_congestion_alert(tls_id, edge_id, net_info, densidad, step)
                    last_alert_sent[key] = step
            
            except traci.exceptions.TraCIException:
                continue

def send_congestion_alert(tls_id, edge_id, net_info, density, step):
    vehs = traci.edge.getLastStepVehicleNumber(edge_id)
    avg_speed = traci.edge.getLastStepMeanSpeed(edge_id) * 3.6  # m/s → km/h
    avg_waiting_time = traci.edge.getWaitingTime(edge_id)  # seg

    controlled_edges = list(net_info.get_tls_streets(tls_id))
    if edge_id not in controlled_edges:
        controlled_edges.insert(0, edge_id)

    vehicle_types = {
        "motorcycle": 0,
        "car": 0,
        "bus": 0,
        "truck": 0
    }

    for v_id in traci.edge.getLastStepVehicleIDs(edge_id):
        v_type = traci.vehicle.getVehicleClass(v_id)
        if v_type == "motorcycle":
            vehicle_types["motorcycle"] += 1
        elif v_type in ["passenger"]:
            vehicle_types["car"] += 1
        elif v_type == "bus":
            vehicle_types["bus"] += 1
        elif v_type in ["truck", "delivery"]:
            vehicle_types["truck"] += 1

    payload = {
        "version": "1.0",
        "type": "data",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "traffic_light_id": tls_id,
        "controlled_edges": controlled_edges,
        "metrics": {
            "vehicles_per_minute": vehs * 60,
            "avg_speed_kmh": round(avg_speed, 2),
            "avg_circulation_time_sec": round(avg_waiting_time, 2),
            "density": round(density, 2)
        },
        "vehicle_stats": vehicle_types
    }

    try:
        load_dotenv()
        storage_api_url = os.getenv("STORAGE_API_URL")
        r = requests.post(storage_api_url, json=payload)
        r.raise_for_status()
        print(f"[PASO {step}] Enviando evento a API → status {r.status_code}")
    except Exception as e:
        print(f"[PASO {step}] Error al enviar evento: {e}")

def run_simulation_with_traci(config_file, gui=True):
    net_info = NetInfo("data/net/san_lorenzo.net.xml")
    sumo_binary = "sumo-gui" if gui else "sumo"
    traci.start([sumo_binary, "-c", config_file])
    last_alert_sent = {}

    print("Simulación iniciada con TRACI...")

    try:
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            veh_ids = traci.vehicle.getIDList()
            print(f"Paso {step} — Vehículos en red: {len(veh_ids)}")
            monitor_congestion_normalized(step, net_info, last_alert_sent)

            sleep(0.3)
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

    run_simulation_with_traci(config_file="output/runs/scenario.sumocfg", gui=True)
