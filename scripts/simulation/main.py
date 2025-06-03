from collections import deque
import json
import os
import subprocess
import traci
from lxml import etree
from time import sleep
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from generate_trips import generate_trips
from config_simulation import create_config

load_dotenv()

VISIBLE_RANGE = float(os.getenv("VISIBLE_RANGE"))
SHOW_GUI = bool(os.getenv("SHOW_GUI"))
recent_detections = deque(maxlen=1000)

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


def monitor_congestion_normalized(step, net_info, last_alert_sent, interval=30, threshold=0.1):
    tls_ids = traci.trafficlight.getIDList()

    for tls_id in tls_ids:
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        edges = set(lane.split("_")[0] for lane in controlled_lanes)

        for edge_id in edges:
            key = (tls_id, edge_id)
            if key in last_alert_sent and step - last_alert_sent[key] < interval:
                continue
            
            try:
                # Contar solo vehículos dentro del rango visible
                visible_vehicles = []
                lane_id = edge_id + "_0"  # Tomamos el primer carril como referencia
                lane_length = traci.lane.getLength(lane_id)
                
                for v_id in traci.edge.getLastStepVehicleIDs(edge_id):
                    try:
                        v_lane = traci.vehicle.getLaneID(v_id)
                        if v_lane.startswith(edge_id):  # Asegurarse de que el vehículo está en la misma edge
                            pos = traci.vehicle.getLanePosition(v_id)
                            # Comprobar si el vehículo está dentro del rango visible
                            if lane_length - pos <= VISIBLE_RANGE:
                                visible_vehicles.append(v_id)
                    except traci.exceptions.TraCIException:
                        continue
                
                # Calcular densidad solo con vehículos visibles
                vehs_in_range = len(visible_vehicles)
                # Usar el rango visible como el denominador para la densidad
                # (en lugar de toda la longitud del borde)
                visible_segment_length = min(VISIBLE_RANGE, lane_length)
                density = vehs_in_range / visible_segment_length if visible_segment_length > 0 else 0
                
                if density > threshold and vehs_in_range > 0:
                    edge_name = net_info.get_edge_name(edge_id)
                    intersection = ", ".join(net_info.get_tls_streets(tls_id))
                    print(f"[PASO {step}] Alta densidad en '{edge_name}' cerca de la intersección con {intersection}")
                    print(f"[PASO {step}] Vehículos visibles: {vehs_in_range}, Densidad: {density:.3f}")
                    send_congestion_alert(tls_id, edge_id, net_info, density, step, visible_vehicles)
                    last_alert_sent[key] = step
                    
            except traci.exceptions.TraCIException as e:
                print(f"[PASO {step}] Error TraCI en {edge_id}: {e}")
                continue
            except Exception as e:
                print(f"[PASO {step}] Error en {edge_id}: {e}")
                continue


def send_congestion_alert(tls_id, edge_id, net_info, density, step, visible_vehicles):
    global recent_detections

    vehicle_types = {
        "motorcycle": 0,
        "car": 0,
        "bus": 0,
        "truck": 0
    }

    # Solo procesar vehículos ya verificados como visibles
    for v_id in visible_vehicles:
        # Track first-time sightings for rate tracking
        if v_id not in [vid for vid, _ in recent_detections]:
            recent_detections.append((v_id, step))

        # Vehicle type classification
        try:
            v_type = traci.vehicle.getVehicleClass(v_id)
            if v_type == "motorcycle":
                vehicle_types["motorcycle"] += 1
            elif v_type == "passenger":
                vehicle_types["car"] += 1
            elif v_type == "bus":
                vehicle_types["bus"] += 1
            elif v_type in ["truck", "delivery"]:
                vehicle_types["truck"] += 1
        except traci.exceptions.TraCIException:
            continue

    # Calculate real-time metrics only for visible vehicles
    try:
        avg_speed_kmh = sum(
            traci.vehicle.getSpeed(v_id) * 3.6 for v_id in visible_vehicles
        ) / len(visible_vehicles) if visible_vehicles else 0
    except Exception as e:
        print(f"[PASO {step}] Error al calcular velocidad promedio: {e}")
        avg_speed_kmh = 0

    try:
        avg_wait_sec = sum(
            traci.vehicle.getWaitingTime(v_id) for v_id in visible_vehicles
        ) / len(visible_vehicles) if visible_vehicles else 0
    except Exception as e:
        print(f"[PASO {step}] Error al calcular tiempo de espera: {e}")
        avg_wait_sec = 0

    # Count how many vehicles entered the field of view in the last 60 steps (approx. seconds)
    vehicle_rate = sum(1 for _, t in recent_detections if step - t <= 60)

    controlled_edges = list(net_info.get_tls_streets(tls_id))
    if edge_id not in controlled_edges:
        controlled_edges.insert(0, edge_id)

    payload = {
        "version": "1.0",
        "type": "data",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "traffic_light_id": tls_id,
        "controlled_edges": controlled_edges,
        "metrics": {
            "vehicles_per_minute": vehicle_rate,
            "avg_speed_kmh": round(avg_speed_kmh, 2),
            "avg_circulation_time_sec": round(avg_wait_sec, 2),
            "density": round(density, 2),
            "visible_vehicle_count": len(visible_vehicles)  # Añadir el conteo explícito de vehículos visibles
        },
        "vehicle_stats": vehicle_types
    }

    # Save the payload locally
    try:
        os.makedirs("output/alerts", exist_ok=True)
        file_path = f"output/alerts/congestion_alert_step_{step}.json"
        with open(file_path, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"[PASO {step}] Alerta guardada localmente en {file_path}")
    except Exception as e:
        print(f"[PASO {step}] Error al guardar alerta localmente: {e}")

    # Send to control
    try:
        load_dotenv()
        control_url = os.getenv("CONTROL_API_URL").rstrip("/")
        full_url = f"{control_url}/process"
        print(f"[PASO {step}] Enviando evento a API → {full_url}")
        r = requests.post(full_url, json=payload)
        r.raise_for_status()
        print(f"[PASO {step}] Enviando evento a API → status {r.status_code}")
    except Exception as e:
        print(f"[PASO {step}] Error al enviar evento: {e}")


def run_simulation_with_traci(config_file, gui=True):
    net_info = NetInfo("data/net/san_lorenzo.net.xml")
    sumo_binary = "sumo" if gui else "sumo"
    traci.start([sumo_binary, "-c", config_file])
    last_alert_sent = {}

    print("Simulación iniciada con TRACI...")
    print(f"Rango visible configurado: {VISIBLE_RANGE} metros")

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

    run_simulation_with_traci(config_file="output/runs/scenario.sumocfg", gui=SHOW_GUI)