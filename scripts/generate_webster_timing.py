#!/usr/bin/env python3
"""
Genera temporización Webster óptima para sim_C (comparador adaptativo).

Aplica la fórmula de Webster a los volúmenes de tráfico observados en
sim_A para calcular tiempos de ciclo y verde óptimos por semáforo.

Fórmula de Webster:
    C_0 = (1.5·L + 5) / (1 − Y)
    g_i = (C_0 − L) · y_i / Y

    donde:
        L  = tiempo perdido total = n_green_phases × 4 s
        y_i = volumen_critico_fase_i / s_i  (s_i = 1800 veh/h = 0.5 veh/s)
        Y  = Σ y_i (ratio de saturación total)

Output:
    sim_C/traffic_lights.add.xml  — temporización Webster para todos los semáforos
    sim_C/                        — copia de estructura sim_A lista para ejecutar

Uso:
    uv run python scripts/generate_webster_timing.py
"""

import math
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._sim_utils import require_sim_dir  # noqa: E402

SIM_A_DIR = require_sim_dir("A")
SIM_C_DIR = PROJECT_ROOT / "sim_C"

NETWORK_FILE = SIM_A_DIR / "network.net.xml"
ROUTES_FILE = SIM_A_DIR / "routes.rou.xml"

# Webster constants
SAT_FLOW_VEH_PER_SEC = 0.5      # 1800 veh/h/lane = 0.5 veh/s/lane
LOST_TIME_PER_PHASE = 4.0        # seconds lost per green phase (standard)
MIN_CYCLE = 40.0                  # seconds
MAX_CYCLE = 120.0
MIN_GREEN = 5.0                   # minimum green per phase
AMBER_DURATION = 3.0              # yellow duration


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def parse_tl_logics(network_file: Path) -> dict:
    """
    Parse <tlLogic> elements from network.net.xml.

    Returns:
        {tl_id: [{"duration": float, "state": str}, ...]}
    """
    tl_logics = {}
    for _, elem in ET.iterparse(str(network_file)):
        if elem.tag == "tlLogic":
            tl_id = elem.get("id")
            phases = []
            for ph in elem.findall("phase"):
                phases.append({
                    "duration": float(ph.get("duration", 30)),
                    "state": ph.get("state", ""),
                })
            tl_logics[tl_id] = phases
            elem.clear()
    return tl_logics


def parse_tl_connections(network_file: Path) -> dict:
    """
    Parse <connection tl="..." linkIndex="..." from="..."> from network.

    Returns:
        {tl_id: {link_index: from_edge}}
    """
    tl_connections: dict = defaultdict(dict)
    for _, elem in ET.iterparse(str(network_file)):
        if elem.tag == "connection":
            tl_id = elem.get("tl")
            link_idx = elem.get("linkIndex")
            from_edge = elem.get("from")
            if tl_id and link_idx is not None and from_edge:
                tl_connections[tl_id][int(link_idx)] = from_edge
            elem.clear()
    return dict(tl_connections)


def count_vehicles_per_edge(routes_file: Path) -> dict:
    """
    Count how many vehicles use each edge in their route.

    Returns:
        {edge_id: count}
    """
    edge_counts: dict = defaultdict(int)
    current_route_edges = None

    for event, elem in ET.iterparse(str(routes_file), events=["start", "end"]):
        if event == "start" and elem.tag == "route":
            edges_str = elem.get("edges", "")
            current_route_edges = edges_str.split() if edges_str else []
        elif event == "end" and elem.tag == "vehicle":
            if current_route_edges:
                for edge in current_route_edges:
                    edge_counts[edge] += 1
            current_route_edges = None
            elem.clear()
        elif event == "end" and elem.tag == "route":
            # standalone route (referenced by ID) — count edges too
            edges_str = elem.get("edges", "")
            if edges_str:
                for edge in edges_str.split():
                    # Don't double-count; vehicles will be counted separately
                    pass
            elem.clear()

    # Second pass: accumulate via vehicle → route lookup
    # (routes.rou.xml has inline routes, so the above captures them)
    return dict(edge_counts)


# ---------------------------------------------------------------------------
# Webster calculation
# ---------------------------------------------------------------------------

def is_green_state(char: str) -> bool:
    """Returns True if the signal state character represents a green phase."""
    return char in ("G", "g")


def is_amber_state(char: str) -> bool:
    return char in ("y", "Y")


def compute_webster_timing(
    tl_id: str,
    phases: list,
    link_to_edge: dict,
    edge_demand: dict,
) -> tuple:
    """
    Compute Webster optimal cycle and green times for one traffic light.

    Args:
        tl_id:         Traffic light ID
        phases:        List of {duration, state} dicts from network
        link_to_edge:  {link_index: from_edge} for this TL
        edge_demand:   {edge_id: vehicle_count} from routes

    Returns:
        (cycle_time, green_phases) where green_phases is a list of
        {"duration": float, "state": str} for the optimized program.
        Falls back to original phases if Webster produces invalid results.
    """
    if not phases:
        return None, phases

    # Identify green phases (exclude amber and all-red)
    green_phase_idxs = []
    amber_total = 0.0

    for i, ph in enumerate(phases):
        state = ph["state"]
        if not state:
            continue
        if all(is_amber_state(c) or c == "r" for c in state) and any(is_amber_state(c) for c in state):
            amber_total += ph["duration"]
        elif any(is_green_state(c) for c in state):
            green_phase_idxs.append(i)

    if not green_phase_idxs:
        return None, phases

    n_green = len(green_phase_idxs)
    L = n_green * LOST_TIME_PER_PHASE  # total lost time

    # Compute y_i for each green phase
    y_values = []
    for i in green_phase_idxs:
        state = phases[i]["state"]
        # Identify green links in this phase
        green_link_idxs = [j for j, c in enumerate(state) if is_green_state(c)]

        # Find max demand among green links
        max_demand_per_sec = 0.0
        for link_idx in green_link_idxs:
            edge = link_to_edge.get(link_idx)
            if edge:
                veh_count = edge_demand.get(edge, 0)
                # Convert to flow: vehicles per second (rough, whole simulation = 3600s)
                flow = veh_count / 3600.0
                max_demand_per_sec = max(max_demand_per_sec, flow)

        y_i = max_demand_per_sec / SAT_FLOW_VEH_PER_SEC
        y_values.append(y_i)

    Y = sum(y_values)

    # Guard: if Y >= 0.9, network is oversaturated — cap at 0.85
    if Y <= 0.0:
        # No demand data, use original timings
        return None, phases
    if Y >= 0.90:
        Y = 0.85

    # Webster cycle
    C0_raw = (1.5 * L + 5) / (1 - Y)
    C0 = max(MIN_CYCLE, min(MAX_CYCLE, C0_raw))

    # Green splits
    effective_green = C0 - L
    # Distribute across green phases proportional to y_i
    total_y = max(sum(y_values), 1e-9)

    new_greens = []
    for y_i in y_values:
        g_i = effective_green * y_i / total_y
        g_i = max(MIN_GREEN, g_i)
        new_greens.append(g_i)

    # Rescale to fit in C0
    total_green = sum(new_greens)
    total_amber = n_green * AMBER_DURATION
    available = C0 - total_amber
    if total_green > available:
        scale = available / total_green
        new_greens = [max(MIN_GREEN, g * scale) for g in new_greens]

    # Build output phases preserving amber phases, replacing green durations
    out_phases = []
    green_idx = 0
    for i, ph in enumerate(phases):
        state = ph["state"]
        if i in green_phase_idxs:
            out_phases.append({"duration": round(new_greens[green_idx], 1), "state": state})
            green_idx += 1
        else:
            # Keep amber/all-red phases as-is (or normalize amber to 3s)
            d = AMBER_DURATION if any(is_amber_state(c) for c in state) else ph["duration"]
            out_phases.append({"duration": d, "state": state})

    return C0, out_phases


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

def write_additional_xml(tl_timings: dict, out_file: Path) -> None:
    """
    Write sim_C/traffic_lights.add.xml with Webster-optimized tlLogic entries.

    tl_timings: {tl_id: [(duration, state), ...]}
    """
    root = ET.Element("additional")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set(
        "xsi:noNamespaceSchemaLocation",
        "http://sumo.dlr.de/xsd/additional_file.xsd",
    )

    for tl_id, phases in sorted(tl_timings.items()):
        tl_elem = ET.SubElement(root, "tlLogic")
        tl_elem.set("id", tl_id)
        tl_elem.set("type", "static")
        tl_elem.set("programID", "webster")
        tl_elem.set("offset", "0")
        for ph in phases:
            ph_elem = ET.SubElement(tl_elem, "phase")
            ph_elem.set("duration", str(ph["duration"]))
            ph_elem.set("state", ph["state"])

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_file), encoding="UTF-8", xml_declaration=True)


def setup_sim_c(sim_a_dir: Path, sim_c_dir: Path) -> bool:
    """
    Create sim_C directory structure as a copy of sim_A (without logs).
    Copies network, routes, sumocfg; traffic_lights.add.xml will be overwritten.
    """
    sim_c_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "network.net.xml",
        "routes.rou.xml",
        "simulation.sumocfg",
        "nodes.nod.xml",
        "edges.edg.xml",
    ]

    for fname in files_to_copy:
        src = sim_a_dir / fname
        dst = sim_c_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(dst))
            print(f"  Copied: {fname}")
        else:
            print(f"  WARNING: {src} not found, skipping")

    # Ensure logs/sumo_output directory exists
    (sim_c_dir / "logs" / "sumo_output").mkdir(parents=True, exist_ok=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"Generando temporización Webster para sim_C")
    print(f"  Network:  {NETWORK_FILE}")
    print(f"  Routes:   {ROUTES_FILE}")
    print(f"  Output:   {SIM_C_DIR}")
    print()

    if not NETWORK_FILE.exists():
        print(f"ERROR: {NETWORK_FILE} no encontrado")
        return 1
    if not ROUTES_FILE.exists():
        print(f"ERROR: {ROUTES_FILE} no encontrado")
        return 1

    # 1. Parse network
    print("Parseando red SUMO...")
    tl_logics = parse_tl_logics(NETWORK_FILE)
    tl_connections = parse_tl_connections(NETWORK_FILE)
    print(f"  Semáforos encontrados: {len(tl_logics)}")
    for tl_id, phases in tl_logics.items():
        print(f"    {tl_id}: {len(phases)} phases, "
              f"connections={list(tl_connections.get(tl_id, {}).keys())}")

    # 2. Count vehicle demand per edge
    print("\nContando volúmenes de tráfico por edge...")
    edge_demand = count_vehicles_per_edge(ROUTES_FILE)
    total_vehs = len(set(edge_demand.keys()))
    print(f"  Edges con demanda: {total_vehs}")

    # 3. Compute Webster timing for each TL
    print("\nCalculando tiempos Webster...")
    tl_timings = {}

    for tl_id, phases in tl_logics.items():
        link_to_edge = tl_connections.get(tl_id, {})
        cycle, out_phases = compute_webster_timing(tl_id, phases, link_to_edge, edge_demand)

        if cycle is not None:
            print(f"  {tl_id}: C0={cycle:.1f}s, green phases="
                  f"{[p['duration'] for p in out_phases if any(is_green_state(c) for c in p['state'])]}")
        else:
            print(f"  {tl_id}: sin demanda — usando fases originales")

        tl_timings[tl_id] = out_phases

    # 4. Setup sim_C directory
    print(f"\nCreando estructura sim_C en {SIM_C_DIR}...")
    setup_sim_c(SIM_A_DIR, SIM_C_DIR)

    # 5. Write traffic_lights.add.xml
    add_xml_path = SIM_C_DIR / "traffic_lights.add.xml"
    write_additional_xml(tl_timings, add_xml_path)
    print(f"\nTemporización Webster escrita: {add_xml_path}")

    # 6. Print summary
    n_tl_with_data = sum(1 for tl_id in tl_logics if tl_connections.get(tl_id))
    print(f"\n{'='*60}")
    print(f"RESUMEN WEBSTER")
    print(f"{'='*60}")
    print(f"Total semáforos:       {len(tl_logics)}")
    print(f"Con datos de conexión: {n_tl_with_data}")
    print(f"{'='*60}")

    print(f"\nSiguiente paso:")
    print(f"  uv run python scripts/run_sim_c.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
