import os
import subprocess
import datetime
from lxml import etree

SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")

def generate_trips(net_file, output_file):
    """
    Generates vehicle trips for a traffic simulation based on the provided network file and outputs them to a specified file.
    This function creates trips for different types of vehicles (passenger cars, motorcycles, buses, and trucks) using 
    the SUMO `randomTrips.py` tool. Each vehicle type has its own configuration, including color, period, and time slots 
    for trip generation. The generated trips are then merged into a single output file.
    Args:
        net_file (str): Path to the network file (.net.xml) used for the simulation.
        output_file (str): Path to the output file where the merged trips will be saved (.trips.xml).
    Vehicle Configurations:
        - Passenger cars:
            - Color: Yellow
            - Period: 2.5 seconds
            - Time slots: 7:00 AM to 8:00 AM
        - Motorcycles:
            - Color: Green
            - Period: 12.0 seconds
            - Time slots: 7:00 AM to 8:00 AM
        - Buses:
            - Color: Red
            - Period: 8.0 seconds
            - Time slots: 7:00 AM to 8:00 AM
        - Trucks:
            - Color: Orange
            - Length: 16.5 meters
            - Acceleration: 1.0 m/s²
            - Period: 20.0 seconds
            - Time slots: 7:00 AM to 8:00 AM
    Notes:
        - The function includes commented-out code for additional truck restrictions based on the day of the week.
        - The `SUMO_TOOLS` environment variable must be set to the path of the SUMO tools directory.
        - The `merge_trips` function is used to combine all generated trips into a single file.
    Raises:
        subprocess.CalledProcessError: If the `randomTrips.py` script fails during execution.
    Example:
        generate_trips("network.net.xml", "output.trips.xml")
    """
    generated = []

    vehicle_configs = [
        {"id": "passenger", "color": "yellow", "period": 2.0, "slots": [("0", "3000")]},
        {"id": "motorcycle", "color": "green", "period": 15.0, "slots": [("0", "3000")]}, 
        {"id": "bus", "color": "red", "period": 6.0, "slots": [("0", "3000")]}, 
        {"id": "truck", "color": "orange", "period": 6.0, "slots": [("0", "3000")], "attributes": 'length="16.5" accel="1.0"'},
    ]

    for cfg in vehicle_configs:
        for i, (begin, end) in enumerate(cfg["slots"]):
            temp_route = os.path.join(
                os.path.dirname(output_file), 
                os.path.basename(output_file).replace(".trips.xml", f"_{cfg['id']}_{i}.rou.xml")
            )
            cmd = [
                os.path.join(SUMO_TOOLS, "randomTrips.py"),
                "-n", net_file,
                "-o", temp_route,
                "--weights-prefix", "data/net/outer_edges",
                "--speed-exponent", "0.5",
                "--edge-param", "traveltime",
                "--vehicle-class", cfg["id"],
                "--trip-attributes", f'color="{cfg["color"]}" {cfg.get("attributes", "")}',
                "--prefix", f"{cfg['id']}_{i}",
                "--begin", begin,
                "--end", end,
                "--period", str(cfg["period"]),
                "--validate",
                "--random-depart",
                "--min-distance", "50"
            ]
            subprocess.run(cmd, check=True)
            generated.append(temp_route)

    merge_trips(generated, output_file)

def merge_trips(input_files, merged_output):
    """
    Merges multiple SUMO route files into a single file, ensuring unique vehicle types
    and sorting trips or vehicles by their departure time.

    Args:
        input_files (list of str): A list of file paths to the input XML route files.
        merged_output (str): The file path for the output merged XML file.

    The function performs the following steps:
        1. Parses each input file to extract vehicle types (`vType`) and trips or vehicles
           (`trip` or `vehicle` elements).
        2. Ensures that vehicle types are unique across all input files.
        3. Sorts trips or vehicles by their `depart` attribute (departure time).
        4. Writes the merged and sorted data into a new XML file with the specified output path.

    Note:
        - The input files must be valid XML files compatible with SUMO's route file format.
        - The output file will be written in UTF-8 encoding with pretty-printed formatting.

    Example:
        merge_trips(["file1.rou.xml", "file2.rou.xml"], "merged_output.rou.xml")
    """
    from lxml import etree

    root = etree.Element("routes")
    seen_vtypes = set()
    vtypes = []
    trips_or_vehicles = []

    print("Archivos a fusionar:")
    for f in input_files:
        print(f"→", f)
        tree = etree.parse(f)
        for elem in tree.getroot():
            if elem.tag == "vType":
                vid = elem.get("id")
                if vid not in seen_vtypes:
                    vtypes.append(elem)
                    seen_vtypes.add(vid)
            elif elem.tag in ("vehicle", "trip"):
                trips_or_vehicles.append(elem)

    # Ordenar por tiempo de salida
    trips_or_vehicles.sort(key=lambda e: float(e.get("depart")))

    # Ensamblar XML final
    for vt in vtypes:
        root.append(vt)
    for item in trips_or_vehicles:
        root.append(item)

    with open(merged_output, "wb") as f:
        f.write(etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8"))
