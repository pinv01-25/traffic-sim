import os

def create_config(net_file, route_file):
    """
    Generates a SUMO simulation configuration file.

    This function creates a SUMO configuration file (.sumocfg) with the specified
    network, route, and restriction files. The generated file is saved in the
    "output/runs/" directory with the name "scenario.sumocfg".

    Args:
        net_file (str): Path to the network file (.net.xml).
        route_file (str): Path to the route file (.rou.xml).
        restriction_file (str): Path to the restriction file (e.g., additional settings).

    The generated configuration file includes:
        - The absolute paths of the provided network, route, and restriction files.
        - A setting to disable teleportation by setting "time-to-teleport" to -1.

    Raises:
        OSError: If there is an issue writing to the output file.
    """
    net_file_abs = os.path.abspath(net_file)
    route_file_abs = os.path.abspath(route_file)

    with open("output/runs/scenario.sumocfg", "w") as f:
        f.write(f"""
        <configuration>
          <input>
            <net-file value="{net_file_abs}"/>
            <route-files value="{route_file_abs}"/>
            <time-to-teleport value="-1"/> 
          </input>
        </configuration>
        """)
