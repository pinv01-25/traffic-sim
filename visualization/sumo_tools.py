"""Wrappers for native SUMO visualization tools.

This module provides Python wrappers around SUMO's built-in visualization
scripts located in $SUMO_HOME/tools/visualization/. These tools offer
powerful plotting capabilities for comparing simulation outputs.

Reference: https://sumo.dlr.de/docs/Tools/Visualization.html
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import shutil


def _get_sumo_tools_path() -> Path:
    """Get path to SUMO tools directory."""
    sumo_home = os.environ.get('SUMO_HOME', '/usr/share/sumo')
    tools_path = Path(sumo_home) / 'tools'
    if not tools_path.exists():
        # Try common alternative locations
        alternatives = [
            Path('/usr/share/sumo/tools'),
            Path('/opt/sumo/tools'),
            Path.home() / 'sumo/tools',
        ]
        for alt in alternatives:
            if alt.exists():
                return alt
        raise FileNotFoundError(f"SUMO tools not found. Set SUMO_HOME environment variable.")
    return tools_path


def _get_python_executable() -> str:
    """Get Python executable that has matplotlib and sumolib."""
    return sys.executable


def _run_sumo_tool(tool_script: str, args: List[str], output_file: Optional[str] = None) -> Tuple[int, str, str]:
    """Run a SUMO tool script with given arguments.

    Args:
        tool_script: Name of the tool script (e.g., 'plotXMLAttributes.py')
        args: List of arguments to pass to the tool
        output_file: Optional output file path

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    tools_path = _get_sumo_tools_path()
    script_path = tools_path / 'visualization' / tool_script

    if not script_path.exists():
        # Try in sumolib location
        script_path = tools_path / tool_script
        if not script_path.exists():
            raise FileNotFoundError(f"SUMO tool {tool_script} not found at {script_path}")

    cmd = [_get_python_executable(), str(script_path)] + args
    if output_file:
        cmd.extend(['-o', output_file])

    env = os.environ.copy()
    env['SUMO_HOME'] = str(tools_path.parent)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env
    )

    return result.returncode, result.stdout, result.stderr


def plot_xml_attributes(
    input_files: List[str],
    x_attr: str,
    y_attr: str,
    output_file: str,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    id_filter: Optional[str] = None,
    plot_type: str = 'line',
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
) -> bool:
    """Run SUMO's plotXMLAttributes.py tool.

    Creates plots from arbitrary XML attributes. Very flexible for comparing
    any two attributes from SUMO output files.

    Args:
        input_files: List of XML input files (tripinfo, summary, fcd, etc.)
        x_attr: X-axis attribute (e.g., 'depart', 'time', 'begin')
        y_attr: Y-axis attribute (e.g., 'duration', 'speed', 'running')
        output_file: Output PNG file path
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        legend: Whether to show legend
        id_filter: Filter by ID pattern (regex)
        plot_type: Plot type: 'line', 'scatter', 'box', 'bar'
        colors: List of colors for each input file
        labels: List of labels for legend

    Returns:
        True if successful, False otherwise
    """
    args = []

    for i, f in enumerate(input_files):
        args.extend(['-i', f])
        if labels and i < len(labels):
            args.extend(['--label', labels[i]])

    args.extend(['-x', x_attr, '-y', y_attr])

    if title:
        args.extend(['--title', title])
    if xlabel:
        args.extend(['--xlabel', xlabel])
    if ylabel:
        args.extend(['--ylabel', ylabel])
    if not legend:
        args.append('--no-legend')
    if id_filter:
        args.extend(['--filter-id', id_filter])

    # Plot type flags
    if plot_type == 'scatter':
        args.append('--scatterplot')
    elif plot_type == 'box':
        args.append('--boxplot')
    elif plot_type == 'bar':
        args.append('--barplot')

    if colors:
        args.extend(['--colors', ','.join(colors)])

    args.extend(['-o', output_file])

    rc, stdout, stderr = _run_sumo_tool('plotXMLAttributes.py', args)

    if rc != 0:
        print(f"plotXMLAttributes failed: {stderr}")
        return False
    return True


def plot_summary(
    input_files: List[str],
    output_file: str,
    measures: Optional[List[str]] = None,
    title: Optional[str] = None,
    labels: Optional[List[str]] = None,
) -> bool:
    """Run SUMO's plot_summary.py tool.

    Creates timeline plots from summary.xml files. Useful for comparing
    vehicle counts, halting vehicles, etc. over time.

    Args:
        input_files: List of summary.xml files to compare
        output_file: Output PNG file path
        measures: List of measures to plot (e.g., ['running', 'halting'])
        title: Plot title
        labels: Legend labels for each input file

    Returns:
        True if successful, False otherwise
    """
    args = []

    for i, f in enumerate(input_files):
        args.extend(['-i', f])

    if measures:
        args.extend(['-m', ','.join(measures)])

    if title:
        args.extend(['--title', title])

    if labels:
        args.extend(['--label', ','.join(labels)])

    args.extend(['-o', output_file])

    rc, stdout, stderr = _run_sumo_tool('plot_summary.py', args)

    if rc != 0:
        print(f"plot_summary failed: {stderr}")
        return False
    return True


def plot_tripinfo_distributions(
    input_files: List[str],
    output_file: str,
    attribute: str = 'duration',
    title: Optional[str] = None,
    labels: Optional[List[str]] = None,
    bins: int = 20,
) -> bool:
    """Run SUMO's mpl_tripinfos_twoAgainst.py for comparing two tripinfo files.

    Creates overlaid histograms or density plots comparing an attribute
    from multiple simulation runs.

    Args:
        input_files: List of tripinfo.xml files (typically 2 for A/B)
        output_file: Output PNG file path
        attribute: Attribute to compare (duration, timeLoss, waitingTime, etc.)
        title: Plot title
        labels: Legend labels
        bins: Number of histogram bins

    Returns:
        True if successful, False otherwise
    """
    # Use plotXMLAttributes with histogram settings
    args = []

    for f in input_files:
        args.extend(['-i', f])

    args.extend(['-x', attribute])
    args.append('--hist')
    args.extend(['--bins', str(bins)])

    if title:
        args.extend(['--title', title])
    if labels:
        args.extend(['--label', ','.join(labels)])

    args.extend(['-o', output_file])

    rc, stdout, stderr = _run_sumo_tool('plotXMLAttributes.py', args)

    if rc != 0:
        print(f"plot_tripinfo_distributions failed: {stderr}")
        return False
    return True


def plot_net_dump(
    net_file: str,
    dump_files: List[str],
    output_file: str,
    measure: str = 'speed',
    colormap: str = 'RdYlGn',
    title: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> bool:
    """Run SUMO's plot_net_dump.py tool.

    Plots edge-based data onto the network geometry. Very useful for
    visualizing congestion patterns, speeds, or emissions spatially.

    Args:
        net_file: Network file (.net.xml)
        dump_files: List of dump files (edgedata, lanedata)
        output_file: Output PNG file path
        measure: Measure to plot (speed, density, occupancy, etc.)
        colormap: Matplotlib colormap name
        title: Plot title
        min_value: Minimum value for color scale
        max_value: Maximum value for color scale

    Returns:
        True if successful, False otherwise
    """
    args = ['-n', net_file]

    for f in dump_files:
        args.extend(['--dump', f])

    args.extend(['-m', measure])
    args.extend(['--colormap', colormap])

    if title:
        args.extend(['--title', title])
    if min_value is not None:
        args.extend(['--min-value', str(min_value)])
    if max_value is not None:
        args.extend(['--max-value', str(max_value)])

    args.extend(['-o', output_file])

    rc, stdout, stderr = _run_sumo_tool('plot_net_dump.py', args)

    if rc != 0:
        print(f"plot_net_dump failed: {stderr}")
        return False
    return True


def plot_net_speeds(
    net_file: str,
    output_file: str,
    colormap: str = 'RdYlGn',
    title: Optional[str] = None,
) -> bool:
    """Run SUMO's plot_net_speeds.py tool.

    Plots the allowed speeds on the network edges.

    Args:
        net_file: Network file (.net.xml)
        output_file: Output PNG file path
        colormap: Matplotlib colormap name
        title: Plot title

    Returns:
        True if successful, False otherwise
    """
    args = ['-n', net_file]
    args.extend(['--colormap', colormap])

    if title:
        args.extend(['--title', title])

    args.extend(['-o', output_file])

    rc, stdout, stderr = _run_sumo_tool('plot_net_speeds.py', args)

    if rc != 0:
        print(f"plot_net_speeds failed: {stderr}")
        return False
    return True


def plot_trajectories(
    fcd_file: str,
    output_file: str,
    vehicle_ids: Optional[List[str]] = None,
    edges: Optional[List[str]] = None,
    color_by: str = 'speed',
    title: Optional[str] = None,
) -> bool:
    """Run SUMO's plot_trajectories.py tool.

    Creates trajectory plots (time vs position or x vs y) from FCD data.

    Args:
        fcd_file: FCD output file
        output_file: Output PNG file path
        vehicle_ids: Filter to specific vehicle IDs
        edges: Filter to specific edges
        color_by: Attribute to color by (speed, acceleration, etc.)
        title: Plot title

    Returns:
        True if successful, False otherwise
    """
    args = ['-t', fcd_file]

    if vehicle_ids:
        args.extend(['--filter-ids', ','.join(vehicle_ids)])
    if edges:
        args.extend(['--filter-edges', ','.join(edges)])
    if color_by:
        args.extend(['--color', color_by])
    if title:
        args.extend(['--title', title])

    args.extend(['-o', output_file])

    rc, stdout, stderr = _run_sumo_tool('plot_trajectories.py', args)

    if rc != 0:
        print(f"plot_trajectories failed: {stderr}")
        return False
    return True


def generate_all_sumo_plots(
    sim_dir_a: str,
    sim_dir_b: str,
    output_dir: str,
    labels: Tuple[str, str] = ('Baseline', 'Optimized'),
) -> List[str]:
    """Generate all available SUMO native plots for A/B comparison.

    Scans both simulation directories for available outputs and generates
    appropriate comparative visualizations.

    Args:
        sim_dir_a: Directory for simulation A (baseline)
        sim_dir_b: Directory for simulation B (optimized)
        output_dir: Directory to save plots
        labels: Labels for the two runs

    Returns:
        List of generated plot file paths
    """
    from .parsers import get_available_outputs

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    outputs_a = get_available_outputs(sim_dir_a)
    outputs_b = get_available_outputs(sim_dir_b)

    generated = []

    # Summary comparison
    if outputs_a['summary'] and outputs_b['summary']:
        out_file = str(Path(output_dir) / 'sumo_summary_comparison.png')
        success = plot_summary(
            [str(outputs_a['summary']), str(outputs_b['summary'])],
            out_file,
            measures=['running', 'halting'],
            title='Vehicle Count Comparison',
            labels=list(labels),
        )
        if success:
            generated.append(out_file)

    # Tripinfo attribute comparisons
    if outputs_a['tripinfo'] and outputs_b['tripinfo']:
        tripinfo_files = [str(outputs_a['tripinfo']), str(outputs_b['tripinfo'])]

        # Duration scatter
        out_file = str(Path(output_dir) / 'sumo_duration_scatter.png')
        success = plot_xml_attributes(
            tripinfo_files,
            'depart', 'duration',
            out_file,
            title='Travel Time by Departure Time',
            xlabel='Departure Time (s)',
            ylabel='Duration (s)',
            plot_type='scatter',
            labels=list(labels),
        )
        if success:
            generated.append(out_file)

        # Time loss scatter
        out_file = str(Path(output_dir) / 'sumo_timeloss_scatter.png')
        success = plot_xml_attributes(
            tripinfo_files,
            'depart', 'timeLoss',
            out_file,
            title='Time Loss by Departure Time',
            xlabel='Departure Time (s)',
            ylabel='Time Loss (s)',
            plot_type='scatter',
            labels=list(labels),
        )
        if success:
            generated.append(out_file)

        # Waiting time
        out_file = str(Path(output_dir) / 'sumo_waitingtime_scatter.png')
        success = plot_xml_attributes(
            tripinfo_files,
            'depart', 'waitingTime',
            out_file,
            title='Waiting Time by Departure Time',
            xlabel='Departure Time (s)',
            ylabel='Waiting Time (s)',
            plot_type='scatter',
            labels=list(labels),
        )
        if success:
            generated.append(out_file)

    # Network visualization (if edgedata available)
    net_file_a = outputs_a.get('network')
    if net_file_a and outputs_a.get('edgedata'):
        out_file = str(Path(output_dir) / 'sumo_net_speed_A.png')
        success = plot_net_dump(
            str(net_file_a),
            [str(outputs_a['edgedata'])],
            out_file,
            measure='speed',
            title=f'Network Speed - {labels[0]}',
        )
        if success:
            generated.append(out_file)

    net_file_b = outputs_b.get('network')
    if net_file_b and outputs_b.get('edgedata'):
        out_file = str(Path(output_dir) / 'sumo_net_speed_B.png')
        success = plot_net_dump(
            str(net_file_b),
            [str(outputs_b['edgedata'])],
            out_file,
            measure='speed',
            title=f'Network Speed - {labels[1]}',
        )
        if success:
            generated.append(out_file)

    return generated


def check_sumo_tools_available() -> bool:
    """Check if SUMO visualization tools are available.

    Returns:
        True if tools are available, False otherwise
    """
    try:
        tools_path = _get_sumo_tools_path()
        viz_path = tools_path / 'visualization'
        return viz_path.exists() and (viz_path / 'plotXMLAttributes.py').exists()
    except FileNotFoundError:
        return False
