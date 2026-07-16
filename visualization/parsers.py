"""Parsers for common SUMO XML outputs used by visualization tools.

Provides lightweight functions that return pandas DataFrames for plotting.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def parse_tripinfo(path: str) -> pd.DataFrame:
    """Parse a SUMO tripinfo.xml file into a DataFrame.

    Columns: id, depart, arrival, duration, routeLength, departDelay, timeLoss, waitingTime, departLane
    """
    rows: List[Dict] = []
    for _event, elem in ET.iterparse(path):
        if elem.tag == 'tripinfo':
            try:
                rows.append({
                    'id': elem.get('id'),
                    'depart': float(elem.get('depart', 0.0)),
                    'arrival': float(elem.get('arrival', 0.0)),
                    'duration': float(elem.get('duration', 0.0)),
                    'routeLength': float(elem.get('routeLength', 0.0)),
                    'departDelay': float(elem.get('departDelay', 0.0)),
                    'timeLoss': float(elem.get('timeLoss', 0.0)),
                    'waitingTime': float(elem.get('waitingTime', 0.0) or 0.0),
                    'departLane': elem.get('departLane') or ''
                })
            except Exception:
                # skip malformed entries
                pass
            elem.clear()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def parse_summary(path: str) -> pd.DataFrame:
    """Parse a SUMO summary.xml file into a DataFrame.

    SUMO writes summary output as <summary><step time="..." running="..."
    halting="..."/></summary>; older/alternative files may use <interval>.
    """
    rows: List[Dict] = []
    for _event, elem in ET.iterparse(path):
        if elem.tag in ('step', 'interval'):
            rows.append({k: _try_cast(v) for k, v in elem.items()})
            elem.clear()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _try_cast(s: str):
    if s is None:
        return s
    if _is_float(s):
        return float(s)
    return s


def parse_fcd(path: str, sample_rate: int = 1) -> pd.DataFrame:
    """Parse a SUMO fcd-output.xml file into a DataFrame.

    FCD (Floating Car Data) contains position and speed of every vehicle
    at each timestep.

    Args:
        path: Path to fcd.xml file
        sample_rate: Only keep every Nth timestep (1 = all, 10 = every 10th)

    Returns:
        DataFrame with columns: time, id, x, y, speed, angle, edge, lane, pos, slope
    """
    rows: List[Dict] = []
    timestep_count = 0

    for _event, elem in ET.iterparse(path, events=['end']):
        if elem.tag == 'timestep':
            timestep_count += 1
            if timestep_count % sample_rate != 0:
                elem.clear()
                continue

            time = float(elem.get('time', 0.0))
            for vehicle in elem.findall('vehicle'):
                try:
                    rows.append({
                        'time': time,
                        'id': vehicle.get('id'),
                        'x': float(vehicle.get('x', 0.0)),
                        'y': float(vehicle.get('y', 0.0)),
                        'speed': float(vehicle.get('speed', 0.0)),
                        'angle': float(vehicle.get('angle', 0.0)),
                        'edge': vehicle.get('lane', '').rsplit('_', 1)[0] if vehicle.get('lane') else '',
                        'lane': vehicle.get('lane', ''),
                        'pos': float(vehicle.get('pos', 0.0)),
                        'slope': float(vehicle.get('slope', 0.0)) if vehicle.get('slope') else 0.0,
                    })
                except (ValueError, TypeError):
                    continue
            elem.clear()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def get_available_outputs(sim_dir: str) -> Dict[str, Optional[Path]]:
    """Scan a simulation directory for available SUMO output files.

    Returns dict mapping output type to path (or None if not found).
    """
    sim_path = Path(sim_dir)

    # Common locations
    search_paths = [
        sim_path / 'logs' / 'sumo_output',
        sim_path,
    ]

    outputs = {
        'tripinfo': None,
        'summary': None,
        'fcd': None,
        'queue': None,
        'edgedata': None,
        'lanedata': None,
        'detector': None,
        'emission': None,
        'network': None,
    }

    file_patterns = {
        'tripinfo': ['tripinfo.xml', '*tripinfo*.xml'],
        'summary': ['summary.xml', '*summary*.xml'],
        'fcd': ['fcd.xml', '*fcd*.xml', 'fcd-output.xml'],
        'queue': ['queue.xml', '*queue*.xml'],
        'edgedata': ['edgedata.xml', '*edgedata*.xml', 'edge_data.xml'],
        'lanedata': ['lanedata.xml', '*lanedata*.xml'],
        'detector': ['detector.xml', '*detector*.xml', 'e1_*.xml', 'e2_*.xml'],
        'emission': ['emission.xml', '*emission*.xml'],
        'network': ['network.net.xml', '*.net.xml'],
    }

    for search_path in search_paths:
        if not search_path.exists():
            continue
        for output_type, patterns in file_patterns.items():
            if outputs[output_type] is not None:
                continue
            for pattern in patterns:
                matches = list(search_path.glob(pattern))
                if matches:
                    outputs[output_type] = matches[0]
                    break

    return outputs


def compute_trip_statistics(df: pd.DataFrame) -> Dict:
    """Compute comprehensive statistics from tripinfo DataFrame.

    Returns dict with all computed metrics.
    """
    if df.empty:
        return {}

    stats = {}

    numeric_cols = ['duration', 'routeLength', 'departDelay', 'timeLoss', 'waitingTime']

    for col in numeric_cols:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        if len(data) == 0:
            continue

        stats[col] = {
            'count': len(data),
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'q95': float(data.quantile(0.95)),
        }

    # Derived metrics
    if 'duration' in df.columns and 'routeLength' in df.columns:
        valid = (df['duration'] > 0) & (df['routeLength'] > 0)
        if valid.sum() > 0:
            speeds = df.loc[valid, 'routeLength'] / df.loc[valid, 'duration']
            stats['avg_speed_ms'] = {
                'mean': float(speeds.mean()),
                'median': float(speeds.median()),
                'std': float(speeds.std()),
            }
            stats['avg_speed_kmh'] = {
                'mean': float(speeds.mean() * 3.6),
                'median': float(speeds.median() * 3.6),
            }

    # Time efficiency: (duration - timeLoss) / duration
    if 'duration' in df.columns and 'timeLoss' in df.columns:
        valid = df['duration'] > 0
        if valid.sum() > 0:
            efficiency = (df.loc[valid, 'duration'] - df.loc[valid, 'timeLoss']) / df.loc[valid, 'duration']
            stats['time_efficiency'] = {
                'mean': float(efficiency.mean()),
                'median': float(efficiency.median()),
            }

    return stats


def compute_summary_statistics(df: pd.DataFrame) -> Dict:
    """Compute statistics from summary DataFrame.

    Returns dict with aggregated metrics over the simulation.
    """
    if df.empty:
        return {}

    stats = {}

    # Find time column
    time_col = None
    for candidate in ('begin', 'time', 't'):
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col:
        stats['simulation_time'] = {
            'start': float(df[time_col].min()),
            'end': float(df[time_col].max()),
            'duration': float(df[time_col].max() - df[time_col].min()),
        }

    metric_cols = ['running', 'halting', 'entered', 'left', 'teleported',
                   'waiting', 'loaded', 'inserted', 'collisions']

    for col in metric_cols:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        if len(data) == 0:
            continue

        stats[col] = {
            'mean': float(data.mean()),
            'max': float(data.max()),
            'total': float(data.sum()) if col in ['entered', 'left', 'teleported', 'collisions'] else None,
        }

    # Peak congestion: max halting / max running ratio
    if 'halting' in df.columns and 'running' in df.columns:
        running = df['running'].replace(0, np.nan)
        congestion_ratio = df['halting'] / running
        stats['congestion_ratio'] = {
            'mean': float(congestion_ratio.mean()),
            'max': float(congestion_ratio.max()),
        }

    return stats
