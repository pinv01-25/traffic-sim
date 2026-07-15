"""Utilities to apply signal timing overrides via TraCI.

`apply_durations_to_tls` is the single primitive that rewrites phase
durations for one traffic light while preserving its phase structure:
green time is split among phases containing green indications, red time
among the remaining non-yellow phases, and yellow phases keep their
original duration.

This module expects a TraCI connection to already be established.
"""
import csv
from pathlib import Path
from typing import List, Optional

import traci


def _has_green(state: str) -> bool:
    """Check if a phase state contains green signals."""
    return 'G' in state or 'g' in state


def _is_yellow(state: str) -> bool:
    """Check if a phase state is yellow only."""
    return all(c in 'yY' for c in state if c not in 'rR')


def apply_durations_to_tls(
    tls_id: str,
    green_total: float,
    red_total: float,
    rows: Optional[List[dict]] = None,
) -> bool:
    """Apply total green/red durations to one traffic light.

    Preserves the existing program's states and number of phases:
    `green_total` is split evenly among green phases, `red_total` among
    non-green non-yellow phases, and yellow phases keep their duration.
    If no green phase exists, phase 0 is treated as green.

    Args:
        tls_id: Traffic light ID
        green_total: Total green time to distribute (seconds)
        red_total: Total red time to distribute (seconds)
        rows: Optional list to append per-phase CSV rows to

    Returns:
        True if a new program was applied.
    """
    try:
        defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
        if not defs:
            return False

        logic = defs[0]
        phases = list(getattr(logic, 'phases', ()) or ())
        if not phases:
            return False

        green_indices = []
        yellow_indices = []
        red_indices = []
        for i, phase in enumerate(phases):
            state = getattr(phase, 'state', '') or ''
            if _has_green(state):
                green_indices.append(i)
            elif _is_yellow(state):
                yellow_indices.append(i)
            else:
                red_indices.append(i)

        if not green_indices:
            green_indices = [0]
            red_indices = [i for i in range(1, len(phases)) if i not in yellow_indices]

        green_per_phase = green_total / len(green_indices)
        red_per_phase = red_total / len(red_indices) if red_indices else 0.0

        new_phases = []
        for i, phase in enumerate(phases):
            state = getattr(phase, 'state', '') or ''
            if i in green_indices:
                new_dur = green_per_phase
            elif i in yellow_indices:
                new_dur = phase.duration
            else:
                new_dur = red_per_phase

            new_phases.append(traci.trafficlight.Phase(
                duration=new_dur,
                state=state,
                minDur=new_dur,
                maxDur=new_dur,
            ))

            if rows is not None:
                rows.append({
                    'tls_id': tls_id,
                    'phase_idx': i,
                    'state': state,
                    'assigned_duration': float(new_dur),
                    'original_duration': float(phase.duration),
                    'is_green': i in green_indices,
                    'is_yellow': i in yellow_indices,
                })

        new_logic = traci.trafficlight.Logic(
            programID=getattr(logic, 'programID', '0'),
            type=getattr(logic, 'type', 0),
            currentPhaseIndex=0,
            phases=tuple(new_phases),
            subParameter=getattr(logic, 'subParameter', {}) or {},
        )
        traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, new_logic)
        return True

    except Exception as e:
        print(f"Warning: Could not apply timing to {tls_id}: {e}")
        return False


def apply_timings_to_all_tls(green_time: float, cycle_time: float, out_csv: Optional[str] = None):
    """Apply `green_time` within `cycle_time` to all connected traffic lights.

    Per traffic light: yellow phases keep their duration and the red total
    is `cycle_time - green_time - yellow_total`.

    Returns the number of phase rows processed (also written to `out_csv`).
    """
    gt = float(green_time)
    ct = float(cycle_time)
    if ct <= 0:
        raise ValueError('cycle_time must be > 0')
    if gt < 0:
        raise ValueError('green_time must be >= 0')

    rows: List[dict] = []
    for tls in traci.trafficlight.getIDList():
        try:
            defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)
            if not defs:
                continue
            phases = list(getattr(defs[0], 'phases', ()) or ())
            yellow_total = sum(
                p.duration for p in phases if _is_yellow(getattr(p, 'state', '') or '')
                and not _has_green(getattr(p, 'state', '') or '')
            )
            red_total = max(ct - gt - yellow_total, 0.0)
            apply_durations_to_tls(tls, gt, red_total, rows=rows)
        except Exception as e:
            print(f"Warning: Could not apply timing to {tls}: {e}")
            continue

    if out_csv and rows:
        try:
            p = Path(out_csv)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w', newline='') as cf:
                writer = csv.DictWriter(cf, fieldnames=[
                    'tls_id', 'phase_idx', 'state', 'assigned_duration',
                    'original_duration', 'is_green', 'is_yellow'
                ])
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            print(f"Warning: Could not write CSV: {e}")

    return len(rows)
