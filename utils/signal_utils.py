"""Utilities to apply signal timing overrides via TraCI.

Provides a helper that inspects traffic light phase definitions to detect
which phases contain green indications and distributes requested green
time among them, with the remainder assigned to non-green phases.

This module expects a TraCI connection to already be established.
"""
from typing import List, Optional, Tuple
import traci
import os
import csv
from pathlib import Path


def _get_phases_from_definition(tls_id: str) -> List[Tuple[str, float]]:
    """Get phases (state, duration) from traffic light definition.

    Returns list of (state_string, duration) tuples.
    """
    phases = []
    try:
        defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
        if defs and len(defs) > 0:
            logic = defs[0]
            if hasattr(logic, 'phases'):
                for phase in logic.phases:
                    state = phase.state if hasattr(phase, 'state') else ''
                    duration = phase.duration if hasattr(phase, 'duration') else 0.0
                    phases.append((state, duration))
    except Exception as e:
        print(f"Warning: Could not get phases for {tls_id}: {e}")
    return phases


def _has_green(state: str) -> bool:
    """Check if a phase state contains green signals."""
    return 'G' in state or 'g' in state


def _is_yellow(state: str) -> bool:
    """Check if a phase state is yellow only."""
    return all(c in 'yY' for c in state if c not in 'rR')


def apply_timings_to_all_tls(green_time: float, cycle_time: float, out_csv: Optional[str] = None):
    """Apply `green_time` and `cycle_time` to all traffic lights connected.

    The function detects which phases include green indicators ('G' or 'g')
    and distributes `green_time` among those phases; remaining time is split
    among the other phases (red/yellow).

    Uses setCompleteRedYellowGreenDefinition to update the entire program.
    """
    gt = float(green_time)
    ct = float(cycle_time)
    if ct <= 0:
        raise ValueError('cycle_time must be > 0')
    if gt < 0:
        raise ValueError('green_time must be >= 0')

    red_time = max(ct - gt, 0.0)
    tls_list = traci.trafficlight.getIDList()

    rows = []

    for tls in tls_list:
        try:
            # Get current definition
            defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)
            if not defs or len(defs) == 0:
                continue

            logic = defs[0]
            if not hasattr(logic, 'phases') or len(logic.phases) == 0:
                continue

            phases = list(logic.phases)

            # Identify green phases vs non-green phases
            green_indices = []
            yellow_indices = []
            red_indices = []

            for i, phase in enumerate(phases):
                state = phase.state if hasattr(phase, 'state') else ''
                if _has_green(state):
                    green_indices.append(i)
                elif _is_yellow(state):
                    yellow_indices.append(i)
                else:
                    red_indices.append(i)

            # If no green phases detected, treat first phase as green
            if not green_indices:
                green_indices = [0]
                red_indices = list(range(1, len(phases)))

            # Calculate durations
            # Green time divided among green phases
            green_per_phase = gt / len(green_indices) if green_indices else 0

            # Yellow phases keep original duration (typically 3s)
            # Red phases get remaining time after green and yellow
            yellow_time = sum(phases[i].duration for i in yellow_indices)
            red_total = max(ct - gt - yellow_time, 0.0)
            red_per_phase = red_total / len(red_indices) if red_indices else 0

            # Build new phases with updated durations
            new_phases = []
            for i, phase in enumerate(phases):
                state = phase.state if hasattr(phase, 'state') else ''

                if i in green_indices:
                    new_dur = green_per_phase
                elif i in yellow_indices:
                    new_dur = phase.duration  # Keep yellow as-is
                else:
                    new_dur = red_per_phase

                # Create new Phase object
                new_phase = traci.trafficlight.Phase(
                    duration=new_dur,
                    state=state,
                    minDur=new_dur,
                    maxDur=new_dur
                )
                new_phases.append(new_phase)

                # Record for CSV
                rows.append({
                    'tls_id': tls,
                    'phase_idx': i,
                    'state': state,
                    'assigned_duration': float(new_dur),
                    'original_duration': float(phase.duration),
                    'is_green': i in green_indices,
                    'is_yellow': i in yellow_indices,
                })

            # Create new Logic and apply it
            new_logic = traci.trafficlight.Logic(
                programID=logic.programID if hasattr(logic, 'programID') else '0',
                type=logic.type if hasattr(logic, 'type') else 0,
                currentPhaseIndex=0,
                phases=tuple(new_phases),
                subParameter=logic.subParameter if hasattr(logic, 'subParameter') else {}
            )

            traci.trafficlight.setCompleteRedYellowGreenDefinition(tls, new_logic)

        except Exception as e:
            print(f"Warning: Could not apply timing to {tls}: {e}")
            continue

    # Write CSV if requested
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
                for r in rows:
                    writer.writerow(r)
        except Exception as e:
            print(f"Warning: Could not write CSV: {e}")

    return len(rows)
