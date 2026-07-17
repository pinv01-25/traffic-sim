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
    """Check if a phase state is a yellow transition (yellow+red only)."""
    return any(c in 'yY' for c in state) and all(c in 'yYrR' for c in state)


def _find_priority_green_phase(
    tls_id: str,
    phases: List,
    green_indices: List[int],
    priority_edge: str,
) -> Optional[int]:
    """Find the green phase whose green signals serve `priority_edge`.

    Uses getControlledLinks: signal index i is served by in-lane links[i][0][0];
    a phase serves the edge if any of its G/g signals enter from that edge.
    """
    try:
        links = traci.trafficlight.getControlledLinks(tls_id)
    except Exception:
        return None

    for i in green_indices:
        state = getattr(phases[i], 'state', '') or ''
        for sig_idx, ch in enumerate(state):
            if ch not in 'Gg' or sig_idx >= len(links) or not links[sig_idx]:
                continue
            in_lane = links[sig_idx][0][0]
            if in_lane.rsplit('_', 1)[0] == priority_edge:
                return i
    return None


#: Cota del reparto direccional: la aproximación prioritaria nunca recibe
#: menos del 35% ni más del 65% del presupuesto de verde. Servicio mínimo
#: para las transversales — sin cota, 70/20 gridlockea la red (medido).
PRIORITY_SHARE_MIN = 0.35
PRIORITY_SHARE_MAX = 0.65


def apply_durations_to_tls(
    tls_id: str,
    green_total: float,
    red_total: float,
    rows: Optional[List[dict]] = None,
    priority_edge: Optional[str] = None,
    preserve_cycle: bool = False,
) -> bool:
    """Apply total green/red durations to one traffic light.

    Preserves the existing program's states and number of phases. Yellow
    phases always keep their duration. Distribution:

    - `preserve_cycle=True` (modo dinámico): el optimizador piensa
      green+red como el ciclo del cruce. Si el programa no tiene fases
      todo-rojo, ese presupuesto completo se reparte entre las fases
      verdes (si solo se usara green_total, el ciclo se encogería y
      recomendaciones como 27/62 lo colapsarían a ~30s).
    - Con `priority_edge`: la fase verde que sirve ese edge recibe una
      proporción del presupuesto igual a green/(green+red), acotada a
      [PRIORITY_SHARE_MIN, PRIORITY_SHARE_MAX]; el resto se reparte entre
      las demás fases verdes.
    - Sin `priority_edge` (modo estático --green-time): `green_total`
      entre fases verdes y `red_total` entre fases todo-rojo, como siempre.

    If no green phase exists, phase 0 is treated as green.

    Args:
        tls_id: Traffic light ID
        green_total: Total green time to distribute (seconds)
        red_total: Total red time to distribute (seconds)
        rows: Optional list to append per-phase CSV rows to
        priority_edge: Edge whose approach should receive the larger share
        preserve_cycle: Treat green+red as the full cycle budget when the
            program has no dedicated all-red phases

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

        priority_idx = None
        if priority_edge and len(green_indices) > 1:
            priority_idx = _find_priority_green_phase(tls_id, phases, green_indices, priority_edge)

        # Presupuesto de verde: en modo dinámico sin fases todo-rojo, el
        # green+red del optimizador ES el ciclo útil del cruce.
        if preserve_cycle and not red_indices:
            green_budget = green_total + red_total
            red_per_phase = 0.0
        else:
            green_budget = green_total
            red_per_phase = red_total / len(red_indices) if red_indices else 0.0

        new_durations = []
        if priority_idx is not None:
            total = green_total + red_total
            share = green_total / total if total > 0 else 0.5
            share = max(PRIORITY_SHARE_MIN, min(share, PRIORITY_SHARE_MAX))
            other_greens = [i for i in green_indices if i != priority_idx]
            priority_dur = share * green_budget if other_greens else green_budget
            other_per_phase = (green_budget - priority_dur) / len(other_greens) if other_greens else 0.0
            for i, phase in enumerate(phases):
                if i == priority_idx:
                    new_durations.append(priority_dur)
                elif i in green_indices:
                    new_durations.append(other_per_phase)
                elif i in yellow_indices:
                    new_durations.append(float(phase.duration))
                else:
                    new_durations.append(red_per_phase)
        else:
            green_per_phase = green_budget / len(green_indices)
            for i, phase in enumerate(phases):
                if i in green_indices:
                    new_durations.append(green_per_phase)
                elif i in yellow_indices:
                    new_durations.append(float(phase.duration))
                else:
                    new_durations.append(red_per_phase)

        if rows is not None:
            for i, (phase, new_dur) in enumerate(zip(phases, new_durations, strict=False)):
                rows.append({
                    'tls_id': tls_id,
                    'phase_idx': i,
                    'state': getattr(phase, 'state', '') or '',
                    'assigned_duration': float(new_dur),
                    'original_duration': float(phase.duration),
                    'is_green': i in green_indices,
                    'is_yellow': i in yellow_indices,
                })

        # Idempotencia: re-aplicar un programa igual resetea la fase en curso
        # y perturba el cruce sin cambiar nada — mejor no tocar.
        if all(abs(float(p.duration) - d) < 0.5 for p, d in zip(phases, new_durations, strict=False)):
            return True

        # Posicional: Phase/Logic de libsumo no aceptan keyword args
        new_phases = tuple(
            traci.trafficlight.Phase(d, getattr(p, 'state', '') or '', d, d)
            for p, d in zip(phases, new_durations, strict=False)
        )

        # Mantener la fase en curso para no resetear el semáforo al aplicar
        try:
            current_phase = int(traci.trafficlight.getPhase(tls_id))
        except Exception:
            current_phase = 0

        new_logic = traci.trafficlight.Logic(
            getattr(logic, 'programID', '0'),
            getattr(logic, 'type', 0),
            current_phase,
            new_phases,
            getattr(logic, 'subParameter', {}) or {},
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
