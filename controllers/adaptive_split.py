"""Reparto de verde por equisaturación: proporcional a colas con piso."""
import traci
from utils.signal_utils import _green_phase_edges, apply_phase_durations


def compute_split(queues, budget, floor=0.2):
    """Reparte `budget` segundos de verde entre fases proporcional a `queues`.

    queues: dict fase_verde_idx -> vehículos en cola. Devuelve {} si no hay
    demanda o presupuesto inválido (el llamador no debe tocar el cruce).
    El piso garantiza servicio mínimo a toda fase con demanda registrada.
    """
    total = sum(queues.values())
    if not queues or total <= 0 or budget <= 0:
        return {}
    shares = {i: max(q / total, floor) for i, q in queues.items()}
    norm = sum(shares.values())
    return {i: budget * s / norm for i, s in shares.items()}


class AdaptiveSplitController:
    """Equisaturación local: reparte el verde por colas reales, una vez por ciclo."""

    def __init__(self, visible_count):
        self._visible_count = visible_count
        self._tls = {}   # tls_id -> {'phase_edges', 'budget', 'yellow_total', 'next_update'}

    def register(self, tls_id):
        defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
        if not defs:
            return
        phases = list(getattr(defs[0], 'phases', ()) or ())
        phase_edges = _green_phase_edges(tls_id, phases)
        if len(phase_edges) < 2:
            return  # sin reparto posible
        green_total = sum(phases[i].duration for i in phase_edges)
        cycle_total = sum(p.duration for p in phases)
        self._tls[tls_id] = {
            'phase_edges': phase_edges,
            'budget': float(green_total),
            'yellow_total': float(cycle_total - green_total),
            'next_update': 0.0,
        }

    def set_cycle_budget(self, tls_id, budget):
        if tls_id in self._tls and budget > 0:
            self._tls[tls_id]['budget'] = float(budget)

    def tick(self, current_time):
        applied = 0
        for tls_id, info in self._tls.items():
            if current_time < info['next_update']:
                continue
            queues = {
                i: sum(self._visible_count(e) for e in edges)
                for i, edges in info['phase_edges'].items()
            }
            split = compute_split(queues, info['budget'])
            if split and apply_phase_durations(tls_id, split):
                applied += 1
            info['next_update'] = current_time + info['budget'] + info['yellow_total']
        return applied
