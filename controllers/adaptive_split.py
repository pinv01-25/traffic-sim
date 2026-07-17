"""Reparto de verde por equisaturación: proporcional a colas con piso."""


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
