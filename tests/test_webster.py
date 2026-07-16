"""Tests del cálculo de temporización Webster."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_webster_timing import (  # noqa: E402
    AMBER_DURATION,
    MAX_CYCLE,
    MIN_CYCLE,
    compute_webster_timing,
)

PHASES = [
    {'duration': 30.0, 'state': 'GGrr'},
    {'duration': 3.0, 'state': 'yyrr'},
    {'duration': 30.0, 'state': 'rrGG'},
    {'duration': 3.0, 'state': 'rryy'},
]
LINK_TO_EDGE = {0: 'edge_ns', 1: 'edge_ns', 2: 'edge_ew', 3: 'edge_ew'}


def test_webster_asymmetric_demand():
    edge_demand = {'edge_ns': 3000, 'edge_ew': 600}
    cycle, out = compute_webster_timing('tl1', PHASES, LINK_TO_EDGE, edge_demand)

    assert cycle is not None
    assert MIN_CYCLE <= cycle <= MAX_CYCLE
    # Estructura preservada
    assert [p['state'] for p in out] == [p['state'] for p in PHASES]
    # Ámbar normalizado a 3s
    assert out[1]['duration'] == AMBER_DURATION
    assert out[3]['duration'] == AMBER_DURATION
    # Más verde para la fase de mayor demanda
    assert out[0]['duration'] > out[2]['duration']


def test_webster_no_demand_keeps_original():
    cycle, out = compute_webster_timing('tl1', PHASES, LINK_TO_EDGE, {})
    assert cycle is None
    assert out == PHASES
