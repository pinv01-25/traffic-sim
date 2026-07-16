"""Tests de la primitiva de aplicación de tiempos a semáforos."""
from tests.conftest import FakeLogic, FakePhase


def _program_4_phases():
    return FakeLogic(phases=(
        FakePhase(duration=30.0, state='GGrr'),
        FakePhase(duration=3.0, state='yyrr'),
        FakePhase(duration=30.0, state='rrGG'),
        FakePhase(duration=3.0, state='rryy'),
    ))


def test_apply_durations_preserves_states_and_splits_green(fake_traci):
    from utils.signal_utils import apply_durations_to_tls

    fake_traci.trafficlight.programs['tl1'] = _program_4_phases()

    ok = apply_durations_to_tls('tl1', green_total=40.0, red_total=20.0)
    assert ok

    _, logic = fake_traci.trafficlight.applied[-1]
    states = [p.state for p in logic.phases]
    durations = [p.duration for p in logic.phases]

    # Estados y número de fases intactos
    assert states == ['GGrr', 'yyrr', 'rrGG', 'rryy']
    # Ambas fases son "verdes" (contienen G): 40/2 cada una; amarillo intacto.
    # En este programa no hay fases solo-rojas, así que red_total no se reparte.
    assert durations == [20.0, 3.0, 20.0, 3.0]
    # minDur/maxDur fijados a la duración
    assert all(p.minDur == p.duration and p.maxDur == p.duration for p in logic.phases)


def test_apply_durations_distributes_red_to_red_phases(fake_traci):
    from utils.signal_utils import apply_durations_to_tls

    fake_traci.trafficlight.programs['tl1'] = FakeLogic(phases=(
        FakePhase(duration=30.0, state='GG'),
        FakePhase(duration=3.0, state='yy'),
        FakePhase(duration=30.0, state='rr'),
    ))

    assert apply_durations_to_tls('tl1', green_total=25.0, red_total=15.0)
    _, logic = fake_traci.trafficlight.applied[-1]
    assert [p.duration for p in logic.phases] == [25.0, 3.0, 15.0]
    assert [p.state for p in logic.phases] == ['GG', 'yy', 'rr']


def test_wrapper_matches_previous_behavior(fake_traci):
    """apply_timings_to_all_tls(green, cycle): rojo = ciclo - verde - amarillo."""
    from utils.signal_utils import apply_timings_to_all_tls

    fake_traci.trafficlight.programs['tl1'] = FakeLogic(phases=(
        FakePhase(duration=30.0, state='GG'),
        FakePhase(duration=3.0, state='yy'),
        FakePhase(duration=30.0, state='rr'),
    ))

    n_rows = apply_timings_to_all_tls(30.0, 60.0)
    assert n_rows == 3

    _, logic = fake_traci.trafficlight.applied[-1]
    # verde=30, amarillo=3 intacto, rojo = 60-30-3 = 27
    assert [p.duration for p in logic.phases] == [30.0, 3.0, 27.0]


def test_apply_is_idempotent_and_preserves_current_phase(fake_traci):
    """Re-aplicar el mismo programa no debe resetear el semáforo."""
    from utils.signal_utils import apply_durations_to_tls

    fake_traci.trafficlight.programs['tl1'] = FakeLogic(phases=(
        FakePhase(duration=30.0, state='GG'),
        FakePhase(duration=3.0, state='yy'),
        FakePhase(duration=30.0, state='rr'),
    ))
    fake_traci.trafficlight.phase['tl1'] = 2  # cruce a mitad de ciclo

    assert apply_durations_to_tls('tl1', green_total=25.0, red_total=15.0)
    assert len(fake_traci.trafficlight.applied) == 1
    # Mantiene la fase en curso (no salta a la 0)
    assert fake_traci.trafficlight.applied[-1][1].currentPhaseIndex == 2

    # Segunda aplicación idéntica: no hay set (no perturba el cruce)
    assert apply_durations_to_tls('tl1', green_total=25.0, red_total=15.0)
    assert len(fake_traci.trafficlight.applied) == 1


def test_apply_directional_split_prioritizes_congested_edge(fake_traci):
    """Con priority_edge, el verde largo va a la fase que sirve ese edge."""
    from utils.signal_utils import apply_durations_to_tls

    fake_traci.trafficlight.programs['tl1'] = FakeLogic(phases=(
        FakePhase(duration=42.0, state='GGrr'),
        FakePhase(duration=3.0, state='yyrr'),
        FakePhase(duration=42.0, state='rrGG'),
        FakePhase(duration=3.0, state='rryy'),
    ))
    # Señales 0-1 entran desde edge_norte; 2-3 desde edge_sur
    fake_traci.trafficlight.links['tl1'] = [
        [('edge_norte_0', 'out_0', '')], [('edge_norte_1', 'out_1', '')],
        [('edge_sur_0', 'out_2', '')], [('edge_sur_1', 'out_3', '')],
    ]

    assert apply_durations_to_tls('tl1', 70.0, 20.0, priority_edge='edge_sur')
    durations = [p.duration for p in fake_traci.trafficlight.applied[-1][1].phases]
    # Fase 2 ('rrGG') sirve edge_sur → 70s; fase 0 recibe el rojo (20s); amarillas intactas
    assert durations == [20.0, 3.0, 70.0, 3.0]

    # Sin priority_edge conocido cae al reparto igualitario
    assert apply_durations_to_tls('tl1', 70.0, 20.0, priority_edge='edge_inexistente')
    durations = [p.duration for p in fake_traci.trafficlight.applied[-1][1].phases]
    assert durations == [35.0, 3.0, 35.0, 3.0]
