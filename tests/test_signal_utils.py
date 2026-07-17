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


def test_apply_directional_split_bounded_and_cycle_preserving(fake_traci):
    """Modo dinámico: presupuesto = verde+rojo, prioridad acotada a 65%."""
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

    # 70/20 → presupuesto 90; share pedido 78% se acota a 65% → 58.5/31.5.
    # Fase 2 ('rrGG') sirve edge_sur → recibe la parte mayor; amarillas intactas.
    assert apply_durations_to_tls('tl1', 70.0, 20.0, priority_edge='edge_sur',
                                  preserve_cycle=True)
    durations = [p.duration for p in fake_traci.trafficlight.applied[-1][1].phases]
    assert durations == [31.5, 3.0, 58.5, 3.0]

    # Sin priority_edge conocido: presupuesto completo repartido igual (45/45,
    # ciclo preservado — NO 35/35 que encogería el ciclo)
    assert apply_durations_to_tls('tl1', 70.0, 20.0, priority_edge='edge_inexistente',
                                  preserve_cycle=True)
    durations = [p.duration for p in fake_traci.trafficlight.applied[-1][1].phases]
    assert durations == [45.0, 3.0, 45.0, 3.0]


def test_apply_cycle_preserved_for_light_traffic_recommendation(fake_traci):
    """27/62 (tráfico ligero) no debe colapsar el ciclo a ~30s."""
    from utils.signal_utils import apply_durations_to_tls

    fake_traci.trafficlight.programs['tl1'] = FakeLogic(phases=(
        FakePhase(duration=42.0, state='GGrr'),
        FakePhase(duration=3.0, state='yyrr'),
        FakePhase(duration=42.0, state='rrGG'),
        FakePhase(duration=3.0, state='rryy'),
    ))
    fake_traci.trafficlight.links['tl1'] = [
        [('edge_norte_0', 'o', '')], [('edge_norte_1', 'o', '')],
        [('edge_sur_0', 'o', '')], [('edge_sur_1', 'o', '')],
    ]

    # share pedido 27/89 = 30% se acota a 35% → 31.15/57.85 sobre presupuesto 89
    assert apply_durations_to_tls('tl1', 27.0, 62.0, priority_edge='edge_norte',
                                  preserve_cycle=True)
    durations = [p.duration for p in fake_traci.trafficlight.applied[-1][1].phases]
    cycle = sum(durations)
    assert abs(cycle - (89.0 + 6.0)) < 0.01  # ciclo ≈ presupuesto + amarillos
    assert durations[0] == round(0.35 * 89.0, 10)  # prioridad acotada abajo
