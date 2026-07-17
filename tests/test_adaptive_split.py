from controllers.adaptive_split import compute_split
from tests.conftest import FakeLogic, FakePhase


def test_split_proportional():
    # colas 8 vs 2, presupuesto 84 → 80%/20% (el piso no muerde)
    assert compute_split({0: 8, 2: 2}, 84.0) == {0: 67.2, 2: 16.8}

def test_split_floor_protects_minority():
    out = compute_split({0: 100, 2: 1}, 84.0)
    assert out[2] >= 0.2 / 1.2 * 84.0 - 1e-9   # piso normalizado
    assert abs(sum(out.values()) - 84.0) < 1e-9

def test_split_noop_without_demand():
    assert compute_split({0: 0, 2: 0}, 84.0) == {}
    assert compute_split({}, 84.0) == {}
    assert compute_split({0: 5}, 0.0) == {}


def _setup_tl(fake_traci):
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


def test_controller_splits_by_queues_each_cycle(fake_traci, monkeypatch):
    from controllers.adaptive_split import AdaptiveSplitController

    monkeypatch.setenv('ADAPTIVE_SPLIT', 'queue')
    _setup_tl(fake_traci)
    queues = {'edge_norte': 8, 'edge_sur': 2}
    ctl = AdaptiveSplitController(visible_count=lambda e: queues.get(e, 0))
    ctl.register('tl1')

    assert ctl.tick(0.0) == 1
    logic = fake_traci.trafficlight.applied[-1][1]
    assert [round(p.duration, 1) for p in logic.phases] == [67.2, 3.0, 16.8, 3.0]

    assert ctl.tick(30.0) == 0            # dentro del mismo ciclo: no toca
    queues['edge_norte'], queues['edge_sur'] = 2, 8
    assert ctl.tick(95.0) == 1            # ciclo vencido: re-reparte
    logic = fake_traci.trafficlight.applied[-1][1]
    assert [round(p.duration, 1) for p in logic.phases] == [16.8, 3.0, 67.2, 3.0]


def test_controller_noop_without_demand(fake_traci, monkeypatch):
    from controllers.adaptive_split import AdaptiveSplitController

    monkeypatch.setenv('ADAPTIVE_SPLIT', 'queue')
    _setup_tl(fake_traci)
    ctl = AdaptiveSplitController(visible_count=lambda e: 0)
    ctl.register('tl1')
    assert ctl.tick(0.0) == 0
    assert fake_traci.trafficlight.applied == []


def test_controller_default_equal_split_ignores_queues(fake_traci):
    """Default 'equal': el presupuesto del pipeline se reparte en partes iguales."""
    from controllers.adaptive_split import AdaptiveSplitController

    _setup_tl(fake_traci)
    queues = {'edge_norte': 8, 'edge_sur': 2}
    ctl = AdaptiveSplitController(visible_count=lambda e: queues.get(e, 0))
    ctl.register('tl1')
    ctl.set_cycle_budget('tl1', 54.0)
    assert ctl.tick(0.0) == 1
    logic = fake_traci.trafficlight.applied[-1][1]
    assert [round(p.duration, 1) for p in logic.phases] == [27.0, 3.0, 27.0, 3.0]


def test_controller_cycle_budget_from_pipeline(fake_traci):
    from controllers.adaptive_split import AdaptiveSplitController

    _setup_tl(fake_traci)
    queues = {'edge_norte': 5, 'edge_sur': 5}
    ctl = AdaptiveSplitController(visible_count=lambda e: queues.get(e, 0))
    ctl.register('tl1')
    ctl.set_cycle_budget('tl1', 104.0)     # política del pipeline: ciclo útil mayor
    assert ctl.tick(0.0) == 1
    logic = fake_traci.trafficlight.applied[-1][1]
    assert [round(p.duration, 1) for p in logic.phases] == [52.0, 3.0, 52.0, 3.0]
