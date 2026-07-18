from tests.conftest import FakeLogic, FakePhase


def _setup_2phase(fake_traci, green_duration=30.0):
    fake_traci.trafficlight.programs['tl1'] = FakeLogic(phases=(
        FakePhase(duration=green_duration, state='GGrr'),
        FakePhase(duration=3.0, state='yyrr'),
        FakePhase(duration=green_duration, state='rrGG'),
        FakePhase(duration=3.0, state='rryy'),
    ))
    fake_traci.trafficlight.links['tl1'] = [
        [('in_n0', 'out_n0', '')], [('in_n1', 'out_n1', '')],
        [('in_s0', 'out_s0', '')], [('in_s1', 'out_s1', '')],
    ]


def test_lane_queue_ratio_normalizes_by_capacity(fake_traci):
    from controllers.max_pressure import lane_queue_ratio

    fake_traci.lane.lengths['short'] = 7.5   # capacidad 1
    fake_traci.lane.halting['short'] = 1
    fake_traci.lane.lengths['long'] = 75.0   # capacidad 10
    fake_traci.lane.halting['long'] = 1

    assert lane_queue_ratio('short') == 1.0
    assert lane_queue_ratio('long') == 0.1


def test_derive_yellow_state_turns_green_to_yellow_rest_to_red():
    from utils.signal_utils import derive_yellow_state

    assert derive_yellow_state('GGrr') == 'yyrr'
    assert derive_yellow_state('rGgr') == 'ryyr'


def test_phases_conflict_ring_adjacency():
    from utils.signal_utils import phases_conflict

    two_phases = (FakePhase(30, 'GGrr'), FakePhase(3, 'yyrr'),
                  FakePhase(30, 'rrGG'), FakePhase(3, 'rryy'))
    # ≤2 fases verdes: nunca hay conflicto (nada más a lo que saltar)
    assert phases_conflict(two_phases, 0, 0) is False
    assert phases_conflict(two_phases, 0, 2) is False

    three_phases = (
        FakePhase(20, 'Grrrrr'), FakePhase(3, 'yrrrrr'),
        FakePhase(20, 'rrGrrr'), FakePhase(3, 'rryrrr'),
        FakePhase(20, 'rrrrGr'), FakePhase(3, 'rrrryr'),
    )
    # anillo de verdes: 0, 2, 4 (cíclico) -> adyacentes entre sí
    assert phases_conflict(three_phases, 0, 2) is False
    assert phases_conflict(three_phases, 2, 4) is False
    assert phases_conflict(three_phases, 4, 0) is False  # adyacencia cíclica


def test_register_skips_single_green_phase_tls(fake_traci):
    from controllers.max_pressure import MaxPressureController

    fake_traci.trafficlight.programs['tl1'] = FakeLogic(phases=(
        FakePhase(duration=30.0, state='GGGG'),
    ))
    fake_traci.trafficlight.links['tl1'] = [
        [('in_a', 'out_a', '')], [('in_b', 'out_b', '')],
        [('in_c', 'out_c', '')], [('in_d', 'out_d', '')],
    ]
    ctl = MaxPressureController()
    ctl.register('tl1')
    assert ctl.tick(0.0) == 0
    assert fake_traci.trafficlight.states == {}


def test_no_switch_before_min_green(fake_traci):
    from controllers.max_pressure import MaxPressureController

    _setup_2phase(fake_traci)
    fake_traci.lane.halting['in_s0'] = 5
    fake_traci.lane.halting['in_s1'] = 5  # fase 2 muy presionada

    ctl = MaxPressureController()
    ctl.register('tl1')
    assert ctl.tick(0.0) == 0
    assert fake_traci.trafficlight.states == {}


def test_switch_sequence_yellow_then_all_red_then_target_green(fake_traci):
    from controllers.max_pressure import MaxPressureController
    from config import MAX_PRESSURE_CONFIG

    _setup_2phase(fake_traci)
    # Longitud realista (capacidad 20) para que la cola cargada no dispare
    # por sí sola el modo congestionado (ratio 0.25 < umbral 0.5) y esta
    # secuencia mida solo min_green/histéresis "normales".
    for lane in ('in_n0', 'in_n1', 'in_s0', 'in_s1'):
        fake_traci.lane.lengths[lane] = 150.0
    fake_traci.lane.halting['in_s0'] = 5
    fake_traci.lane.halting['in_s1'] = 5

    ctl = MaxPressureController()
    ctl.register('tl1')

    # Derive timings from config: base_green=30, min_green_floor=0.75
    base_green = 30.0
    min_green = max(base_green * MAX_PRESSURE_CONFIG["min_green_floor"], 5.0)
    check_interval = MAX_PRESSURE_CONFIG["check_interval"]
    yellow_time = 3.0  # from TRAFFIC_LIGHT_CONFIG
    all_red_time = MAX_PRESSURE_CONFIG["all_red_time"]

    # First check at t=0, no min_green barrier yet
    assert ctl.tick(0.0) == 0
    # Check at t=check_interval, still less than min_green
    assert ctl.tick(check_interval) == 0
    # Check at t=2*check_interval, still less than min_green
    assert ctl.tick(2 * check_interval) == 0
    # Next check after min_green is reached: ceiling(min_green/check_interval)*check_interval
    next_check_time = ((min_green / check_interval) + 1) * check_interval
    assert ctl.tick(next_check_time) == 1   # elapsed >= min_green: switch decidido
    assert fake_traci.trafficlight.states['tl1'][-1] == 'yyrr'

    # Check during yellow phase
    assert ctl.tick(next_check_time + 1.0) == 0  # sigue en amarillo
    # Yellow ends at yellow_time after switch
    assert ctl.tick(next_check_time + yellow_time + 0.1) == 1  # amarillo vencido: pasa a all-red
    assert fake_traci.trafficlight.states['tl1'][-1] == 'rrrr'

    # Check during all-red phase
    assert ctl.tick(next_check_time + yellow_time + 0.5) == 0  # sigue en all-red
    # All-red ends at all_red_time after yellow ends
    all_red_end_time = next_check_time + yellow_time + all_red_time + 0.1
    assert ctl.tick(all_red_end_time) == 1  # all-red vencido: aplica verde destino
    assert fake_traci.trafficlight.states['tl1'][-1] == 'rrGG'


def test_hysteresis_blocks_switch_on_small_pressure_gap(fake_traci):
    from controllers.max_pressure import MaxPressureController

    _setup_2phase(fake_traci)
    for lane in ('in_n0', 'in_n1', 'in_s0', 'in_s1'):
        fake_traci.lane.lengths[lane] = 75.0  # capacidad 10

    fake_traci.lane.halting['in_n0'] = 5
    fake_traci.lane.halting['in_n1'] = 5   # fase 0 (actual): presión 1.0
    fake_traci.lane.halting['in_s0'] = 4
    fake_traci.lane.halting['in_s1'] = 5   # fase 2: presión 0.9 + bonus starvation < 1.15

    ctl = MaxPressureController()
    ctl.register('tl1')
    for t in (0.0, 5.0, 10.0):
        ctl.tick(t)
    assert fake_traci.trafficlight.states == {}  # margen insuficiente: no cambia


def test_starvation_bonus_eventually_forces_service(fake_traci):
    from controllers.max_pressure import MaxPressureController

    _setup_2phase(fake_traci)
    fake_traci.lane.lengths['in_n0'] = 75.0  # capacidad 10
    fake_traci.lane.halting['in_n0'] = 1     # fase 0 (actual): presión fija 0.1
    # fase 2 sin demanda propia: solo gana por el bonus de starvation

    ctl = MaxPressureController()
    ctl.register('tl1')

    t = 0.0
    while t < 60.0 and fake_traci.trafficlight.states == {}:
        ctl.tick(t)
        t += 5.0

    assert fake_traci.trafficlight.states.get('tl1', [None])[0] == 'yyrr'


def test_hard_green_max_forces_switch_despite_lower_pressure(fake_traci):
    from controllers.max_pressure import MaxPressureController

    _setup_2phase(fake_traci)  # base_green=30 -> max_green = 30*1.5 = 45
    fake_traci.lane.halting['in_n0'] = 5
    fake_traci.lane.halting['in_n1'] = 5  # fase 0 (actual) siempre gana en presión

    ctl = MaxPressureController()
    ctl.register('tl1')
    for t in (0.0, 15.0, 30.0, 40.0):
        assert ctl.tick(t) == 0  # sigue por debajo del verde máximo

    assert ctl.tick(45.0) == 1  # 45 >= max_green: fuerza corte pese a menor presión
    assert fake_traci.trafficlight.states['tl1'][-1] == 'yyrr'


def test_stall_detector_excludes_phase_whose_queue_never_drains(fake_traci):
    """Si una fase recibe verde repetidas veces y su cola de entrada no
    baja, el bloqueo está aguas abajo (no es un problema de reparto local):
    tras `stall_limit` servicios sin mejora se la excluye por un tiempo.

    Usa un TLS de 3 fases (no 2): con solo 2 fases, cuando la única
    estancada excluye a la fase actual de alternativas, el modo drenaje
    (`test_stall_fallback_drains_heaviest_absolute_queue_not_all`) la
    reactiva de inmediato por ser la única candidata — comportamiento
    correcto, pero que enmascararía la ventana de exclusión que este test
    mide. Con una tercera fase sin demanda de por medio, esa alternativa
    siempre queda disponible y la exclusión se sostiene todo el cooldown.
    """
    from controllers.max_pressure import MaxPressureController

    fake_traci.trafficlight.programs['tl3'] = FakeLogic(phases=(
        FakePhase(duration=10.0, state='Grrrrr'),
        FakePhase(duration=3.0, state='yrrrrr'),
        FakePhase(duration=10.0, state='rrGrrr'),
        FakePhase(duration=3.0, state='rryrrr'),
        FakePhase(duration=10.0, state='rrrrGr'),
        FakePhase(duration=3.0, state='rrrryr'),
    ))
    fake_traci.trafficlight.links['tl3'] = [
        [('in_a', 'out_a', '')], [('in_a2', 'out_a2', '')],
        [('in_b', 'out_b', '')], [('in_b2', 'out_b2', '')],
        [('in_c', 'out_c', '')], [('in_c2', 'out_c2', '')],
    ]
    # Longitud realista (capacidad 20): evita que el propio mecanismo de
    # congestión (no lo que este test mide) altere min_green/histéresis y
    # enmascare el conteo de reintentos del detector de estancamiento.
    for lane in ('in_a', 'in_a2', 'in_b', 'in_b2', 'in_c', 'in_c2'):
        fake_traci.lane.lengths[lane] = 150.0
    fake_traci.lane.halting['in_a'] = 5
    fake_traci.lane.halting['in_a2'] = 5  # fase 0: presión enorme, cola fija (nunca baja)

    ctl = MaxPressureController()
    ctl.register('tl3')

    green0_applied_times = []
    first_stall_window = None  # (inicio, fin) de la primera exclusión observada
    t = 0.0
    while t < 400.0:
        applied = ctl.tick(t)
        if applied and fake_traci.trafficlight.states['tl3'][-1] == 'Grrrrr':
            green0_applied_times.append(t)
        stalled_until = ctl._tls['tl3']['stalled_until'][0]
        if first_stall_window is None and stalled_until > 0:
            first_stall_window = (stalled_until - 60.0, stalled_until)  # stall_cooldown_sec=60
        t += 1.0

    assert first_stall_window is not None  # se marcó estancada en algún momento

    # Ninguna reaplicación de la fase 0 debería caer durante esa primera
    # exclusión (el propio mecanismo puede volver a estancarse más tarde y
    # eso es esperado: no forma parte de lo que este test mide).
    start, end = first_stall_window
    assert not any(gt for gt in green0_applied_times if start < gt < end)


def test_neighbor_stall_propagates_to_upstream_pressure(fake_traci):
    """Coordinación cruce a cruce: si tl2 tiene una fase estancada que entra
    por 'shared_lane', tl1 debe ver 'shared_lane' como bloqueada cuando es
    su carril de salida, aunque su propia lectura del carril esté vacía."""
    from controllers.max_pressure import MaxPressureController, compute_pressure

    _setup_2phase(fake_traci)  # tl1: fase0 sirve in_n0/in_n1 -> out_n0/out_n1

    fake_traci.trafficlight.programs['tl2'] = FakeLogic(phases=(
        FakePhase(duration=30.0, state='GGrr'),
        FakePhase(duration=3.0, state='yyrr'),
        FakePhase(duration=30.0, state='rrGG'),
        FakePhase(duration=3.0, state='rryy'),
    ))
    fake_traci.trafficlight.links['tl2'] = [
        [('out_n0', 'out2_a', '')], [('in2_1', 'out2_b', '')],
        [('in2_2', 'out2_c', '')], [('in2_3', 'out2_d', '')],
    ]

    ctl = MaxPressureController()
    ctl.register('tl1')
    ctl.register('tl2')

    blocked = ctl._blocked_lanes(0.0)
    assert 'out_n0' not in blocked  # nadie estancado todavía

    ctl._tls['tl2']['stalled_until'][0] = 500.0  # fase0 de tl2 (entra por out_n0) estancada

    blocked = ctl._blocked_lanes(100.0)
    assert 'out_n0' in blocked

    phase_links_tl1 = ctl._tls['tl1']['phase_links'][0]  # [(in_n0,out_n0), (in_n1,out_n1)]
    pressure_normal = compute_pressure(phase_links_tl1, blocked_lanes=set())
    pressure_blocked = compute_pressure(phase_links_tl1, blocked_lanes=blocked)
    assert pressure_blocked < pressure_normal  # penalizada por el vecino estancado


def test_congestion_delays_switch_via_boosted_min_green_and_hysteresis(fake_traci, monkeypatch):
    """Con ocupación local alta (>= congestion_ratio_threshold) el verde
    mínimo y la histéresis efectivos se escalan: la misma diferencia de
    presión que conmutaría sin congestión debe esperar al verde mínimo reforzado."""
    from controllers.max_pressure import MaxPressureController
    from config import MAX_PRESSURE_CONFIG

    # Re-enable low congestion threshold for this test (currently disabled at 99.0)
    monkeypatch.setitem(MAX_PRESSURE_CONFIG, "congestion_ratio_threshold", 0.3)

    _setup_2phase(fake_traci, green_duration=30.0)
    fake_traci.lane.halting['in_n0'] = 5
    fake_traci.lane.halting['in_n1'] = 5    # fase actual (0): cola alta -> ocupación alta
    fake_traci.lane.halting['in_s0'] = 10
    fake_traci.lane.halting['in_s1'] = 10   # fase 2: presión mucho mayor

    ctl = MaxPressureController()
    ctl.register('tl1')

    # Derive timings from config
    base_green = 30.0
    min_green = max(base_green * MAX_PRESSURE_CONFIG["min_green_floor"], 5.0)
    congested_min_green = max(base_green * MAX_PRESSURE_CONFIG["min_green_floor_high"], 5.0)
    check_interval = MAX_PRESSURE_CONFIG["check_interval"]

    assert ctl.tick(0.0) == 0
    # Sin congestión boost, min_green ya habilitaría decisión alrededor de t=10;
    # con el boost congestionado, el verde mínimo efectivo es mayor y todavía
    # no debe conmutar.
    assert ctl.tick(check_interval) == 0
    assert ctl.tick(2 * check_interval) == 0
    assert fake_traci.trafficlight.states == {}
    # Siguiente check tras superar congested_min_green: espera hasta
    # que ambos el tiempo y la presión (fase 2 muy por encima de fase 0)
    # sobrevivan el threshold reforzado.
    next_check_time = ((congested_min_green / check_interval) + 1) * check_interval
    assert ctl.tick(next_check_time) == 1
    assert fake_traci.trafficlight.states['tl1'][-1] == 'yyrr'


def test_stall_fallback_drains_heaviest_absolute_queue_not_all(fake_traci, monkeypatch):
    """Cuando todas las fases alternativas están estancadas, el fallback ya
    no debe reactivarlas todas a la vez (sospechoso de los colapsos
    autoinfligidos en demanda_baja/demanda_media): debe entrar en modo
    drenaje, sirviendo solo la de mayor cola de entrada absoluta y
    dejando estancadas las demás."""
    from controllers.max_pressure import MaxPressureController
    from config import MAX_PRESSURE_CONFIG as _CFG
    # Piso dinámico aplanado: este test mide stall/spillback, no el piso
    monkeypatch.setitem(_CFG, "min_green_floor_high", _CFG["min_green_floor"])
    from config import MAX_PRESSURE_CONFIG

    fake_traci.trafficlight.programs['tl3'] = FakeLogic(phases=(
        FakePhase(duration=20.0, state='Grrrrr'),
        FakePhase(duration=3.0, state='yrrrrr'),
        FakePhase(duration=20.0, state='rrGrrr'),
        FakePhase(duration=3.0, state='rryrrr'),
        FakePhase(duration=20.0, state='rrrrGr'),
        FakePhase(duration=3.0, state='rrrryr'),
    ))
    fake_traci.trafficlight.links['tl3'] = [
        [('in_a', 'out_a', '')], [('in_a2', 'out_a2', '')],
        [('in_b', 'out_b', '')], [('in_b2', 'out_b2', '')],
        [('in_c', 'out_c', '')], [('in_c2', 'out_c2', '')],
    ]
    fake_traci.lane.halting['in_a'] = 1
    fake_traci.lane.halting['in_b'] = 5
    fake_traci.lane.halting['in_c'] = 10  # corredor más cargado (fase 4)

    ctl = MaxPressureController()
    ctl.register('tl3')
    info = ctl._tls['tl3']
    assert info['current_phase'] == 0

    # Ambas alternativas quedan marcadas estancadas (lejos en el futuro).
    info['stalled_until'][2] = 1e9
    info['stalled_until'][4] = 1e9

    # Derive min_green from config: base_green=20
    base_green = 20.0
    min_green = max(base_green * MAX_PRESSURE_CONFIG["min_green_floor"], 5.0)
    check_interval = MAX_PRESSURE_CONFIG["check_interval"]
    next_check_time = ((min_green / check_interval) + 1) * check_interval

    assert ctl.tick(0.0) == 0
    assert ctl.tick(next_check_time) == 1  # verde mínimo superado: decide

    assert info['transition'] is not None
    assert info['transition']['target'] == 4  # se eligió el corredor más cargado, no la 2

    # La fase drenada se desestancó; la otra alternativa sigue marcada.
    assert info['stalled_until'][4] == 0.0
    assert info['stalled_until'][2] == 1e9


def test_full_but_flowing_output_lane_reduces_pressure(fake_traci):
    """Un carril de salida físicamente lleno pero avanzando (halting=0,
    occupancy alta) no debe pasar desapercibido: lane_queue_ratio por sí
    sola daría out_ratio=0 y no vería el spillback inminente (causa raíz
    del deadlock de demanda_media/seed_43)."""
    from controllers.max_pressure import compute_pressure

    phase_links = [('in_a', 'out_a')]
    fake_traci.lane.halting['in_a'] = 0
    fake_traci.lane.halting['out_a'] = 0  # salida "libre" según halting
    fake_traci.lane.occupancy['out_a'] = 0.6  # pero físicamente bastante llena
    fake_traci.lane.speeds['out_a'] = 0.3  # y parada (bajo el umbral blando)

    pressure_flowing_full = compute_pressure(phase_links)

    fake_traci.lane.occupancy['out_a'] = 0.0  # misma cola, salida realmente libre
    pressure_free = compute_pressure(phase_links)

    assert pressure_flowing_full < pressure_free


def test_full_and_fast_output_lane_does_not_reduce_pressure(fake_traci):
    """Un carril de salida lleno pero fluyendo a velocidad de crucero
    (occupancy alta, speed alta) NO debe penalizar la fase: es tráfico
    normal en un tramo saturado, no spillback. Antes de este chequeo, el
    término blando usaba max(ratio, occupancy) sin condición de velocidad
    y sobre-penalizaba este caso (causa del colapso de inserción en
    demanda_media/42-43)."""
    from controllers.max_pressure import compute_pressure

    phase_links = [('in_a', 'out_a')]
    fake_traci.lane.halting['in_a'] = 0
    fake_traci.lane.halting['out_a'] = 0
    fake_traci.lane.occupancy['out_a'] = 0.9  # lleno...
    fake_traci.lane.speeds['out_a'] = 8.0     # ...pero fluyendo rápido

    pressure_full_flowing = compute_pressure(phase_links)

    fake_traci.lane.occupancy['out_a'] = 0.0  # salida realmente libre
    pressure_free = compute_pressure(phase_links)

    assert pressure_full_flowing == pressure_free  # sin penalización por ocupación


def test_spillback_gate_requires_low_speed_not_just_high_occupancy(fake_traci):
    """El gate duro exige ocupación alta Y velocidad baja: un carril lleno
    pero drenando (velocidad normal) no debe disparar el bloqueo forzado,
    aunque supere el umbral de ocupación."""
    from controllers.max_pressure import (
        COORDINATION_BLOCK_RATIO,
        compute_pressure,
    )

    phase_links = [('in_a', 'out_a')]
    fake_traci.lane.halting['in_a'] = 0
    fake_traci.lane.occupancy['out_a'] = 0.9  # sobre el gate de ocupación
    fake_traci.lane.speeds['out_a'] = 8.0     # pero fluyendo: no está atascado

    pressure_flowing = compute_pressure(phase_links)
    assert pressure_flowing > -COORDINATION_BLOCK_RATIO  # el gate no se disparó

    fake_traci.lane.speeds['out_a'] = 0.2  # ahora sí, parado de verdad
    pressure_stalled = compute_pressure(phase_links)
    assert pressure_stalled == -COORDINATION_BLOCK_RATIO  # gate disparado


def test_spillback_gate_makes_full_output_lane_lose_to_alternative(fake_traci, monkeypatch):
    """Gate duro: una fase cuya salida está >= spillback_occupancy_gate debe
    perder siempre frente a una alternativa con entrada menor pero salida
    libre — evita repetir vehículos varados dentro del cruce."""
    from controllers.max_pressure import MaxPressureController
    from config import MAX_PRESSURE_CONFIG as _CFG
    # Piso dinámico aplanado: este test mide stall/spillback, no el piso
    monkeypatch.setitem(_CFG, "min_green_floor_high", _CFG["min_green_floor"])
    from config import MAX_PRESSURE_CONFIG

    _setup_2phase(fake_traci)
    for lane in ('in_n0', 'in_n1', 'in_s0', 'in_s1'):
        fake_traci.lane.lengths[lane] = 150.0

    # Fase actual (0): entrada fuerte, pero su salida está casi llena.
    fake_traci.lane.halting['in_n0'] = 8
    fake_traci.lane.halting['in_n1'] = 8
    fake_traci.lane.occupancy['out_n0'] = 0.9
    fake_traci.lane.occupancy['out_n1'] = 0.9
    fake_traci.lane.speeds['out_n0'] = 0.2  # parado de verdad, no solo lleno
    fake_traci.lane.speeds['out_n1'] = 0.2

    # Fase 2: entrada más floja, pero salida completamente libre.
    fake_traci.lane.halting['in_s0'] = 2
    fake_traci.lane.halting['in_s1'] = 2
    fake_traci.lane.occupancy['out_s0'] = 0.0
    fake_traci.lane.occupancy['out_s1'] = 0.0

    ctl = MaxPressureController()
    ctl.register('tl1')

    # Derive min_green from config: base_green=30
    base_green = 30.0
    min_green = max(base_green * MAX_PRESSURE_CONFIG["min_green_floor"], 5.0)
    check_interval = MAX_PRESSURE_CONFIG["check_interval"]
    next_check_time = ((min_green / check_interval) + 1) * check_interval

    assert ctl.tick(0.0) == 0
    assert ctl.tick(next_check_time) == 1  # min_green superado: el gate fuerza el cambio
    assert ctl._tls['tl1']['transition']['target'] == 2


def test_set_cycle_budget_caps_reference_green_max(fake_traci):
    from controllers.max_pressure import MaxPressureController
    from config import MAX_PRESSURE_CONFIG

    _setup_2phase(fake_traci)
    fake_traci.lane.halting['in_n0'] = 5
    fake_traci.lane.halting['in_n1'] = 5

    ctl = MaxPressureController()
    ctl.register('tl1')
    assert ctl.set_cycle_budget('tl1', 20.0) is True  # referencia = 10s/fase
    # el techo de referencia (10s) queda por debajo del min_green,
    # así que set_cycle_budget lo clampea al min_green.

    # Derive min_green from config: base_green=30
    base_green = 30.0
    min_green = max(base_green * MAX_PRESSURE_CONFIG["min_green_floor"], 5.0)
    check_interval = MAX_PRESSURE_CONFIG["check_interval"]
    next_check_time = ((min_green / check_interval) + 1) * check_interval

    assert ctl.tick(0.0) == 0
    assert ctl.tick(next_check_time) == 1
    assert fake_traci.trafficlight.states['tl1'][-1] == 'yyrr'

    assert ctl.set_cycle_budget('nonexistent', 10.0) is False


def _set_vehicles(fake_traci, n_halted, n_moving):
    """Puebla el fake de vehículos: n_halted a v=0.0, n_moving a v=10.0."""
    fake_traci.vehicle.order.clear()
    fake_traci.vehicle.vehicle_speeds.clear()
    for i in range(n_halted):
        vid = f'halted_{i}'
        fake_traci.vehicle.order.append(vid)
        fake_traci.vehicle.vehicle_speeds[vid] = 0.0
    for i in range(n_moving):
        vid = f'moving_{i}'
        fake_traci.vehicle.order.append(vid)
        fake_traci.vehicle.vehicle_speeds[vid] = 10.0


def test_freeze_detector_does_not_trigger_with_flowing_traffic(fake_traci):
    """Tráfico fluido (fracción detenida baja) nunca dispara recuperación,
    aunque se sostenga por mucho más que freeze_trigger_sec."""
    from controllers.max_pressure import MaxPressureController

    _setup_2phase(fake_traci)
    _set_vehicles(fake_traci, n_halted=2, n_moving=18)  # 10% detenido, muy bajo umbral 0.7

    ctl = MaxPressureController()
    ctl.register('tl1')

    t = 0.0
    while t < 120.0:
        applied = ctl.tick(t)
        assert ctl._recovery_until is None
        t += 10.0

    assert fake_traci.trafficlight.set_program_calls == []


def test_freeze_detector_triggers_recovery_after_sustained_window(fake_traci):
    """Fracción detenida >= freeze_halt_fraction sostenida por
    freeze_trigger_sec dispara recuperación: setProgram con el program id
    original en todos los TLS registrados."""
    from controllers.max_pressure import MaxPressureController

    _setup_2phase(fake_traci)
    fake_traci.trafficlight.programs['tl2'] = FakeLogic(phases=(
        FakePhase(duration=30.0, state='GGrr'),
        FakePhase(duration=3.0, state='yyrr'),
        FakePhase(duration=30.0, state='rrGG'),
        FakePhase(duration=3.0, state='rryy'),
    ))
    fake_traci.trafficlight.links['tl2'] = [
        [('in2_n0', 'out2_n0', '')], [('in2_n1', 'out2_n1', '')],
        [('in2_s0', 'out2_s0', '')], [('in2_s1', 'out2_s1', '')],
    ]
    fake_traci.trafficlight.program_id['tl1'] = 'orig1'
    fake_traci.trafficlight.program_id['tl2'] = 'orig2'

    ctl = MaxPressureController()
    ctl.register('tl1')
    ctl.register('tl2')

    _set_vehicles(fake_traci, n_halted=9, n_moving=1)  # 90% detenido >= 0.7

    t = 0.0
    triggered_at = None
    while t <= 90.0:
        ctl.tick(t)
        if ctl._recovery_until is not None:
            triggered_at = t
            break
        t += 10.0

    assert triggered_at is not None
    assert ('tl1', 'orig1') in fake_traci.trafficlight.set_program_calls
    assert ('tl2', 'orig2') in fake_traci.trafficlight.set_program_calls


def test_decide_does_not_run_during_recovery(fake_traci):
    """Durante recuperación, MP no toma ninguna decisión: 0 cambios de
    estado (setRedYellowGreenState) pese a presión/tiempo que normalmente
    forzarían un cambio."""
    from controllers.max_pressure import MaxPressureController

    _setup_2phase(fake_traci)
    for lane in ('in_n0', 'in_n1', 'in_s0', 'in_s1'):
        fake_traci.lane.lengths[lane] = 150.0
    fake_traci.lane.halting['in_s0'] = 5
    fake_traci.lane.halting['in_s1'] = 5  # presión fuerte para forzar cambio de no estar en recuperación

    ctl = MaxPressureController()
    ctl.register('tl1')
    ctl._recovery_until = 1000.0  # forzar entrada en recuperación directamente

    for t in (0.0, 10.0, 50.0, 100.0, 500.0):
        assert ctl.tick(t) == 0

    assert fake_traci.trafficlight.states == {}


def test_recovery_expires_and_resumes_mp_with_clean_stall_state(fake_traci):
    """Al expirar recovery_duration_sec, MP retoma decisiones y el estado
    de estancamiento quedó limpio (borrón y cuenta nueva)."""
    from controllers.max_pressure import MaxPressureController

    _setup_2phase(fake_traci)
    for lane in ('in_n0', 'in_n1', 'in_s0', 'in_s1'):
        fake_traci.lane.lengths[lane] = 150.0

    ctl = MaxPressureController()
    ctl.register('tl1')

    info = ctl._tls['tl1']
    info['stall_count'][0] = 2
    info['stall_strikes'][0] = 1
    info['stalled_until'][0] = 999.0
    info['last_halting'][0] = 7

    ctl._recovery_until = 100.0
    fake_traci.trafficlight.phase['tl1'] = 2  # SUMO dejó corriendo la fase 2 del plan original

    assert ctl.tick(50.0) == 0  # sigue en recuperación

    assert ctl.tick(100.0) == 0  # expira: re-sincroniza y sale, sin decidir todavía en este tick
    assert ctl._recovery_until is None
    assert info['current_phase'] == 2
    assert info['stall_count'][0] == 0
    assert info['stall_strikes'][0] == 0
    assert info['stalled_until'][0] == 0.0
    assert info['last_halting'][0] is None

    # MP retoma decisiones normalmente tras la recuperación.
    fake_traci.lane.halting['in_s0'] = 5
    fake_traci.lane.halting['in_s1'] = 5
    t = 100.0
    switched = False
    while t < 300.0:
        if ctl.tick(t):
            switched = True
            break
        t += 5.0
    assert switched
