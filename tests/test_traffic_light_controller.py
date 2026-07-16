"""Tests del controlador de semáforos (camino dinámico)."""


def test_update_traffic_light_clamps_and_delegates(fake_traci, monkeypatch):
    import controllers.traffic_light_controller as tlc

    calls = []
    monkeypatch.setattr(
        tlc, 'apply_durations_to_tls',
        lambda tls_id, green, red, **kw: calls.append((tls_id, green, red)) or True,
    )

    controller = tlc.TrafficLightController()
    ok = controller.update_traffic_light(
        'tl1', {'optimization': {'green_time_sec': 200, 'red_time_sec': 2}}
    )

    assert ok
    # Clamp a [min_phase_duration=10, max_phase_duration=120]
    assert calls == [('tl1', 120, 10)]


def test_update_traffic_light_returns_false_on_failure(fake_traci, monkeypatch):
    import controllers.traffic_light_controller as tlc

    monkeypatch.setattr(tlc, 'apply_durations_to_tls', lambda *a, **k: False)
    controller = tlc.TrafficLightController()
    assert not controller.update_traffic_light(
        'tl1', {'optimization': {'green_time_sec': 30, 'red_time_sec': 30}}
    )
