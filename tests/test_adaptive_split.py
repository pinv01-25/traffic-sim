from controllers.adaptive_split import compute_split


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
