"""Tests del veredicto A/B insesgado (compute_verdict, system_time_statistics)."""
import numpy as np
import pandas as pd

from visualization.ab_test import compute_verdict, system_time_statistics


# =============================================================================
# system_time_statistics
# =============================================================================

def _summary_df(running, inserted):
    return pd.DataFrame({'time': list(range(len(running))), 'running': running, 'inserted': inserted})


def test_system_time_statistics_basic():
    # A: 10 pasos con 10 vehículos circulando siempre; insertó 10.
    df_a = _summary_df([10] * 10, list(range(1, 11)))
    # B: mismo tiempo de sistema pero llegó a insertar menos vehículos.
    df_b = _summary_df([10] * 10, list(range(1, 6)) + [5] * 5)
    result = system_time_statistics(df_a, df_b)
    assert result['system_time_a'] == 100.0
    assert result['system_time_b'] == 100.0
    assert result['system_time_improvement_pct'] == 0.0
    assert result['inserted_a'] == 10.0
    assert result['inserted_b'] == 5.0
    # B normaliza peor por vehículo insertado (mismo tiempo, menos vehículos).
    assert result['mean_time_per_vehicle_b'] > result['mean_time_per_vehicle_a']


def test_system_time_statistics_empty():
    result = system_time_statistics(pd.DataFrame(), pd.DataFrame())
    assert result['system_time_a'] != result['system_time_a']  # NaN
    assert result['system_time_improvement_pct'] != result['system_time_improvement_pct']


def test_system_time_statistics_gridlock_case():
    """Caso demanda_baja/seed_44: B colapsa, atrapa vehículos → tiempo de
    sistema sube aunque la media de duration (sesgada) hubiera bajado."""
    # A: tráfico fluido, pocos vehículos circulando por paso.
    df_a = _summary_df([15] * 100, [i for i in range(1, 101)])
    # B: gridlock, muchos vehículos atrapados circulando "despacio" por paso.
    df_b = _summary_df([40] * 100, [min(i, 60) for i in range(1, 101)])
    result = system_time_statistics(df_a, df_b)
    assert result['system_time_b'] > result['system_time_a']
    assert result['system_time_improvement_pct'] < 0  # empeora, insesgado


# =============================================================================
# compute_verdict
# =============================================================================

def test_verdict_collapse_to_zero_is_tie():
    st = {
        'percent_improvement': 50.0,
        'throughput': {'completed_a': 0, 'completed_b': 100, 'throughput_change_pct': float('nan')},
    }
    v = compute_verdict(st)
    assert v['verdict'] == 'tie'
    assert 'colapso' in v['reason'].lower()


def test_verdict_throughput_collapse_with_positive_mean_is_misleading_improvement():
    """Caso real: demanda_baja/seed_44 — imp +18.1%, throughput -55%."""
    st = {
        'percent_improvement': 18.1,
        'permutation_test_pvalue': 0.0001,
        'throughput': {'completed_a': 1430, 'completed_b': 638, 'throughput_change_pct': -55.38},
    }
    v = compute_verdict(st)
    assert v['verdict'] == 'misleading_improvement'
    assert 'supervivencia' in v['reason'].lower()


def test_verdict_throughput_collapse_with_negative_mean_is_regression():
    st = {
        'percent_improvement': -10.0,
        'throughput': {'completed_a': 1000, 'completed_b': 400, 'throughput_change_pct': -60.0},
    }
    v = compute_verdict(st)
    assert v['verdict'] == 'regression'


def test_verdict_baseline_collapse_with_negative_mean_is_misleading_regression():
    """Caso real: demanda_alta/seed_44 — imp -52%, throughput +116% (baseline colapsó)."""
    st = {
        'percent_improvement': -52.0,
        'permutation_test_pvalue': 0.0001,
        'throughput': {'completed_a': 1049, 'completed_b': 2268, 'throughput_change_pct': 116.2},
    }
    v = compute_verdict(st)
    assert v['verdict'] == 'misleading_regression'
    assert 'colaps' in v['reason'].lower()


def test_verdict_throughput_up_with_positive_mean_is_improvement():
    st = {
        'percent_improvement': 20.0,
        'throughput': {'completed_a': 1000, 'completed_b': 1300, 'throughput_change_pct': 30.0},
    }
    v = compute_verdict(st)
    assert v['verdict'] == 'improvement'


def test_verdict_stable_throughput_uses_system_time_when_available():
    st = {
        'percent_improvement': 8.0,  # sesgado, no debería usarse como headline
        'permutation_test_pvalue': 0.001,
        'throughput': {'completed_a': 1000, 'completed_b': 1020, 'throughput_change_pct': 2.0},
        'system_time': {'system_time_improvement_pct': -6.0},  # insesgado: empeora
    }
    v = compute_verdict(st)
    assert v['verdict'] == 'regression'
    assert v['headline_pct'] == -6.0


def test_verdict_stable_throughput_small_headline_is_tie():
    st = {
        'percent_improvement': 1.0,
        'permutation_test_pvalue': 0.001,
        'throughput': {'completed_a': 1000, 'completed_b': 1005, 'throughput_change_pct': 0.5},
    }
    v = compute_verdict(st)
    assert v['verdict'] == 'tie'


def test_verdict_stable_throughput_not_significant_is_tie():
    st = {
        'percent_improvement': 5.0,
        'permutation_test_pvalue': 0.4,
        'throughput': {'completed_a': 1000, 'completed_b': 1005, 'throughput_change_pct': 0.5},
    }
    v = compute_verdict(st)
    assert v['verdict'] == 'tie'


def test_verdict_no_data_is_tie():
    v = compute_verdict({})
    assert v['verdict'] == 'tie'
    assert v['headline_pct'] is None


def test_verdict_paired_wilcoxon_counts_as_significant():
    st = {
        'percent_improvement': 5.0,
        'throughput': {'completed_a': 1000, 'completed_b': 1005, 'throughput_change_pct': 0.5},
        'paired': {'wilcoxon_pvalue': 0.001},
    }
    v = compute_verdict(st)
    assert v['verdict'] == 'improvement'
