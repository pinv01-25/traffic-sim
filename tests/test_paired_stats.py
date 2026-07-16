"""Tests del análisis pareado por vehículo (misma flota en A y B)."""
import pandas as pd

from visualization.ab_test import paired_statistics


def _make_runs():
    ids = [f'veh{i}' for i in range(10)]
    dur_a = [100.0 + i for i in range(10)]
    # 8 vehículos mejoran 10s, 2 empeoran 5s
    dur_b = [a - 10 for a in dur_a[:8]] + [a + 5 for a in dur_a[8:]]
    df_a = pd.DataFrame({'id': ids + ['solo_en_A'], 'duration': dur_a + [999.0]})
    df_b = pd.DataFrame({'id': ids + ['solo_en_B'], 'duration': dur_b + [999.0]})
    return df_a, df_b


def test_paired_statistics_basic():
    df_a, df_b = _make_runs()
    result = paired_statistics(df_a, df_b, metric='duration')

    assert result['n_paired'] == 10  # los no-pareados se excluyen
    assert result['pct_improved'] == 80.0
    assert result['mean_delta'] == (-10 * 8 + 5 * 2) / 10
    assert result['median_delta'] == -10.0
    assert 0.0 <= result['wilcoxon_pvalue'] <= 1.0


def test_paired_statistics_empty_or_disjoint():
    df_a = pd.DataFrame({'id': ['a'], 'duration': [1.0]})
    df_b = pd.DataFrame({'id': ['b'], 'duration': [2.0]})
    assert paired_statistics(df_a, df_b) == {}
    assert paired_statistics(pd.DataFrame(), df_b) == {}
