"""Tests de las anotaciones de sesgo de supervivencia en los gráficos del A/B.

Las métricas de tripinfo solo incluyen viajes COMPLETADOS: si un lado del
A/B colapsa (gridlock o bloqueo de inserción), su muestra queda dominada
por los viajes rápidos que escaparon temprano. Estos tests cubren los
helpers que anotan n por lado en leyendas/títulos cuando las muestras no
son comparables, sin necesidad de renderizar matplotlib.
"""
import pandas as pd

from visualization.ab_test import _count_never_inserted, _never_inserted_note
from visualization.plots import (
    MIN_BIN_SAMPLES,
    SAMPLE_NONCOMPARABLE_THRESHOLD_PCT,
    legend_label_with_n,
    plot_improvement_summary,
    plot_metric_comparison_bars,
    samples_noncomparable,
    title_with_sample_note,
)


def test_samples_noncomparable_flags_large_difference():
    # Caso real: corredor/seed_42 baseline completa 349 viajes vs 1022 del dinámico.
    assert samples_noncomparable(349, 1022) is True


def test_samples_noncomparable_false_within_threshold():
    # <=10% de diferencia relativa: no se anota.
    assert samples_noncomparable(349, 360) is False


def test_samples_noncomparable_handles_none_and_zero():
    assert samples_noncomparable(None, 100) is False
    assert samples_noncomparable(100, None) is False
    assert samples_noncomparable(0, 0) is False


def test_samples_noncomparable_respects_custom_threshold():
    # Diferencia del 5%: por debajo del umbral por defecto (10%) pero por
    # encima de uno más estricto (2%).
    assert samples_noncomparable(100, 105) is False
    assert samples_noncomparable(100, 105, threshold_pct=2.0) is True


def test_legend_label_with_n_format():
    assert legend_label_with_n('Baseline', 349) == 'Baseline (n=349)'
    assert legend_label_with_n('Dynamic', 1022) == 'Dynamic (n=1022)'


def test_title_with_sample_note_appends_when_noncomparable():
    title = title_with_sample_note('Travel time boxplot', 349, 1022)
    assert 'Travel time boxplot' in title
    assert 'n=349 vs n=1022' in title
    assert 'no comparables' in title
    # La nota debe apuntar al gráfico insesgado (auto-contenido: el PNG
    # puede verse suelto, fuera del HTML, así que "ver nota" a secas no
    # sirve — tiene que decir dónde está la evidencia real).
    assert '32_throughput_timeline.png' in title


def test_title_with_sample_note_unchanged_when_comparable():
    title = title_with_sample_note('Travel time boxplot', 349, 360)
    assert title == 'Travel time boxplot'


def test_sample_noncomparable_threshold_constant_matches_task_spec():
    # El umbral documentado en la auditoría es 10%.
    assert SAMPLE_NONCOMPARABLE_THRESHOLD_PCT == 10.0


def test_min_bin_samples_constant():
    assert MIN_BIN_SAMPLES == 5


def test_never_inserted_note_empty_when_none_or_zero():
    assert _never_inserted_note(None) == ''
    assert _never_inserted_note(0) == ''


def test_never_inserted_note_reports_count():
    note = _never_inserted_note(37)
    assert '37' in note
    assert 'nunca insertados' in note


def test_count_never_inserted_from_summary_columns():
    df_summary = pd.DataFrame({
        'loaded': [0, 50, 120, 200],
        'inserted': [0, 40, 90, 150],
    })
    # loaded.max()=200, inserted.max()=150 -> 50 vehículos nunca insertados.
    assert _count_never_inserted(df_summary) == 50


def test_count_never_inserted_none_when_columns_missing():
    df_summary = pd.DataFrame({'running': [1, 2, 3]})
    assert _count_never_inserted(df_summary) is None


def test_count_never_inserted_none_when_empty_or_no_backlog():
    assert _count_never_inserted(pd.DataFrame()) is None
    df_summary = pd.DataFrame({'loaded': [0, 100], 'inserted': [0, 100]})
    assert _count_never_inserted(df_summary) is None


def test_improvement_summary_uses_median_for_noncomparable_metric(tmp_path):
    """Caso real corredor/seed_42: departDelay medio "empeora" -281% porque
    el baseline colapsado (349 completados) solo insertó los pocos
    vehículos con poca espera de origen, mientras que el dinámico (1022
    completados) admitió casi toda la demanda incluyendo los que esperaron
    mucho para entrar. La media compara peras con manzanas; la mediana
    (igual que en la tabla del HTML) sí refleja la mejora real."""
    stats_a = {
        'duration': {'mean': 75.76, 'median': 70.0, 'count': 349},
        'departDelay': {'mean': 149.88, 'median': 83.92, 'count': 349},
    }
    stats_b = {
        'duration': {'mean': 57.60, 'median': 50.0, 'count': 1022},
        'departDelay': {'mean': 571.63, 'median': 21.54, 'count': 1022},
    }
    plot_improvement_summary(stats_a, stats_b, str(tmp_path),
                             labels=('Baseline', 'Dynamic'), filename='18.png')
    assert (tmp_path / '18.png').exists()


def test_metric_comparison_bars_uses_median_for_noncomparable_metric(tmp_path, monkeypatch):
    """06_metric_comparison_bars.png tenía el mismo bug que 18: el % de
    diferencia de departDelay se calculaba con la media cruda, mostrando
    +281% de "empeoramiento" cuando en realidad la mediana muestra una
    mejora real (el baseline colapsado solo insertó los pocos vehículos
    con poca espera de origen). Verificamos que la anotación use la
    mediana — parcheamos ax.annotate para capturar el texto anotado sin
    depender de inspeccionar píxeles del PNG."""
    import matplotlib.axes

    captured = []
    original_annotate = matplotlib.axes.Axes.annotate

    def spy_annotate(self, text, *args, **kwargs):
        captured.append(text)
        return original_annotate(self, text, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, 'annotate', spy_annotate)

    stats_a = {
        'duration': {'mean': 75.76, 'median': 70.0, 'std': 10, 'count': 349},
        'departDelay': {'mean': 149.88, 'median': 83.92, 'std': 10, 'count': 349},
    }
    stats_b = {
        'duration': {'mean': 57.60, 'median': 50.0, 'std': 10, 'count': 1022},
        'departDelay': {'mean': 571.63, 'median': 21.54, 'std': 10, 'count': 1022},
    }
    plot_metric_comparison_bars(stats_a, stats_b, ['duration', 'departDelay'],
                                str(tmp_path), labels=('Baseline', 'Dynamic'),
                                filename='06.png')

    # departDelay: mediana 83.92 -> 21.54 baja ~74.3% (B < A, verde = mejora
    # en la convención de este gráfico), no el +281% "en contra" que daría
    # la media cruda dominada por sesgo de supervivencia.
    depart_delay_annotation = captured[1]
    assert depart_delay_annotation.startswith('~-74.')


def test_improvement_summary_median_switch_matches_html_table_logic():
    """El signo/valor de la mejora reportada para una métrica no comparable
    debe coincidir con el cálculo que ya usa `_write_html_report` (mediana),
    no con la media cruda que domina el sesgo de supervivencia."""
    n_a, n_b = 349, 1022
    assert samples_noncomparable(n_a, n_b) is True
    median_a, median_b = 83.92, 21.54
    expected_pct = (median_a - median_b) / median_a * 100
    assert expected_pct > 0  # mejora real, no la "degradación" -281% de la media
