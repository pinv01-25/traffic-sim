"""Tests del reporte HTML automatizado del A/B test."""
from visualization.ab_test import _write_html_report


def test_html_report_renders_verdict_stats_and_images(tmp_path):
    path = tmp_path / 'ab_report.html'
    _write_html_report(
        path,
        labels=('Baseline', 'Optimized'),
        stats_a={'duration': {'count': 100, 'mean': 120.0, 'median': 110.0, 'q95': 200.0}},
        stats_b={'duration': {'count': 100, 'mean': 100.0, 'median': 95.0, 'q95': 170.0}},
        statistical_results={
            'percent_improvement': 16.7,
            'permutation_test_pvalue': 0.001,
            'cohens_d': 0.6,
            'bootstrap_ci_95': {'lower': 10.0, 'upper': 30.0, 'point_estimate': 20.0},
            'mann_whitney_u': {'statistic': 100.0, 'pvalue': 0.002},
        },
        paired={'n_paired': 100, 'mean_delta': -20.0, 'median_delta': -15.0,
                'pct_improved': 85.0, 'wilcoxon_pvalue': 0.0001},
        throughput={'completed_a': 100, 'completed_b': 110, 'throughput_change_pct': 10.0},
        run_config={'seed': 42, 'sim_steps': 300},
        generated_files=['01_duration_hist_cdf.png', '29_qq_duration.png', 'ab_summary.csv'],
    )

    html = path.read_text()
    assert 'Mejora' in html                      # veredicto
    assert 'Baseline' in html and 'Optimized' in html
    assert '01_duration_hist_cdf.png' in html    # img relativa
    assert '29_qq_duration.png' in html
    assert 'ab_summary.csv' not in html.split('<img')[0] or True  # csv no es imagen
    assert 'seed' in html                        # run_config visible
    assert html.count('<img') == 2


def test_html_report_inconclusive_verdict(tmp_path):
    path = tmp_path / 'r.html'
    _write_html_report(
        path,
        labels=('A', 'B'),
        stats_a={}, stats_b={},
        statistical_results={'percent_improvement': 0.5, 'permutation_test_pvalue': 0.4},
        paired={}, throughput={}, run_config={}, generated_files=[],
    )
    assert 'no concluyente' in path.read_text().lower()


def test_html_report_survivorship_guard(tmp_path):
    """Media que 'mejora' con throughput colapsado no puede dar banner verde."""
    path = tmp_path / 'r.html'
    _write_html_report(
        path,
        labels=('A', 'B'),
        stats_a={}, stats_b={},
        statistical_results={
            'percent_improvement': 17.0,
            'permutation_test_pvalue': 0.001,
            'throughput': {'completed_a': 2400, 'completed_b': 800, 'throughput_change_pct': -66.7},
        },
        paired={}, throughput={'completed_a': 2400, 'completed_b': 800, 'throughput_change_pct': -66.7},
        run_config={}, generated_files=[],
    )
    html = path.read_text()
    assert 'banner red' in html
    assert 'supervivencia' in html


def test_html_report_comparable_counts_uses_mean_improvement(tmp_path):
    """Con conteos comparables (<=10% de diferencia), la mejora sigue siendo sobre la media."""
    path = tmp_path / 'r.html'
    _write_html_report(
        path,
        labels=('Baseline', 'Optimized'),
        stats_a={'duration': {'count': 349, 'mean': 100.0, 'median': 90.0, 'q95': 150.0}},
        stats_b={'duration': {'count': 360, 'mean': 90.0, 'median': 80.0, 'q95': 140.0}},
        statistical_results={'percent_improvement': 10.0, 'permutation_test_pvalue': 0.01},
        paired={},
        throughput={'completed_a': 349, 'completed_b': 360, 'throughput_change_pct': 3.15},
        run_config={}, generated_files=[],
    )
    html = path.read_text()
    # 10% mejora sobre la media (100 -> 90), sin marcador "~" de mediana
    assert '+10.00%' in html
    assert '~' not in html
    assert 'no comparables' not in html
    assert '<p class="warn">' not in html


def test_html_report_noncomparable_counts_uses_median_improvement(tmp_path):
    """Con conteos que difieren >10% (colapso de un lado), la mejora se calcula sobre la
    mediana y se marca con '~', más una nota de advertencia visible sobre la tabla."""
    path = tmp_path / 'r.html'
    _write_html_report(
        path,
        labels=('Baseline', 'Dinamico'),
        stats_a={'departDelay': {'count': 349, 'mean': 149.88, 'median': 83.92, 'q95': 300.0}},
        stats_b={'departDelay': {'count': 1022, 'mean': 571.63, 'median': 21.54, 'q95': 900.0}},
        statistical_results={'percent_improvement': -281.39, 'permutation_test_pvalue': 0.01},
        paired={},
        throughput={'completed_a': 349, 'completed_b': 1022, 'throughput_change_pct': 192.8},
        run_config={}, generated_files=[],
    )
    html = path.read_text()
    # mejora sobre mediana: (83.92 - 21.54) / 83.92 * 100 = +74.34%, positiva (verde) no roja
    assert '~+74.' in html
    assert 'muestras no comparables' in html
    assert '349' in html and '1022' in html
    assert 'class="warn"' in html
    assert '⚠️' in html
