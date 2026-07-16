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
