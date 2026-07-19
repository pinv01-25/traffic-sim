"""Tests del generador de reporte de mega-campaña con JSON sintéticos mínimos."""
import json

import pytest

from scripts.build_campaign_report import (
    build_report_html,
    discover_runs,
    global_score,
    markdown_to_html,
    missing_runs,
    verdict_bucket,
)


def _make_ab_report(verdict="improvement", seed=1):
    return {
        "labels": ["Fixed", "Dynamic"],
        "run_config": {"seed": seed, "cycle_time": 60.0},
        "statistical_tests": {
            "permutation_test_pvalue": 0.01,
            "cohens_d": 0.4,
            "bootstrap_ci_95": {"lower": 1.0, "upper": 5.0, "point_estimate": 3.0},
            "mann_whitney_u": {"statistic": 100.0, "pvalue": 0.01},
            "percent_improvement": 10.0,
            "paired": {
                "n_paired": 50,
                "mean_delta": -2.0,
                "median_delta": -1.5,
                "pct_improved": 60.0,
                "wilcoxon_pvalue": 0.02,
            },
            "throughput": {
                "completed_a": 100,
                "completed_b": 150,
                "throughput_change_pct": 50.0,
            },
            "system_time": {
                "system_time_a": 1000.0,
                "system_time_b": 700.0,
                "system_time_improvement_pct": 30.0,
                "mean_time_per_vehicle_a": 10.0,
                "mean_time_per_vehicle_b": 7.0,
                "mean_time_per_vehicle_improvement_pct": 30.0,
            },
            "verdict": {"verdict": verdict, "reason": "sintético", "headline_pct": 30.0},
        },
    }


@pytest.fixture
def synthetic_root(tmp_path):
    root = tmp_path / "campanas"
    for campaign, scenario, seed, verdict in [
        ("mopc_60", "corredor", 1, "improvement"),
        ("mopc_60", "corredor", 2, "regression"),
        ("mopc_60", "demanda_baja", 1, "tie"),
    ]:
        run_dir = root / campaign / scenario / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "ab_report.json").write_text(json.dumps(_make_ab_report(verdict, seed)))
    return root


def test_discover_runs_finds_partial_data(synthetic_root):
    data = discover_runs(synthetic_root)
    assert "seed_1" in data["mopc_60"]["corredor"]
    assert "seed_2" in data["mopc_60"]["corredor"]
    assert data["mopc_60"]["demanda_baja"]["seed_1"]["statistical_tests"]["verdict"]["verdict"] == "tie"
    # escenarios/campañas sin datos existen como dict vacío (no rompe)
    assert data["netconvert_90"]["corredor"] == {}


def test_verdict_bucket_classification():
    assert verdict_bucket("improvement") == "favorable"
    assert verdict_bucket("misleading_regression") == "favorable"
    assert verdict_bucket("regression") == "desfavorable"
    assert verdict_bucket("misleading_improvement") == "desfavorable"
    assert verdict_bucket("tie") == "tie"
    assert verdict_bucket(None) == "tie"


def test_missing_runs_reports_gaps(synthetic_root):
    data = discover_runs(synthetic_root)
    missing = missing_runs(data, seeds=[1, 2, 3])
    # 3 campañas x 6 escenarios x 3 seeds = 54 combinaciones esperadas, solo 3 presentes
    assert len(missing) == 54 - 3
    assert "netconvert_90/corredor/seed_1" in missing


def test_global_score_counts_favorable_and_desfavorable(synthetic_root):
    data = discover_runs(synthetic_root)
    score = global_score(data)
    assert score["n"] == 3
    assert score["favorable"] == 1
    assert score["desfavorable"] == 1
    assert score["tie"] == 1


def test_build_report_html_renders_with_partial_data(synthetic_root):
    html_text = build_report_html(data=discover_runs(synthetic_root), seeds=[1, 2], analysis_path=None)
    assert "<html" in html_text
    assert "Resumen ejecutivo" in html_text
    assert "data:image/png;base64" in html_text
    assert "análisis pendiente" in html_text.lower()


def test_markdown_to_html_renders_tables_and_preserves_mermaid_blocks():
    md_text = (
        "## Título\n\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n\n"
        "Texto con **negrita** y `codigo`.\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        "  A --> B\n"
        "```\n"
    )
    out = markdown_to_html(md_text)
    assert "<table>" in out
    assert "<th>a</th>" in out
    assert "<strong>negrita</strong>" in out
    assert '<pre class="mermaid">' in out
    assert "flowchart LR" in out
    # el bloque mermaid no debe pasar por el conversor de markdown
    assert "<p>flowchart" not in out
    # no deben quedar artefactos crudos de markdown fuera de bloques de código
    assert "##" not in out.replace('<pre class="mermaid">', "")


def test_build_report_html_embeds_mermaid_script_and_diagrams(synthetic_root, tmp_path):
    analysis_path = tmp_path / "analysis.md"
    analysis_path.write_text(
        "## 1. Sección\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        "  A --> B\n"
        "```\n\n"
        "*Figura: ejemplo.*\n"
    )
    html_text = build_report_html(
        data=discover_runs(synthetic_root), seeds=[1, 2], analysis_path=analysis_path
    )
    assert "mermaid.initialize" in html_text
    assert '<pre class="mermaid">' in html_text
