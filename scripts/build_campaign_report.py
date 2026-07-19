"""Generador de reporte HTML autocontenido para la mega-campaña A/B.

Agrega resultados de results/campanas/<campana>/<escenario>/seed_<n>/ab_report.json
en un único archivo HTML (CSS inline, imágenes matplotlib embebidas en base64).

La campaña puede estar corriendo todavía: el script agrega lo que exista y
reporta corridas faltantes, sin fallar si faltan celdas de la grilla.

Uso:
    uv run python scripts/build_campaign_report.py \
        [--out results/campanas/reporte_final.html] \
        [--analysis results/campanas/analysis.md] \
        [--root results/campanas]
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import markdown as markdown_lib  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_MERMAID_JS = REPO_ROOT / "scripts" / "vendor" / "mermaid.min.js"

CAMPAIGNS = ["mopc_60", "netconvert_90", "largo_120"]
CAMPAIGN_LABELS = {
    "mopc_60": "Fixed-Time MOPC (60s / 25s)",
    "netconvert_90": "Netconvert (90s / 42s)",
    "largo_120": "Largo (120s / 57s)",
}
# ciclo (s), verde perdido por ciclo (s) -> tiempo perdido por ciclo (%)
CAMPAIGN_CYCLE_LOSS = {
    "mopc_60": (60, 10),
    "netconvert_90": (90, 6),
    "largo_120": (120, 6),
}
SCENARIOS = [
    "corredor",
    "demanda_alta",
    "demanda_baja",
    "demanda_media",
    "demanda_saturada",
    "hora_pico",
]
N_SEEDS_EXPECTED = 10

FAVORABLE_VERDICTS = {"improvement", "misleading_regression"}
DESFAVORABLE_VERDICTS = {"regression", "misleading_improvement"}
TIE_VERDICTS = {"tie", "inconclusive", "no_conclusivo"}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def discover_runs(root: Path) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """Devuelve {campana: {escenario: {seed_dir_name: parsed_json}}}."""
    data: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for campaign in CAMPAIGNS:
        data[campaign] = {}
        for scenario in SCENARIOS:
            data[campaign][scenario] = {}
            scen_dir = root / campaign / scenario
            if not scen_dir.is_dir():
                continue
            for seed_dir in sorted(scen_dir.glob("seed_*")):
                report_path = seed_dir / "ab_report.json"
                if not report_path.is_file():
                    continue
                try:
                    parsed = json.loads(report_path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                data[campaign][scenario][seed_dir.name] = parsed
    return data


def missing_runs(data: Dict[str, Dict[str, Dict[str, dict]]], seeds: List[int]) -> List[str]:
    missing = []
    for campaign in CAMPAIGNS:
        for scenario in SCENARIOS:
            present = set(data.get(campaign, {}).get(scenario, {}).keys())
            for seed in seeds:
                key = f"seed_{seed}"
                if key not in present:
                    missing.append(f"{campaign}/{scenario}/{key}")
    return missing


def verdict_bucket(verdict: Optional[str]) -> str:
    if verdict in FAVORABLE_VERDICTS:
        return "favorable"
    if verdict in DESFAVORABLE_VERDICTS:
        return "desfavorable"
    return "tie"


# ---------------------------------------------------------------------------
# Helpers matplotlib -> base64
# ---------------------------------------------------------------------------

def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _img_tag(data_uri: str, alt: str) -> str:
    return f'<img src="{data_uri}" alt="{html.escape(alt)}" style="max-width:100%;height:auto;">'


# ---------------------------------------------------------------------------
# Sección 2: grilla de veredictos (heatmap)
# ---------------------------------------------------------------------------

def build_verdict_grid_images(data, seeds: List[int]) -> Dict[str, str]:
    color_map = {"favorable": "#2e7d32", "tie": "#9e9e9e", "desfavorable": "#c62828", "faltante": "#eeeeee"}
    images = {}
    for campaign in CAMPAIGNS:
        n_scen = len(SCENARIOS)
        n_seeds = len(seeds)
        fig, ax = plt.subplots(figsize=(max(6, n_seeds * 0.6), max(3, n_scen * 0.5)))
        grid = []
        for scenario in SCENARIOS:
            row = []
            for seed in seeds:
                run = data.get(campaign, {}).get(scenario, {}).get(f"seed_{seed}")
                if run is None:
                    row.append("faltante")
                else:
                    verdict = run.get("statistical_tests", {}).get("verdict", {}).get("verdict")
                    row.append(verdict_bucket(verdict))
            grid.append(row)

        for i, row in enumerate(grid):
            for j, bucket in enumerate(row):
                ax.add_patch(plt.Rectangle((j, len(grid) - 1 - i), 1, 1,
                                            facecolor=color_map[bucket], edgecolor="white"))
        ax.set_xlim(0, n_seeds)
        ax.set_ylim(0, n_scen)
        ax.set_xticks([x + 0.5 for x in range(n_seeds)])
        ax.set_xticklabels([str(s) for s in seeds], rotation=45, fontsize=8)
        ax.set_yticks([y + 0.5 for y in range(n_scen)])
        ax.set_yticklabels(list(reversed(SCENARIOS)), fontsize=8)
        ax.set_title(f"Veredictos — {CAMPAIGN_LABELS[campaign]}", fontsize=10)
        ax.set_aspect("equal")
        images[campaign] = _fig_to_data_uri(fig)
    return images


# ---------------------------------------------------------------------------
# Sección 3: distribuciones (boxplots)
# ---------------------------------------------------------------------------

def _collect_metric_by_scenario(data, campaign: str, metric_path: List[str]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {s: [] for s in SCENARIOS}
    for scenario in SCENARIOS:
        for run in data.get(campaign, {}).get(scenario, {}).values():
            node: Any = run
            ok = True
            for key in metric_path:
                if isinstance(node, dict) and key in node:
                    node = node[key]
                else:
                    ok = False
                    break
            if ok and isinstance(node, (int, float)):
                out[scenario].append(float(node))
    return out


def build_distribution_image(data, metric_path: List[str], title: str, ylabel: str) -> Optional[str]:
    campaign_colors = {"mopc_60": "#1565c0", "netconvert_90": "#ef6c00", "largo_120": "#6a1b9a"}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    positions = []
    box_data = []
    colors = []
    labels = []
    width = 0.25
    any_data = False
    for i, scenario in enumerate(SCENARIOS):
        for j, campaign in enumerate(CAMPAIGNS):
            values = _collect_metric_by_scenario(data, campaign, metric_path)[scenario]
            if values:
                any_data = True
            pos = i + (j - 1) * width
            positions.append(pos)
            box_data.append(values if values else [float("nan")])
            colors.append(campaign_colors[campaign])
        labels.append(scenario)
    if not any_data:
        plt.close(fig)
        return None
    bp = ax.boxplot(box_data, positions=positions, widths=width * 0.9, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(len(SCENARIOS)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=campaign_colors[c], alpha=0.7) for c in CAMPAIGNS]
    ax.legend(handles, [CAMPAIGN_LABELS[c] for c in CAMPAIGNS], fontsize=8, loc="best")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


# ---------------------------------------------------------------------------
# Sección 5: scatter thr% vs system_time%
# ---------------------------------------------------------------------------

def build_scatter_image(data) -> Optional[str]:
    campaign_colors = {"mopc_60": "#1565c0", "netconvert_90": "#ef6c00", "largo_120": "#6a1b9a"}
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    any_data = False
    for campaign in CAMPAIGNS:
        xs, ys = [], []
        for scenario in SCENARIOS:
            for run in data.get(campaign, {}).get(scenario, {}).values():
                st = run.get("statistical_tests", {})
                thr = st.get("throughput", {}).get("throughput_change_pct")
                sysT = st.get("system_time", {}).get("system_time_improvement_pct")
                if thr is not None and sysT is not None:
                    xs.append(thr)
                    ys.append(sysT)
        if xs:
            any_data = True
            ax.scatter(xs, ys, color=campaign_colors[campaign], alpha=0.7,
                       label=CAMPAIGN_LABELS[campaign], edgecolors="white", s=45)
    if not any_data:
        plt.close(fig)
        return None
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1] * 0.95, ylim[1] * 0.95, "coherente\n(mejora)", ha="right", va="top",
            fontsize=8, color="#2e7d32")
    ax.text(xlim[0] * 0.95, ylim[0] * 0.95, "coherente\n(regresión)", ha="left", va="bottom",
            fontsize=8, color="#c62828")
    ax.text(xlim[1] * 0.95, ylim[0] * 0.95, "incoherente\n(thr+ / sys-)", ha="right", va="bottom",
            fontsize=8, color="#9e9e9e")
    ax.text(xlim[0] * 0.95, ylim[1] * 0.95, "incoherente\n(thr- / sys+)", ha="left", va="top",
            fontsize=8, color="#9e9e9e")
    ax.set_xlabel("Δ Throughput (%)")
    ax.set_ylabel("Δ Tiempo de sistema (%)")
    ax.set_title("Coherencia thr% vs tiempo de sistema%")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


# ---------------------------------------------------------------------------
# Markdown -> HTML (librería `markdown` de PyPI, extensiones tables + fenced_code)
# ---------------------------------------------------------------------------
#
# Nota de diseño: se descartó deliberadamente PlantUML para los diagramas
# (requiere un runtime Java o un servidor de render aparte) a favor de
# Mermaid, que cubre flowcharts y diagramas de secuencia y se renderiza
# 100% en el navegador con un único <script> JS vendorizado — el reporte
# sigue siendo un archivo HTML autocontenido, sin red ni dependencias
# externas en tiempo de visualización.

_MERMAID_FENCE_RE = re.compile(r"```mermaid\n(.*?)\n```", re.S)


def markdown_to_html(md_text: str) -> str:
    """Convierte markdown (tablas GFM, negritas/cursivas, listas, código, encabezados)
    a HTML usando la librería `markdown`, preservando los bloques ```mermaid``` intactos
    (se extraen antes de pasar por el conversor y se reinyectan después como
    <pre class="mermaid"> para que Mermaid.js los renderice en el navegador)."""
    mermaid_blocks: List[str] = []

    def _stash(match: "re.Match[str]") -> str:
        mermaid_blocks.append(match.group(1))
        return f"MERMAIDBLOCKPLACEHOLDER{len(mermaid_blocks) - 1}ENDPLACEHOLDER"

    stashed_text = _MERMAID_FENCE_RE.sub(_stash, md_text)
    converted = markdown_lib.markdown(
        stashed_text, extensions=["tables", "fenced_code"]
    )

    def _restore(match: "re.Match[str]") -> str:
        idx = int(match.group(1))
        code = html.escape(mermaid_blocks[idx])
        return f'<pre class="mermaid">{code}</pre>'

    converted = re.sub(
        r"(?:<p>)?MERMAIDBLOCKPLACEHOLDER(\d+)ENDPLACEHOLDER(?:</p>)?",
        _restore,
        converted,
    )
    return converted


# ---------------------------------------------------------------------------
# Cálculos de agregación
# ---------------------------------------------------------------------------

def campaign_score(data, campaign: str) -> Dict[str, int]:
    counts = {"favorable": 0, "tie": 0, "desfavorable": 0, "n": 0}
    for scenario in SCENARIOS:
        for run in data.get(campaign, {}).get(scenario, {}).values():
            verdict = run.get("statistical_tests", {}).get("verdict", {}).get("verdict")
            counts[verdict_bucket(verdict)] += 1
            counts["n"] += 1
    return counts


def global_score(data) -> Dict[str, int]:
    totals = {"favorable": 0, "tie": 0, "desfavorable": 0, "n": 0}
    for campaign in CAMPAIGNS:
        cs = campaign_score(data, campaign)
        for k in totals:
            totals[k] += cs[k]
    return totals


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def baseline_comparison_table(data) -> List[Dict[str, Any]]:
    rows = []
    for scenario in SCENARIOS:
        row: Dict[str, Any] = {"scenario": scenario}
        for campaign in CAMPAIGNS:
            thr_vals, sys_vals = [], []
            for run in data.get(campaign, {}).get(scenario, {}).values():
                st = run.get("statistical_tests", {})
                thr = st.get("throughput", {}).get("throughput_change_pct")
                sysT = st.get("system_time", {}).get("system_time_improvement_pct")
                if thr is not None:
                    thr_vals.append(thr)
                if sysT is not None:
                    sys_vals.append(sysT)
            cycle, lost = CAMPAIGN_CYCLE_LOSS[campaign]
            row[campaign] = {
                "median_thr": median(thr_vals),
                "median_sys": median(sys_vals),
                "n": len(thr_vals),
                "cycle_loss_pct": 100.0 * lost / cycle,
            }
        rows.append(row)
    return rows


def statistical_rigor(data) -> Dict[str, Any]:
    out = {}
    for campaign in CAMPAIGNS:
        pvals = []
        ds = []
        for scenario in SCENARIOS:
            for run in data.get(campaign, {}).get(scenario, {}).values():
                st = run.get("statistical_tests", {})
                p = st.get("paired", {}).get("wilcoxon_pvalue")
                if p is not None:
                    pvals.append(p)
                d = st.get("cohens_d")
                if d is not None:
                    ds.append(d)
        out[campaign] = {
            "n": len(pvals),
            "p_lt_05": sum(1 for p in pvals if p < 0.05),
            "p_lt_01": sum(1 for p in pvals if p < 0.01),
            "mean_cohens_d": (sum(ds) / len(ds)) if ds else None,
        }
    return out


# ---------------------------------------------------------------------------
# Construcción del HTML
# ---------------------------------------------------------------------------

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
  max-width: 1180px;
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
  color: #1a1a1a;
  background: #fafafa;
  line-height: 1.5;
}
h1 { font-size: 1.8rem; border-bottom: 3px solid #263238; padding-bottom: 0.4rem; }
h2 { font-size: 1.35rem; margin-top: 2.6rem; color: #263238; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
h3 { font-size: 1.05rem; color: #37474f; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.88rem; background: white; }
th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: right; }
th { background: #263238; color: white; text-align: center; }
td:first-child, th:first-child { text-align: left; }
tr:nth-child(even) { background: #f4f4f4; }
.badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; color: white; }
.badge.favorable { background: #2e7d32; }
.badge.tie { background: #757575; }
.badge.desfavorable { background: #c62828; }
.score-card { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.score-box { flex: 1; min-width: 220px; background: white; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; }
.score-box h3 { margin: 0 0 0.5rem; }
.score-bar { height: 10px; border-radius: 5px; overflow: hidden; display: flex; margin-top: 0.5rem; }
.figure { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 1rem 0; text-align: center; }
.note { font-size: 0.85rem; color: #555; background: #fff8e1; border-left: 4px solid #f9a825; padding: 0.6rem 0.9rem; margin: 1rem 0; }
.missing { font-size: 0.85rem; color: #b71c1c; }
code { background: #eee; padding: 0.1rem 0.3rem; border-radius: 3px; }
pre.mermaid { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 1rem 0; text-align: center; }
footer { margin-top: 3rem; font-size: 0.8rem; color: #777; }
"""


def _mermaid_script_tag() -> str:
    """Embebe mermaid.min.js inline (vendorizado en scripts/vendor/) para que el
    reporte HTML siga siendo un único archivo autocontenido, sin red."""
    if not VENDOR_MERMAID_JS.is_file():
        return "<!-- mermaid.min.js no vendorizado: ver scripts/vendor/mermaid.min.js -->"
    js = VENDOR_MERMAID_JS.read_text()
    return (
        f"<script>{js}</script>\n"
        "<script>mermaid.initialize({startOnLoad:true, theme:'neutral'});</script>"
    )


def _score_box_html(title: str, counts: Dict[str, int]) -> str:
    n = counts["n"] or 1
    fav_pct = 100.0 * counts["favorable"] / n
    tie_pct = 100.0 * counts["tie"] / n
    des_pct = 100.0 * counts["desfavorable"] / n
    return f"""
    <div class="score-box">
      <h3>{html.escape(title)}</h3>
      <div>
        <span class="badge favorable">{counts['favorable']} favorables</span>
        <span class="badge tie">{counts['tie']} ties</span>
        <span class="badge desfavorable">{counts['desfavorable']} desfavorables</span>
        <span style="color:#555;font-size:0.8rem;"> de {counts['n']}</span>
      </div>
      <div class="score-bar">
        <div style="width:{fav_pct}%;background:#2e7d32;"></div>
        <div style="width:{tie_pct}%;background:#9e9e9e;"></div>
        <div style="width:{des_pct}%;background:#c62828;"></div>
      </div>
    </div>"""


def fmt(x, digits=1, suffix=""):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}{suffix}"
    return f"{x}{suffix}"


def render_config_table() -> str:
    try:
        import config as config_module
    except Exception as exc:  # pragma: no cover
        return f"<p class='missing'>No se pudo leer config.py: {html.escape(str(exc))}</p>"

    cfg = getattr(config_module, "MAX_PRESSURE_CONFIG", {})
    config_path = Path(config_module.__file__)
    source = config_path.read_text()
    m = re.search(r"MAX_PRESSURE_CONFIG\s*=\s*\{(.*?)\n\}", source, re.S)
    comments: Dict[str, str] = {}
    if m:
        body = m.group(1)
        for line in body.splitlines():
            line = line.strip()
            kv = re.match(r'"([a-zA-Z0-9_]+)"\s*:.*?#\s*(.*)$', line)
            if kv:
                comments[kv.group(1)] = kv.group(2)
    rows = []
    for key, value in cfg.items():
        rows.append(
            f"<tr><td><code>{html.escape(key)}</code></td>"
            f"<td>{html.escape(str(value))}</td>"
            f"<td style='text-align:left;'>{html.escape(comments.get(key, ''))}</td></tr>"
        )
    return (
        "<table><thead><tr><th>parámetro</th><th>valor</th><th>comentario</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def build_report_html(data, seeds: List[int], analysis_path: Optional[Path]) -> str:
    total = global_score(data)
    missing = missing_runs(data, seeds)
    n_total_expected = len(CAMPAIGNS) * len(SCENARIOS) * len(seeds)
    n_present = n_total_expected - len(missing)

    # Sección 1
    exec_boxes = _score_box_html("GLOBAL", total) + "".join(
        _score_box_html(CAMPAIGN_LABELS[c], campaign_score(data, c)) for c in CAMPAIGNS
    )

    # Sección 2
    grid_images = build_verdict_grid_images(data, seeds)
    grid_html = "".join(
        f'<div class="figure"><h3>{html.escape(CAMPAIGN_LABELS[c])}</h3>{_img_tag(img, c)}</div>'
        for c, img in grid_images.items()
    )

    # Sección 3
    dist_specs = [
        (["throughput", "throughput_change_pct"], "Δ Throughput (%) por escenario", "% cambio throughput"),
        (["system_time", "system_time_improvement_pct"], "Δ Tiempo de sistema (%) por escenario", "% mejora tiempo de sistema"),
        (["paired", "pct_improved"], "% viajes pareados mejorados por escenario", "% pct_improved"),
    ]
    dist_html_parts = []
    for path, title, ylabel in dist_specs:
        img = build_distribution_image(data, path, title, ylabel)
        if img:
            dist_html_parts.append(f'<div class="figure">{_img_tag(img, title)}</div>')
        else:
            dist_html_parts.append(f'<p class="missing">Sin datos suficientes para: {html.escape(title)}</p>')
    dist_html = "".join(dist_html_parts)

    # Sección 4
    comp_rows = baseline_comparison_table(data)
    comp_table_rows = []
    for row in comp_rows:
        cells = [f"<td>{html.escape(row['scenario'])}</td>"]
        for campaign in CAMPAIGNS:
            c = row[campaign]
            cells.append(
                f"<td>Δthr {fmt(c['median_thr'], 1, '%')} / Δsys {fmt(c['median_sys'], 1, '%')} "
                f"(n={c['n']}, pérdida ciclo {fmt(c['cycle_loss_pct'], 1, '%')})</td>"
            )
        comp_table_rows.append(f"<tr>{''.join(cells)}</tr>")
    comp_table = (
        "<table><thead><tr><th>escenario</th>"
        + "".join(f"<th>{html.escape(CAMPAIGN_LABELS[c])}</th>" for c in CAMPAIGNS)
        + "</tr></thead><tbody>" + "".join(comp_table_rows) + "</tbody></table>"
    )

    # Sección 5
    rigor = statistical_rigor(data)
    rigor_rows = "".join(
        f"<tr><td>{html.escape(CAMPAIGN_LABELS[c])}</td>"
        f"<td>{rigor[c]['n']}</td>"
        f"<td>{rigor[c]['p_lt_05']}</td>"
        f"<td>{rigor[c]['p_lt_01']}</td>"
        f"<td>{fmt(rigor[c]['mean_cohens_d'], 3)}</td></tr>"
        for c in CAMPAIGNS
    )
    rigor_table = (
        "<table><thead><tr><th>campaña</th><th>n corridas</th><th>p&lt;0.05</th>"
        "<th>p&lt;0.01</th><th>Cohen's d medio</th></tr></thead>"
        f"<tbody>{rigor_rows}</tbody></table>"
    )
    scatter_img = build_scatter_image(data)
    scatter_html = (
        f'<div class="figure">{_img_tag(scatter_img, "scatter thr vs sys time")}</div>'
        if scatter_img else '<p class="missing">Sin datos suficientes para el scatter.</p>'
    )

    # Sección 6
    if analysis_path and analysis_path.is_file():
        analysis_html = markdown_to_html(analysis_path.read_text())
    else:
        analysis_html = '<p class="missing">Análisis pendiente — este contenido lo escribe otro proceso.</p>'

    # Sección 7
    config_table = render_config_table()
    missing_html = (
        f"<p class='missing'>Corridas faltantes: {len(missing)} de {n_total_expected} "
        f"({n_present} presentes).</p><ul>"
        + "".join(f"<li><code>{html.escape(m)}</code></li>" for m in missing[:200])
        + ("<li>… (truncado)</li>" if len(missing) > 200 else "")
        + "</ul>"
        if missing
        else "<p>Sin corridas faltantes: la campaña está completa.</p>"
    )
    seeds_html = ", ".join(str(s) for s in seeds)

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte de campaña — traffic-sim</title>
<style>{CSS}</style>
{_mermaid_script_tag()}
</head>
<body>
<h1>Reporte de mega-campaña A/B — Optimización dinámica vs. planes fijos</h1>
<p>Comparación de <strong>{len(CAMPAIGNS)} baselines</strong> × <strong>{len(SCENARIOS)} escenarios</strong>
× <strong>{len(seeds)} seeds</strong> = {n_total_expected} corridas esperadas
({n_present} presentes al momento de generar este reporte).</p>

<h2>1. Resumen ejecutivo</h2>
<div class="score-card">{exec_boxes}</div>
<div class="note">
<strong>Nota metodológica.</strong> El veredicto de cada corrida clasifica en <em>favorable</em>
los casos <code>improvement</code> y <code>misleading_regression</code> (mejora insesgada aunque
la media cruda diga lo contrario por censura de supervivencia), en <em>desfavorable</em>
<code>regression</code> y <code>misleading_improvement</code> (mejora aparente de la media
enmascarando colapso de throughput), y el resto como <em>tie</em> / no concluyente.
Esta convención evita que una caída de throughput (menos viajes completados, por lo tanto
menor censura de los peores viajes) se lea como mejora de tiempos.
</div>

<h2>2. Grilla de veredictos (escenario × seed)</h2>
{grid_html}

<h2>3. Distribuciones</h2>
{dist_html}

<h2>4. Comparación entre baselines</h2>
<p>Mediana de Δthroughput y Δtiempo de sistema por escenario y campaña, junto con el
tiempo perdido por ciclo del plan fijo (verde perdido en amarillo/todo-rojo respecto
al ciclo total): 60s → 16.7%, 90s → 6.7%, 120s → 5%.</p>
{comp_table}

<h2>5. Rigor estadístico</h2>
{rigor_table}
<h3>Coherencia throughput% vs tiempo de sistema%</h3>
{scatter_html}

<h2 id="analisis">6. Análisis técnico</h2>
{analysis_html}

<h2>7. Apéndice de reproducibilidad</h2>
<h3>Seeds</h3>
<p><code>{html.escape(seeds_html)}</code></p>
<h3>Comandos de reproducción</h3>
<pre><code>uv run python run_ab.py --scenario &lt;escenario&gt; --seed &lt;seed&gt; --cycle-time &lt;60|90|120&gt;
uv run python scripts/build_campaign_report.py --out results/campanas/reporte_final.html</code></pre>
<h3>MAX_PRESSURE_CONFIG (config.py)</h3>
{config_table}
<h3>Corridas faltantes</h3>
{missing_html}

<footer>Generado por scripts/build_campaign_report.py</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_seeds(root: Path) -> List[int]:
    seeds_path = root.parent / "campaign_seeds.json" if root.name == "campanas" else root / "campaign_seeds.json"
    candidates = [root.parent / "campaign_seeds.json", REPO_ROOT / "results" / "campaign_seeds.json"]
    for c in candidates:
        if c.is_file():
            try:
                return json.loads(c.read_text())
            except (json.JSONDecodeError, OSError):
                pass
    return []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "campanas" / "reporte_final.html")
    parser.add_argument("--analysis", type=Path, default=REPO_ROOT / "results" / "campanas" / "analysis.md")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "results" / "campanas")
    args = parser.parse_args(argv)

    data = discover_runs(args.root)
    seeds = load_seeds(args.root) or list(range(N_SEEDS_EXPECTED))
    html_text = build_report_html(data, seeds, args.analysis)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text)

    total = global_score(data)
    missing = missing_runs(data, seeds)
    print(f"Reporte escrito en {args.out}")
    print(f"Corridas presentes: {total['n']} / {len(CAMPAIGNS) * len(SCENARIOS) * len(seeds)}")
    print(f"Favorables={total['favorable']} Tie={total['tie']} Desfavorables={total['desfavorable']}")
    print(f"Faltantes: {len(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
