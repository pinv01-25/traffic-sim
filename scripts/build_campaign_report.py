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
import datetime as _dt
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


class FigureCounter:
    """Numerador incremental para captions 'Figura N. ...'."""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


def _figure_html(data_uri: str, alt: str, caption: str, counter: FigureCounter) -> str:
    n = counter.next()
    return (
        f'<figure class="figure">{_img_tag(data_uri, alt)}'
        f"<figcaption>Figura {n}. {html.escape(caption)}</figcaption></figure>"
    )


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
:root {
  color-scheme: light;
  --ink: #1c1a17;
  --muted: #66605a;
  --faint: #948d84;
  --paper: #fdfcfa;
  --panel: #f7f5f1;
  --rule: #ddd7cc;
  --rule-strong: #a89e8f;
  --accent: #2b5960;
  --favorable: #2f6b4f;
  --desfavorable: #a1382a;
  --tie: #7a7268;
  --serif: Charter, 'Iowan Old Style', 'Source Serif 4', Georgia, 'Times New Roman', serif;
  --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  font-family: var(--serif);
  color: var(--ink);
  background: var(--paper);
  margin: 0;
  line-height: 1.6;
  font-size: 17px;
}
.page {
  max-width: 76ch;
  margin: 0 auto;
  padding: 3.5rem 1.5rem 5rem;
}
.wide { max-width: 96ch; }

/* ---- Cover ---- */
.cover {
  max-width: 96ch;
  margin: 0 auto;
  padding: 4.5rem 1.5rem 3rem;
  border-bottom: 1px solid var(--rule);
}
.cover .kicker {
  font-family: var(--sans);
  text-transform: uppercase;
  letter-spacing: 0.09em;
  font-size: 0.75rem;
  color: var(--accent);
  margin: 0 0 1rem;
}
.cover h1 {
  font-family: var(--serif);
  font-size: 2.05rem;
  line-height: 1.28;
  font-weight: 600;
  margin: 0 0 0.75rem;
  color: var(--ink);
}
.cover .subtitle {
  font-size: 1.08rem;
  color: var(--muted);
  margin: 0 0 2rem;
  max-width: 62ch;
}
.cover .meta {
  font-family: var(--sans);
  font-size: 0.85rem;
  color: var(--muted);
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1.75rem;
  border-top: 1px solid var(--rule);
  padding-top: 1rem;
}
.cover .meta strong { color: var(--ink); font-weight: 600; }

/* ---- Table of contents ---- */
nav.toc {
  max-width: 96ch;
  margin: 0 auto;
  padding: 1.75rem 1.5rem 2.25rem;
  border-bottom: 1px solid var(--rule);
  font-family: var(--sans);
}
nav.toc h2 {
  font-family: var(--sans);
  text-transform: uppercase;
  letter-spacing: 0.07em;
  font-size: 0.78rem;
  color: var(--muted);
  border: none;
  margin: 0 0 0.85rem;
  padding: 0;
}
nav.toc ol { list-style: none; margin: 0; padding: 0; columns: 2; column-gap: 2.5rem; }
nav.toc li { break-inside: avoid; margin-bottom: 0.45rem; font-size: 0.92rem; }
nav.toc a { color: var(--ink); text-decoration: none; border-bottom: 1px solid transparent; }
nav.toc a:hover { border-bottom-color: var(--accent); }
nav.toc .num { color: var(--faint); font-variant-numeric: tabular-nums; margin-right: 0.4rem; }

/* ---- Headings ---- */
h1 { font-size: 1.8rem; }
h2 {
  font-size: 1.4rem;
  font-weight: 600;
  margin-top: 3.2rem;
  margin-bottom: 1rem;
  color: var(--ink);
  border-bottom: 1px solid var(--rule-strong);
  padding-bottom: 0.4rem;
  scroll-margin-top: 1.5rem;
}
h3 {
  font-size: 1.08rem;
  font-weight: 600;
  color: var(--ink);
  margin-top: 2rem;
  scroll-margin-top: 1.5rem;
}
p { margin: 0.9rem 0; }
a { color: var(--accent); }

/* ---- Tables ---- */
.table-scroll { overflow-x: auto; margin: 1.25rem 0; }
table {
  border-collapse: collapse;
  width: 100%;
  font-family: var(--sans);
  font-size: 0.86rem;
  font-variant-numeric: tabular-nums;
}
thead th {
  background: var(--panel);
  color: var(--ink);
  font-weight: 600;
  text-align: right;
  border-bottom: 2px solid var(--rule-strong);
  padding: 0.55rem 0.85rem;
  white-space: nowrap;
}
thead th:first-child, td:first-child { text-align: left; }
tbody td {
  padding: 0.5rem 0.85rem;
  border-bottom: 1px solid var(--rule);
  text-align: right;
  vertical-align: top;
}
tbody tr:nth-child(even) { background: #faf9f6; }
tbody tr:last-child td { border-bottom: 2px solid var(--rule-strong); }
.num-pos { color: var(--favorable); }
.num-neg { color: var(--desfavorable); }

/* ---- Verdict marks (text + dot, not pills) ---- */
.verdict { font-family: var(--sans); font-size: 0.85rem; white-space: nowrap; }
.verdict::before {
  content: "";
  display: inline-block;
  width: 0.5em;
  height: 0.5em;
  border-radius: 50%;
  margin-right: 0.45em;
}
.verdict.favorable { color: var(--favorable); }
.verdict.favorable::before { background: var(--favorable); }
.verdict.tie { color: var(--tie); }
.verdict.tie::before { background: var(--tie); }
.verdict.desfavorable { color: var(--desfavorable); }
.verdict.desfavorable::before { background: var(--desfavorable); }

/* ---- Score summary ---- */
.score-card { display: flex; gap: 1px; flex-wrap: wrap; margin: 1.5rem 0; background: var(--rule); border: 1px solid var(--rule); }
.score-box { flex: 1; min-width: 200px; background: var(--paper); padding: 1.1rem 1.25rem; }
.score-box h3 { margin: 0 0 0.7rem; font-family: var(--sans); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
.score-box .counts { font-family: var(--sans); font-size: 0.85rem; color: var(--ink); }
.score-box .counts .n { color: var(--muted); }
.score-bar { height: 6px; overflow: hidden; display: flex; margin-top: 0.65rem; background: var(--rule); }

/* ---- Figures ---- */
.figure { margin: 1.75rem 0; text-align: center; }
.figure img { max-width: 100%; height: auto; }
figcaption, .caption {
  font-family: var(--serif);
  font-style: italic;
  font-size: 0.85rem;
  color: var(--muted);
  margin-top: 0.65rem;
  text-align: left;
}

/* ---- Callouts ---- */
.note {
  font-family: var(--sans);
  font-size: 0.88rem;
  color: var(--ink);
  background: var(--panel);
  border-left: 3px solid var(--accent);
  padding: 0.85rem 1.1rem;
  margin: 1.25rem 0;
  line-height: 1.55;
}
.missing { font-family: var(--sans); font-size: 0.85rem; color: var(--desfavorable); }

code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--panel);
  padding: 0.1rem 0.35rem;
  font-size: 0.85em;
  border: 1px solid var(--rule);
}
pre { background: var(--panel); border: 1px solid var(--rule); padding: 0.9rem 1.1rem; overflow-x: auto; font-size: 0.85rem; }
pre code { background: none; border: none; padding: 0; }

pre.mermaid {
  background: var(--paper);
  border: 1px solid var(--rule);
  padding: 1.25rem;
  margin: 1.5rem 0;
  text-align: center;
  overflow-x: auto;
  max-width: 100%;
}

footer {
  max-width: 96ch;
  margin: 4rem auto 0;
  padding: 1.5rem;
  border-top: 1px solid var(--rule);
  font-family: var(--sans);
  font-size: 0.78rem;
  color: var(--faint);
}

@media print {
  body { background: white; }
  .cover { break-after: page; }
  h2 { break-before: page; }
  a { color: var(--ink); }
  .table-scroll { overflow-x: visible; }
}
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
      <div class="counts">
        <span class="verdict favorable">{counts['favorable']} favorables</span> &nbsp;
        <span class="verdict tie">{counts['tie']} ties</span> &nbsp;
        <span class="verdict desfavorable">{counts['desfavorable']} desfavorables</span>
        <span class="n"> &nbsp;de {counts['n']}</span>
      </div>
      <div class="score-bar">
        <div style="width:{fav_pct}%;background:var(--favorable);"></div>
        <div style="width:{tie_pct}%;background:var(--tie);"></div>
        <div style="width:{des_pct}%;background:var(--desfavorable);"></div>
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
        '<div class="table-scroll"><table><thead><tr><th>parámetro</th>'
        "<th>valor</th><th>comentario</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def build_report_html(data, seeds: List[int], analysis_path: Optional[Path]) -> str:
    total = global_score(data)
    missing = missing_runs(data, seeds)
    n_total_expected = len(CAMPAIGNS) * len(SCENARIOS) * len(seeds)
    n_present = n_total_expected - len(missing)
    fig_counter = FigureCounter()

    # Sección 1
    exec_boxes = _score_box_html("Global", total) + "".join(
        _score_box_html(CAMPAIGN_LABELS[c], campaign_score(data, c)) for c in CAMPAIGNS
    )

    # Sección 2
    grid_images = build_verdict_grid_images(data, seeds)
    grid_intro = (
        "<p>Cada mapa de calor siguiente corresponde a un plan fijo (A) evaluado contra "
        "el control dinámico Max-Pressure (B) y despliega, en una grilla, el veredicto "
        "individual de cada corrida: las filas son los escenarios de demanda y las "
        "columnas son las semillas aleatorias (seeds) que fijan el patrón de tránsito "
        "simulado. Cada celda se colorea según el veredicto de esa corrida puntual — "
        "verde para favorable a B, gris para empate y rojo para desfavorable a B — "
        "y las celdas blancas marcan corridas todavía no ejecutadas. El patrón a "
        "observar es si el color domina de forma pareja por fila (un escenario que "
        "favorece o perjudica sistemáticamente a B, independientemente de la semilla) "
        "o si se dispersa de forma irregular (sensibilidad a la semilla, no al escenario).</p>"
    )
    grid_html = grid_intro + "".join(
        _figure_html(
            img,
            c,
            f"Veredicto por escenario (fila) y semilla (columna) para el plan fijo "
            f"{CAMPAIGN_LABELS[c]} (A) contra el control dinámico Max-Pressure (B). "
            "Verde = favorable a B, gris = empate, rojo = desfavorable a B; blanco = "
            "corrida pendiente.",
            fig_counter,
        )
        for c, img in grid_images.items()
    )

    # Sección 3
    dist_intro = (
        "<p>Los siguientes diagramas de caja (boxplots) resumen, por escenario de "
        "demanda, la distribución de una métrica a lo largo de todas las semillas "
        "disponibles y las tres campañas de plan fijo. En cada caja la línea central "
        "marca la mediana, los bordes superior e inferior de la caja delimitan el "
        "rango intercuartílico (percentiles 25 y 75, es decir el 50% central de las "
        "observaciones) y los bigotes se extienden hasta el valor más extremo dentro "
        "de 1.5 veces ese rango; los puntos fuera de esa franja se dibujan como "
        "valores atípicos. El color identifica la campaña (plan fijo A evaluado); "
        "la línea horizontal punteada en cero separa mejora de B (valores positivos) "
        "de regresión de B (valores negativos) respecto del plan fijo correspondiente. "
        "Interesa observar si la caja completa queda por encima o por debajo de cero "
        "(efecto consistente) o si cruza el cero (efecto no concluyente para ese "
        "escenario).</p>"
    )
    dist_specs = [
        (
            ["throughput", "throughput_change_pct"],
            "Variación porcentual de viajes completados (Δthroughput, %) por escenario",
            "Δthroughput (%)",
            "Variación porcentual del número de viajes completados por el control "
            "dinámico (B) respecto del plan fijo (A), agrupada por escenario de "
            "demanda; valores positivos indican que B completó más viajes que A.",
        ),
        (
            ["system_time", "system_time_improvement_pct"],
            "Variación porcentual del tiempo de sistema (Δtiempo de sistema, %) por escenario",
            "Δtiempo de sistema (%)",
            "Variación porcentual del tiempo total de sistema (suma de tiempos de "
            "viaje) del control dinámico (B) respecto del plan fijo (A), agrupada por "
            "escenario de demanda; valores positivos indican que B redujo el tiempo "
            "de sistema respecto de A.",
        ),
        (
            ["paired", "pct_improved"],
            "Porcentaje de viajes pareados con mejora individual de tiempo por escenario",
            "viajes pareados mejorados (%)",
            "Porcentaje de viajes, entre los pareados por origen-destino-salida entre "
            "A y B, en que el tiempo de viaje individual mejoró bajo el control "
            "dinámico (B); un valor de 50% equivale a que la mitad de los viajes "
            "pareados mejoraron y la otra mitad empeoraron.",
        ),
    ]
    dist_html_parts = [dist_intro]
    for path, title, ylabel, caption in dist_specs:
        img = build_distribution_image(data, path, title, ylabel)
        if img:
            dist_html_parts.append(_figure_html(img, title, caption, fig_counter))
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

            def _signed(x):
                if x is None:
                    return "—"
                cls = "num-pos" if x > 0 else ("num-neg" if x < 0 else "")
                return f'<span class="{cls}">{fmt(x, 1, "%")}</span>'

            cells.append(
                "<td>"
                f"Δthr {_signed(c['median_thr'])} &middot; Δsys {_signed(c['median_sys'])}"
                f"<br><span style='color:var(--faint);font-size:0.82em;'>"
                f"n={c['n']} corridas pareadas &middot; pérdida por ciclo {fmt(c['cycle_loss_pct'], 1, '%')}"
                "</span></td>"
            )
        comp_table_rows.append(f"<tr>{''.join(cells)}</tr>")
    comp_table = (
        '<div class="table-scroll"><table><thead><tr><th>escenario de demanda</th>'
        + "".join(
            f"<th>{html.escape(CAMPAIGN_LABELS[c])} (A)<br>vs. control dinámico (B)</th>"
            for c in CAMPAIGNS
        )
        + "</tr></thead><tbody>" + "".join(comp_table_rows) + "</tbody></table>"
        "<p class='caption'>Δthr = variación porcentual mediana de viajes completados "
        "(throughput) entre el plan fijo (A, columna) y el control dinámico (B), sobre "
        "las n corridas pareadas disponibles para esa celda escenario&times;campaña; "
        "valores positivos favorecen a B. Δsys = variación porcentual mediana del "
        "tiempo de sistema bajo el mismo criterio; valores positivos (reducción de "
        "tiempo) favorecen a B. \"pérdida por ciclo\" es el porcentaje de cada ciclo "
        "semafórico del plan fijo consumido en despeje de intersección (verde perdido, "
        "amarillo y todo-rojo) y no varía por escenario: 16.7% para el ciclo de 60 s, "
        "6.7% para 90 s y 5% para 120 s.</p></div>"
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
    rigor_intro = (
        "<p>La tabla siguiente resume, por cada plan fijo (A) evaluado contra el "
        "control dinámico (B), la evidencia estadística agregada sobre todas las "
        "corridas disponibles (todos los escenarios y semillas). El valor p reportado "
        "corresponde al test pareado de Wilcoxon sobre los tiempos de viaje "
        "individuales de A frente a B en cada corrida; las columnas de conteo indican "
        "en cuántas de esas corridas el resultado fue estadísticamente significativo "
        "a los umbrales convencionales p&lt;0.05 y p&lt;0.01. Cohen's d es una medida "
        "estandarizada del tamaño del efecto (diferencia de medias dividida por la "
        "desviación estándar combinada); como referencia orientativa, |d|&asymp;0.2 se "
        "considera un efecto pequeño, 0.5 mediano y 0.8 grande.</p>"
    )
    rigor_table = (
        rigor_intro
        + '<div class="table-scroll"><table><thead><tr><th>plan fijo evaluado (A)</th>'
        "<th>n corridas</th><th>corridas con p&lt;0.05</th>"
        "<th>corridas con p&lt;0.01</th><th>Cohen's d medio</th></tr></thead>"
        f"<tbody>{rigor_rows}</tbody></table>"
        "<p class='caption'>p = valor p del test de Wilcoxon pareado sobre tiempos de "
        "viaje (A vs. B) en cada corrida; \"corridas con p&lt;0.05\" y \"p&lt;0.01\" "
        "cuentan cuántas de las n corridas alcanzaron cada umbral de significancia. "
        "Cohen's d medio es el promedio del tamaño de efecto estandarizado sobre esas "
        "mismas corridas.</p></div>"
    )
    scatter_img = build_scatter_image(data)
    scatter_intro = (
        "<p>El siguiente diagrama de dispersión cruza, para cada corrida individual, "
        "la variación porcentual de throughput (eje horizontal) contra la variación "
        "porcentual de tiempo de sistema (eje vertical), ambas de B respecto de A. Un "
        "resultado es <em>coherente</em> cuando ambas métricas se mueven en la "
        "dirección esperada de forma conjunta: más viajes completados y menor tiempo "
        "de sistema (cuadrante superior derecho, mejora), o menos viajes completados "
        "y mayor tiempo de sistema (cuadrante inferior izquierdo, regresión). Los "
        "cuadrantes restantes son <em>incoherentes</em>: indican que una mejora "
        "aparente en una métrica coincide con un deterioro en la otra, típicamente "
        "porque un throughput más bajo censura del cálculo a los viajes más lentos "
        "(ver nota metodológica del resumen ejecutivo). El color identifica la "
        "campaña de origen de cada punto.</p>"
    )
    scatter_html = (
        scatter_intro + _figure_html(
            scatter_img,
            "dispersión Δthroughput vs Δtiempo de sistema",
            "Cada punto es una corrida individual (escenario, semilla, campaña); eje "
            "horizontal: Δthroughput (%) de B respecto de A; eje vertical: Δtiempo de "
            "sistema (%) de B respecto de A. Los cuadrantes superior-derecho e "
            "inferior-izquierdo son coherentes (mejora o regresión conjunta); los "
            "otros dos son incoherentes.",
            fig_counter,
        )
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

    toc_entries = [
        ("1", "Resumen ejecutivo", "resumen-ejecutivo"),
        ("2", "Grilla de veredictos", "grilla-de-veredictos"),
        ("3", "Distribuciones", "distribuciones"),
        ("4", "Comparación entre plan fijo y control dinámico", "comparacion-entre-baselines"),
        ("5", "Rigor estadístico", "rigor-estadistico"),
        ("6", "Análisis técnico", "analisis"),
        ("7", "Apéndice de reproducibilidad", "apendice-de-reproducibilidad"),
    ]
    toc_html = "".join(
        f'<li><a href="#{anchor}"><span class="num">{n}</span>{html.escape(title)}</a></li>'
        for n, title, anchor in toc_entries
    )

    generated_on = _dt.date.today().isoformat()

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte de campaña — traffic-sim</title>
<style>{CSS}</style>
{_mermaid_script_tag()}
</head>
<body>

<div class="cover">
  <p class="kicker">Traffic-sim &middot; Control jerárquico de semáforos</p>
  <h1>Optimización dinámica vs. planes fijos</h1>
  <p class="subtitle">Reporte de mega-campaña A/B: {len(CAMPAIGNS)} planes fijos (A)
  contrastados contra el control dinámico Max-Pressure (B), evaluados sobre
  {len(SCENARIOS)} escenarios de demanda con {len(seeds)} semillas aleatorias
  por celda escenario&times;campaña.</p>
  <div class="meta">
    <span><strong>Fecha</strong> — {generated_on}</span>
    <span><strong>Corridas</strong> — {n_present} / {n_total_expected}</span>
    <span><strong>Planes fijos (A)</strong> — {", ".join(CAMPAIGN_LABELS[c] for c in CAMPAIGNS)}</span>
  </div>
</div>

<nav class="toc">
  <h2>Índice</h2>
  <ol>{toc_html}</ol>
</nav>

<div class="page">

<h2 id="resumen-ejecutivo">1. Resumen ejecutivo</h2>
<p>
A lo largo de esta campaña se define A como el plan fijo bajo evaluación (uno de
{", ".join(CAMPAIGN_LABELS[c] for c in CAMPAIGNS)}, según la celda) y B como el
control dinámico Max-Pressure, ambos simulados sobre la misma red y el mismo
patrón de demanda dentro de cada corrida. Se agregaron {n_present} de
{n_total_expected} corridas planificadas ({len(CAMPAIGNS)} planes fijos &times;
{len(SCENARIOS)} escenarios de demanda &times; {len(seeds)} semillas). Sobre el
total de corridas presentes, el control dinámico resultó favorable en
{total['favorable']} de {total['n']} ({fmt(100.0 * total['favorable'] / total['n'], 1) if total['n'] else '—'}%),
en empate en {total['tie']} ({fmt(100.0 * total['tie'] / total['n'], 1) if total['n'] else '—'}%)
y desfavorable en {total['desfavorable']} ({fmt(100.0 * total['desfavorable'] / total['n'], 1) if total['n'] else '—'}%).
El desglose por plan fijo, incluido a continuación, muestra si este balance es
homogéneo entre campañas o si concentra las corridas desfavorables en algún
plan fijo en particular.
</p>
<div class="score-card">{exec_boxes}</div>
<div class="note">
<strong>Nota metodológica sobre el veredicto.</strong> Cada corrida individual (una
combinación de plan fijo, escenario y semilla) recibe un veredicto que compara A
contra B y se clasifica en una de tres categorías. Se considera <em>favorable</em>
a B cuando el resultado es <code>improvement</code> (mejora directa y no sesgada)
o <code>misleading_regression</code> (la media cruda de tiempos de viaje sugiere
una regresión, pero se debe a que B completó más viajes y por lo tanto censura
menos los viajes más lentos que A dejó sin terminar; corregido ese sesgo, B es en
realidad mejor). Se considera <em>desfavorable</em> a B cuando el resultado es
<code>regression</code> (regresión directa) o <code>misleading_improvement</code>
(la media cruda sugiere una mejora, pero esta enmascara una caída real de
throughput bajo B). El resto de los casos se clasifica como <em>empate</em> o no
concluyente. Esta convención evita que una caída de throughput bajo B —menos
viajes completados y, en consecuencia, menor censura de los viajes más lentos—
se lea erróneamente como una mejora de los tiempos de viaje.
</div>

<h2 id="grilla-de-veredictos">2. Grilla de veredictos (escenario &times; seed)</h2>
{grid_html}

<h2 id="distribuciones">3. Distribuciones</h2>
{dist_html}

<h2 id="comparacion-entre-baselines">4. Comparación entre plan fijo y control dinámico</h2>
<p>La siguiente tabla resume, para cada combinación de escenario de demanda (fila) y
plan fijo evaluado (columna, A), la mediana de las variaciones porcentuales de
throughput (Δthr) y de tiempo de sistema (Δsys) del control dinámico (B) frente a
ese plan fijo, junto con el número de corridas pareadas agregadas y la pérdida de
capacidad por ciclo inherente al plan fijo (tiempo dedicado a despeje de
intersección — verde perdido, amarillo y todo-rojo — como porcentaje del ciclo
total, un valor fijo por campaña que no depende del escenario). Valores positivos
en Δthr o Δsys favorecen al control dinámico.</p>
{comp_table}

<h2 id="rigor-estadistico">5. Rigor estadístico</h2>
{rigor_table}
<h3>Coherencia entre variación de throughput y variación de tiempo de sistema</h3>
{scatter_html}

<h2 id="analisis">6. Análisis técnico</h2>
{analysis_html}

<h2 id="apendice-de-reproducibilidad">7. Apéndice de reproducibilidad</h2>
<h3>Seeds</h3>
<p><code>{html.escape(seeds_html)}</code></p>
<h3>Comandos de reproducción</h3>
<pre><code>uv run python run_ab.py --scenario &lt;escenario&gt; --seed &lt;seed&gt; --cycle-time &lt;60|90|120&gt;
uv run python scripts/build_campaign_report.py --out results/campanas/reporte_final.html</code></pre>
<h3>MAX_PRESSURE_CONFIG (config.py)</h3>
{config_table}
<h3>Corridas faltantes</h3>
{missing_html}

</div>

<footer>Generado por <code>scripts/build_campaign_report.py</code> &mdash; reproducible a partir de
los <code>ab_report.json</code> en <code>results/campanas/&lt;campaña&gt;/&lt;escenario&gt;/seed_&lt;n&gt;/</code>.</footer>
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
