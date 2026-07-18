#!/usr/bin/env python3
"""Genera results/index.html: resumen consolidado de la campaña A/B.

Layout esperado: escenarios/<escenario>/seed_<n>/ab_report.json
(salida de run_ab.py --output-dir). El veredicto por corrida usa
`compute_verdict` (visualization/ab_test.py) — la misma función única que
usan el JSON, el CSV y el HTML de cada corrida — para que la campaña nunca
se contradiga con el reporte individual. `compute_verdict` ya prioriza el
throughput como guardia dura (colapso ≠ mejora) y usa el tiempo de sistema
insesgado cuando está disponible.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visualization.ab_test import compute_verdict  # noqa: E402

import os
RESULTS = Path(os.environ.get('CAMPAIGN_RESULTS','results'))


def load_report(seed_dir: Path):
    p = seed_dir / 'ab_report.json'
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    st = r.get('statistical_tests', {})
    paired = st.get('paired', {}) or {}
    thr = st.get('throughput', {}) or {}
    st_time = st.get('system_time') or st.get('system_time_approx') or {}
    a, b = thr.get('completed_a'), thr.get('completed_b')
    thr_pct = thr.get('throughput_change_pct')
    verdict = st.get('verdict') or compute_verdict(st)
    return {
        'seed': seed_dir.name.replace('seed_', ''),
        'thr_a': a, 'thr_b': b, 'thr_pct': thr_pct,
        'imp_pct': st.get('percent_improvement'),
        'system_time_pct': st_time.get('system_time_improvement_pct'),
        'n_paired': paired.get('n_paired'),
        'mean_delta': paired.get('mean_delta'),
        'median_delta': paired.get('median_delta'),
        'wilcoxon_p': paired.get('wilcoxon_pvalue'),
        'verdict': verdict.get('verdict'),
        'reason': verdict.get('reason'),
        'html': str((seed_dir / 'ab_report.html').relative_to(RESULTS)),
    }


def classify(r):
    """'win' / 'tie' / 'loss' agregado para el banner de campaña, derivado
    del veredicto insesgado de compute_verdict (5 categorías -> 3)."""
    return {
        'improvement': 'win',
        'misleading_regression': 'win',   # mejora real; la media sesgada mentía
        'regression': 'loss',
        'misleading_improvement': 'loss',  # empeora en realidad; la media sesgada mentía
        'tie': 'tie',
    }[r['verdict']]


def fmt(v, spec='.1f'):
    return format(v, spec) if isinstance(v, (int, float)) and v == v else '—'


HEAD = ("<tr><th>Corrida</th><th>Viajes A → B</th><th>Δ throughput</th>"
        "<th>imp% (sesgado)</th><th>Δ tiempo sistema</th>"
        "<th>n pareado</th><th>Δ mediana par. (s)</th>"
        "<th>p (Wilcoxon)</th><th>Veredicto</th></tr>")

BADGE = {
    'improvement': ('✅ mejora', 'pos'),
    'misleading_regression': ('✅ mejora (media sesgada mentía)', 'pos'),
    'regression': ('❌ empeora', 'neg'),
    'misleading_improvement': ('❌ empeora ⚠️ sesgo de supervivencia', 'neg'),
    'tie': ('≈ sin efecto', ''),
}


def table_rows(rows):
    out = []
    for r in rows:
        badge, css = BADGE[r['verdict']]
        out.append(
            f"<tr><td><a href='{r['html']}' title='{r['reason']}'>seed {r['seed']}</a></td>"
            f"<td>{r['thr_a'] or '—'} → {r['thr_b'] or '—'}</td>"
            f"<td class='{'pos' if (r['thr_pct'] or 0) > 0 else 'neg'}'>{fmt(r['thr_pct'], '+.1f')}%</td>"
            f"<td>{fmt(r['imp_pct'], '+.1f')}%</td>"
            f"<td>{fmt(r['system_time_pct'], '+.1f')}%</td>"
            f"<td>{r['n_paired'] or '—'}</td>"
            f"<td>{fmt(r['median_delta'], '+.1f')}</td><td>{fmt(r['wilcoxon_p'], '.2g')}</td>"
            f"<td class='{css}'>{badge}</td></tr>"
        )
    return ''.join(out)


sections, all_rows = [], []
for sc_dir in sorted((RESULTS / 'escenarios').iterdir()):
    if not sc_dir.is_dir():
        continue
    rows = [r for r in (load_report(d) for d in sorted(sc_dir.glob('seed_*'))) if r]
    if rows:
        all_rows.extend(rows)
        sections.append(f"<h3>{sc_dir.name}</h3><table>{HEAD}{table_rows(rows)}</table>")

if not all_rows:
    raise SystemExit('No hay ab_report.json en escenarios/')

tally = {'win': 0, 'loss': 0, 'tie': 0}
for r in all_rows:
    tally[classify(r)] += 1

if tally['loss'] == 0 and tally['win'] > 0:
    cls = 'green'
    verdict = (f"La optimización mejora o iguala el baseline en las {len(all_rows)} corridas "
               f"({tally['win']} mejoras, {tally['tie']} sin efecto)")
elif tally['win'] == 0:
    cls = 'red'
    verdict = f"La optimización NO mejora el baseline ({tally['loss']} empeoran, {tally['tie']} sin efecto)"
else:
    cls = 'gray'
    verdict = (f"Resultado mixto: {tally['win']} corridas mejoran, {tally['loss']} empeoran, "
               f"{tally['tie']} sin efecto claro — ver análisis por régimen abajo")

html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Campaña A/B — control jerárquico</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1150px; color: #222; padding: 0 1rem; }}
 .banner {{ padding: 1rem 1.4rem; border-radius: 8px; margin: 1rem 0; }}
 .banner.green {{ background: #e6f6ec; border: 1px solid #34a853; }}
 .banner.red {{ background: #fdecea; border: 1px solid #ea4335; }}
 .banner.gray {{ background: #f1f3f4; border: 1px solid #9aa0a6; }}
 table {{ border-collapse: collapse; width: 100%; font-size: .88rem; margin: .6rem 0 1.4rem; }}
 th, td {{ border: 1px solid #ddd; padding: .4rem .55rem; text-align: left; }}
 th {{ background: #f6f8fa; }}
 td.pos {{ color: #188038; font-weight: 600; }} td.neg {{ color: #c5221f; font-weight: 600; }}
</style></head><body>
<h1>Campaña A/B — control jerárquico (ciclo por PSO + reparto local)</h1>
<p>Generado el {datetime.now():%Y-%m-%d %H:%M}. En cada corrida, A (fixed-time) y B (control
jerárquico: traffic-sync decide el largo de ciclo por cluster vía fuzzy+PSO y traffic-sim lo
aplica localmente) usan red, demanda, semilla y flags idénticos — la única diferencia es la
optimización. Métrica primaria: <b>throughput</b> (viajes completados); las duraciones pareadas
(mismo vehículo en A y B) complementan. Cada fila enlaza al reporte completo
(32 gráficos + tests + CSV/JSON).</p>
<div class="banner {cls}"><h2>{verdict}</h2></div>

<h2>Escenarios</h2>
<p>corredor: Saturio_Rios a demanda 2x, transversales al 20% (2.010 veh, 2.500 steps) ·
demanda_baja: 1.750 veh · demanda_media: 3.250 · demanda_alta: 5.000 ·
demanda_saturada: 7.000 · hora_pico: 5.000 con perfil de hora punta (4.500 steps).
En los regímenes de alta demanda el resultado depende de la formación de gridlock
(sensible al timing de las fases); por eso esos escenarios se corren con varias semillas.</p>
{''.join(sections)}
</body></html>
"""
(RESULTS / 'index.html').write_text(html, encoding='utf-8')
print(f"OK: results/index.html — {len(all_rows)} corridas: "
      f"{tally['win']} mejoran, {tally['loss']} empeoran, {tally['tie']} sin efecto")
