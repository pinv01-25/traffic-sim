#!/usr/bin/env python3
"""Genera results/index.html: resumen consolidado de la campaña A/B.

Layout esperado: results/escenarios/<escenario>/seed_<n>/ab_report.json
(salida de run_ab.py --output-dir). El veredicto por corrida prioriza el
throughput (viajes completados): a saturación, la media de duraciones solo
cuenta a los que terminaron — una media que "mejora" con menos viajes
completados es sesgo de supervivencia, no una mejora.
"""
import json
from datetime import datetime
from pathlib import Path

RESULTS = Path('results')

#: Umbral de no-daño en throughput (%): dentro de ±1% se considera igual.
THR_TIE = 1.0
#: Caída de throughput que dispara la marca de sesgo de supervivencia.
THR_SURVIVOR = -5.0


def load_report(seed_dir: Path):
    p = seed_dir / 'ab_report.json'
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    st = r.get('statistical_tests', {})
    paired = st.get('paired', {}) or {}
    thr = st.get('throughput', {}) or {}
    a, b = thr.get('completed_a'), thr.get('completed_b')
    thr_pct = (b - a) / a * 100 if a and b else None
    return {
        'seed': seed_dir.name.replace('seed_', ''),
        'thr_a': a, 'thr_b': b, 'thr_pct': thr_pct,
        'n_paired': paired.get('n_paired'),
        'mean_delta': paired.get('mean_delta'),
        'median_delta': paired.get('median_delta'),
        'wilcoxon_p': paired.get('wilcoxon_pvalue'),
        'html': str((seed_dir / 'ab_report.html').relative_to(RESULTS)),
    }


def classify(r):
    """'win' / 'tie' / 'loss' por corrida, con throughput como métrica primaria."""
    tp, p = r['thr_pct'], r['wilcoxon_p']
    if tp is None:
        return 'tie'
    if tp > THR_TIE:
        return 'win'
    if tp < -THR_TIE:
        return 'loss'
    # Throughput igual: deciden las duraciones pareadas significativas
    if isinstance(p, float) and p < 0.05 and (r['mean_delta'] or 0) < 0:
        return 'win'
    if isinstance(p, float) and p < 0.05 and (r['mean_delta'] or 0) > 0:
        return 'loss'
    return 'tie'


def fmt(v, spec='.1f'):
    return format(v, spec) if isinstance(v, (int, float)) and v == v else '—'


HEAD = ("<tr><th>Corrida</th><th>Viajes A → B</th><th>Δ throughput</th>"
        "<th>n pareado</th><th>Δ media par. (s)</th><th>Δ mediana par. (s)</th>"
        "<th>p (Wilcoxon)</th><th>Veredicto</th></tr>")

BADGE = {'win': ('✅ mejora', 'pos'), 'loss': ('❌ empeora', 'neg'), 'tie': ('≈ sin efecto', '')}


def table_rows(rows):
    out = []
    for r in rows:
        cls = classify(r)
        badge, css = BADGE[cls]
        survivor = (r['thr_pct'] is not None and r['thr_pct'] < THR_SURVIVOR
                    and (r['mean_delta'] or 0) < 0)
        if survivor:
            badge += ' ⚠️ sesgo de supervivencia'
        out.append(
            f"<tr><td><a href='{r['html']}'>seed {r['seed']}</a></td>"
            f"<td>{r['thr_a'] or '—'} → {r['thr_b'] or '—'}</td>"
            f"<td class='{'pos' if (r['thr_pct'] or 0) > 0 else 'neg'}'>{fmt(r['thr_pct'], '+.1f')}%</td>"
            f"<td>{r['n_paired'] or '—'}</td><td>{fmt(r['mean_delta'], '+.1f')}</td>"
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
    raise SystemExit('No hay ab_report.json en results/escenarios/')

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
