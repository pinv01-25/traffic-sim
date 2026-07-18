#!/usr/bin/env python3
"""Re-clasifica los ab_report.json existentes con el veredicto insesgado.

Los reportes generados antes de este cambio no guardan `system_time`
(la métrica primaria insesgada por supervivencia; ver
visualization/ab_test.py:system_time_statistics) — solo
`percent_improvement`, la media de duration sesgada por supervivencia, y
`throughput` (que sí trae la guardia dura). Los directorios sim_A/sim_B de
esas corridas (logs/sumo_output/summary.xml crudo) no se conservan en la
campaña, así que no se puede recalcular `system_time` de forma exacta.

Sin embargo cada ab_report.json SÍ guarda `summary_statistics.<label>.
running.mean` y `data_info.<label>.summary_rows`, con lo que se puede
*aproximar* `system_total_time ≈ running.mean * summary_rows` (asume paso
de summary de 1s, el caso normal en esta campaña — ver el comentario en
system_time_statistics). Esa aproximación (`approx=True`) se usa para
recalcular el veredicto con compute_verdict(); si falta esa información
(reportes muy viejos sin summary_statistics) se reclasifica solo con la
guardia de throughput ya persistida y se marca `needs_resim=True`.

Uso: python -m scripts.rescore_ab   (desde la raíz del repo)
Reescribe cada ab_report.json con `statistical_tests.verdict` recalculado
(y `statistical_tests.system_time_approx` cuando pudo aproximarse) e
imprime una tabla resumen.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visualization.ab_test import compute_verdict  # noqa: E402

RESULTS = Path('results/escenarios')


def approx_system_time(report: dict):
    """Aproxima system_time_statistics() a partir de summary_statistics ya
    guardado en el ab_report.json (sin necesidad del summary.xml crudo)."""
    labels = report.get('labels') or ['Baseline', 'Optimized']
    ss = report.get('summary_statistics') or {}
    di = report.get('data_info') or {}
    try:
        run_a, run_b = ss[labels[0]], ss[labels[1]]
        rows_a = di[labels[0]]['summary_rows']
        rows_b = di[labels[1]]['summary_rows']
        time_a = run_a['running']['mean'] * rows_a
        time_b = run_b['running']['mean'] * rows_b
    except (KeyError, TypeError):
        return None
    if not rows_a or not rows_b:
        return None

    ins_a = (run_a.get('inserted') or {}).get('max')
    ins_b = (run_b.get('inserted') or {}).get('max')

    result = {
        'system_time_a': time_a, 'system_time_b': time_b,
        'inserted_a': ins_a, 'inserted_b': ins_b,
        'approx': True,
    }
    result['system_time_improvement_pct'] = (
        (time_a - time_b) / time_a * 100 if time_a else float('nan')
    )
    if ins_a and time_a is not None:
        mtpv_a = time_a / max(ins_a, 1)
        result['mean_time_per_vehicle_a'] = mtpv_a
    if ins_b and time_b is not None:
        mtpv_b = time_b / max(ins_b, 1)
        result['mean_time_per_vehicle_b'] = mtpv_b
    return result


def rescore(json_path: Path):
    report = json.loads(json_path.read_text())
    st = report.get('statistical_tests') or {}
    approx = approx_system_time(report)
    needs_resim = approx is None

    st_for_verdict = dict(st)
    if approx:
        st_for_verdict['system_time'] = approx
    verdict = compute_verdict(st_for_verdict)

    return report, st, verdict, approx, needs_resim


def _f(v, spec='+.1f'):
    return format(v, spec) if isinstance(v, (int, float)) and v == v else '—'


def main():
    rows = []
    for json_path in sorted(RESULTS.glob('*/seed_*/ab_report.json')):
        report, st, verdict, approx, needs_resim = rescore(json_path)
        thr = st.get('throughput') or {}
        rows.append({
            'scenario': json_path.parent.parent.name,
            'seed': json_path.parent.name.replace('seed_', ''),
            'imp_pct': st.get('percent_improvement'),
            'system_time_pct': (approx or {}).get('system_time_improvement_pct'),
            'thr_pct': thr.get('throughput_change_pct'),
            'verdict': verdict['verdict'],
            'reason': verdict['reason'],
            'needs_resim': needs_resim,
        })

        # Persistir el re-score en el propio ab_report.json.
        report.setdefault('statistical_tests', {})['verdict'] = verdict
        if approx:
            report['statistical_tests']['system_time_approx'] = approx
        json_path.write_text(json.dumps(report, indent=2))

    if not rows:
        print(f'No se encontraron ab_report.json bajo {RESULTS}/')
        return

    col = '{:<18}{:<6}{:>16}{:>14}{:>13}  {}'
    print(col.format('escenario', 'seed', 'imp% (sesgado)', 'system-time%', 'throughput%', 'veredicto'))
    print('-' * 90)
    for r in rows:
        flag = ' [needs_resim: sin summary_statistics]' if r['needs_resim'] else ' (aprox. system-time)'
        print(col.format(
            r['scenario'], r['seed'],
            _f(r['imp_pct']), _f(r['system_time_pct']), _f(r['thr_pct']),
            r['verdict'] + flag,
        ))

    n_resim = sum(r['needs_resim'] for r in rows)
    print(f'\n{len(rows)} corridas re-clasificadas. {n_resim} necesitan re-simulación '
          '(sin datos de summary_statistics para aproximar system_time).')


if __name__ == '__main__':
    main()
