"""Parchea in-situ los ab_report.html ya generados que no incluyen la nota de
sesgo de supervivencia cuando los conteos de viajes completados A/B difieren
en más de 10%.

Los XML crudos de results/escenarios_v6/*/seed_*/ fueron borrados, así que no
se pueden regenerar los reportes desde cero: este script lee el ab_report.json
vecino de cada ab_report.html (statistical_tests.throughput con
completed_a/completed_b/throughput_change_pct) e inyecta la misma nota de
advertencia que ahora escribe visualization.ab_test._write_html_report,
justo antes de la tabla "Métricas clave por viaje".

Idempotente: si la nota ya está presente en el HTML, no se duplica.

Uso:
    uv run python scripts/patch_report_context.py [directorio_raiz]

Por defecto recorre results/escenarios_v6.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MARKER = 'class="warn"'
TABLE_HEADING = '<h2>Métricas clave por viaje</h2>'
THRESHOLD_PCT = 10.0


def _build_note(labels, completed_a, completed_b, throughput_change_pct) -> str:
    label_a = labels[0] if labels else 'A'
    label_b = labels[1] if labels and len(labels) > 1 else 'B'
    pct = f'{throughput_change_pct:+.1f}' if isinstance(throughput_change_pct, (int, float)) else '—'
    return (
        f'<p class="warn">⚠️ Las métricas por viaje solo incluyen viajes completados: '
        f'{label_a} completó {completed_a} y {label_b} completó {completed_b} '
        f'(Δ throughput {pct}%). '
        f'El lado que colapsa promedia solo los viajes rápidos que escaparon.</p>\n'
    )


def _needs_patch(json_path: Path) -> tuple[bool, str]:
    """Return (should_patch, note_html) for a given ab_report.json."""
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False, ''

    throughput = data.get('statistical_tests', {}).get('throughput') or {}
    completed_a = throughput.get('completed_a')
    completed_b = throughput.get('completed_b')
    if completed_a is None or completed_b is None or max(completed_a, completed_b) <= 0:
        return False, ''

    diff_pct = abs(completed_a - completed_b) / max(completed_a, completed_b) * 100
    if diff_pct <= THRESHOLD_PCT:
        return False, ''

    note = _build_note(
        data.get('labels'), completed_a, completed_b,
        throughput.get('throughput_change_pct'),
    )
    return True, note


def patch_report(html_path: Path) -> str:
    """Patch a single ab_report.html in place. Returns a status string:
    'patched', 'already_patched', 'not_applicable', or 'no_json'."""
    json_path = html_path.with_name('ab_report.json')
    if not json_path.exists():
        return 'no_json'

    should_patch, note = _needs_patch(json_path)
    if not should_patch:
        return 'not_applicable'

    html = html_path.read_text(encoding='utf-8')
    if MARKER in html:
        return 'already_patched'

    if TABLE_HEADING not in html:
        return 'not_applicable'

    patched_html = html.replace(TABLE_HEADING, TABLE_HEADING + '\n' + note, 1)
    html_path.write_text(patched_html, encoding='utf-8')
    return 'patched'


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path('results/escenarios_v6')
    if not root.exists():
        print(f'No existe: {root}', file=sys.stderr)
        return 1

    counts = {'patched': 0, 'already_patched': 0, 'not_applicable': 0, 'no_json': 0}
    for html_path in sorted(root.rglob('ab_report.html')):
        status = patch_report(html_path)
        counts[status] += 1
        print(f'{status:16s} {html_path}')

    print(
        f'\nTotal: {sum(counts.values())} reportes | '
        f'parcheados: {counts["patched"]} | '
        f'ya parcheados: {counts["already_patched"]} | '
        f'no aplica (conteos comparables): {counts["not_applicable"]} | '
        f'sin json: {counts["no_json"]}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
