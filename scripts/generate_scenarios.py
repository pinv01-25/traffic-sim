#!/usr/bin/env python3
"""
Genera escenarios de demanda deterministas a partir de un ZIP de simulación.

Cada escenario modifica solo routes.rou.xml (misma red, mismos semáforos):
  - demanda_baja:     35% de los vehículos, misma ventana (flujo libre)
  - demanda_media:    65% de los vehículos, misma ventana
  - demanda_alta:     100% (la demanda original, incluida por completitud)
  - demanda_saturada: 140% (todos + 40% duplicados intercalados con rutas iguales)
  - hora_pico:        100% pero con salidas concentradas en el centro de la
                      ventana (perfil triangular: ~2/3 de la flota en el tercio
                      central), imitando una hora punta

Todo es determinista: sin RNG, selección por índice y offsets fijos, así el
A/B por escenario mantiene condiciones idénticas entre corridas.

Uso:
    uv run python scripts/generate_scenarios.py sim.zip --out scenarios/
"""
import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

SCENARIOS = {
    'demanda_baja': {'keep': 0.35},
    'demanda_media': {'keep': 0.65},
    'demanda_alta': {'keep': 1.0},
    'demanda_saturada': {'keep': 1.0, 'extra': 0.4},
    'hora_pico': {'keep': 1.0, 'peak': True},
}


def _peak_map(t: float, t_max: float) -> float:
    """Remapea un depart uniforme a un perfil triangular centrado en t_max/2.

    La función es el inverso de la CDF de una distribución triangular, así
    que salidas uniformes se transforman en una "campana" de hora pico.
    """
    u = min(max(t / t_max, 0.0), 1.0)
    if u <= 0.5:
        new_u = (u / 2.0) ** 0.5          # rama ascendente
    else:
        new_u = 1.0 - ((1.0 - u) / 2.0) ** 0.5
    return round(new_u * t_max, 2)


def build_scenario(routes_root: ET.Element, name: str, cfg: dict) -> ET.Element:
    """Construye un nuevo <routes> según la configuración del escenario."""
    out = ET.Element('routes', routes_root.attrib)

    # vTypes tal cual
    for vtype in routes_root.findall('vType'):
        out.append(vtype)

    # Pares (route, vehicle) en orden de aparición
    routes_by_id = {r.get('id'): r for r in routes_root.findall('route')}
    vehicles = sorted(routes_root.findall('vehicle'), key=lambda v: float(v.get('depart', 0)))
    n = len(vehicles)
    departs = [float(v.get('depart', 0)) for v in vehicles]
    t_max = max(departs) if departs else 0.0

    keep = cfg.get('keep', 1.0)
    n_keep = int(round(n * keep))
    # Selección determinista uniforme: índice i*n/n_keep
    kept_idx = sorted({int(i * n / n_keep) for i in range(n_keep)}) if n_keep else []

    pairs = []
    for i in kept_idx:
        veh = vehicles[i]
        depart = float(veh.get('depart', 0))
        if cfg.get('peak'):
            depart = _peak_map(depart, t_max)
        pairs.append((veh, depart, ''))

    # Duplicados para saturación: mismo vehículo, offset fijo de medio headway
    if cfg.get('extra'):
        n_extra = int(round(n * cfg['extra']))
        extra_idx = sorted({int(i * n / n_extra) for i in range(n_extra)}) if n_extra else []
        headway = t_max / n if n else 1.0
        for i in extra_idx:
            veh = vehicles[i]
            depart = float(veh.get('depart', 0)) + headway / 2.0
            pairs.append((veh, depart, '_dup'))

    pairs.sort(key=lambda p: p[1])

    for veh, depart, suffix in pairs:
        route = routes_by_id.get(veh.get('route'))
        vid = veh.get('id') + suffix
        rid = veh.get('route') + suffix

        r = ET.SubElement(out, 'route', dict(route.attrib))
        r.set('id', rid)
        v = ET.SubElement(out, 'vehicle', dict(veh.attrib))
        v.set('id', vid)
        v.set('route', rid)
        v.set('depart', f'{depart:.2f}')

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description='Genera escenarios de demanda a partir de un ZIP de simulación')
    parser.add_argument('zip_file', help='ZIP base con routes.rou.xml y demás archivos SUMO')
    parser.add_argument('--out', type=Path, default=Path('scenarios'), help='Directorio de salida (default: scenarios/)')
    parser.add_argument('--only', nargs='+', choices=sorted(SCENARIOS), help='Generar solo estos escenarios')
    args = parser.parse_args()

    names = args.only or list(SCENARIOS)
    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='scenarios_') as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(args.zip_file) as zf:
            zf.extractall(tmp_path)

        routes_file = tmp_path / 'routes.rou.xml'
        tree = ET.parse(routes_file)

        base_files = [p for p in tmp_path.iterdir() if p.name != 'routes.rou.xml']

        for name in names:
            new_root = build_scenario(tree.getroot(), name, SCENARIOS[name])
            n_veh = len(new_root.findall('vehicle'))

            out_zip = args.out / f'{name}.zip'
            with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                for p in base_files:
                    zf.write(p, p.name)
                zf.writestr('routes.rou.xml', ET.tostring(new_root, encoding='unicode', xml_declaration=True))

            print(f'{name}: {n_veh} vehículos → {out_zip}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
