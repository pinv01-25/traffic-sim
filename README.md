# Traffic-Sim: Simulador de Tráfico Inteligente

Traffic-Sim es un sistema de simulación de tráfico urbano que utiliza SUMO (Simulation of Urban MObility) para detectar cuellos de botella en tiempo real y optimizar dinámicamente los semáforos mediante comunicación con el servicio `traffic-control`.

## Características

- **Detección automática de cuellos de botella** — analiza densidad, velocidad y longitud de colas
- **Comunicación con traffic-control** — envía datos crudos a `/ingest` y recibe optimizaciones
- **Control dinámico de semáforos** — actualiza tiempos en tiempo real vía TraCI
- **Dos modos de ejecución** — GUI (interfaz gráfica) y Headless (consola)
- **Reproducibilidad con seeds** — control del comportamiento estocástico de conductores
- **Pipeline A/B completo** — 24 gráficos + tests estadísticos + análisis de viajes incompletos
- **Experimento multi-seed** — cuantifica variabilidad entre runs para validar significancia
- **Comparador Webster (sim_C)** — baseline fixed-time óptimo teórico para benchmarking

---

## Arquitectura

```
traffic-sim/
├── run_simulation.py               # Entry point principal (CLI)
├── simulation_orchestrator.py      # Orquestador: TraCI + detección + HTTP
├── config.py                       # Configuración centralizada
├── visualization/                  # Pipeline de visualización y análisis A/B
│   ├── __init__.py                 # Exports: generate_ab_test, analyze_incomplete_trips, ...
│   ├── ab_test.py                  # compare_runs(), estadísticas, viajes incompletos
│   ├── parsers.py                  # parse_tripinfo(), parse_fcd(), parse_summary(), ...
│   ├── plots.py                    # Todas las funciones de plotting
│   └── sumo_tools.py               # Wrappers de herramientas nativas SUMO
├── scripts/                        # Análisis académicos y experimentos adicionales
│   ├── run_multiseed.py            # Experimento multi-seed (reproducibilidad)
│   ├── aggregate_seeds.py          # Agrega resultados multi-seed + violin plots
│   ├── generate_webster_timing.py  # Genera sim_C con tiempos Webster óptimos
│   ├── run_sim_c.py                # Ejecuta sim_C headless
│   ├── compare_three_runs.py       # Comparación 3-way: A vs B vs C
│   └── _sim_utils.py               # Auto-detección de directorios de simulación
├── detectors/
│   └── bottleneck_detector.py
├── controllers/
│   └── traffic_light_controller.py
├── services/
│   └── traffic_control_client.py
└── utils/
    ├── logger.py
    └── signal_utils.py
```

---

## Requisitos

- Python ≥ 3.10
- SUMO ≥ 1.8.0

### Instalación de SUMO

```bash
# Ubuntu/Debian
sudo apt-get install sumo sumo-tools sumo-gui

# macOS
brew install sumo
```

### Instalación de dependencias Python

```bash
pip install uv
uv sync
```

---

## Uso

### Ejecución básica

```bash
# Headless (por defecto)
uv run python run_simulation.py simulation.zip

# Con interfaz gráfica
uv run python run_simulation.py simulation.zip --gui

# Con semilla fija (reproducibilidad)
uv run python run_simulation.py simulation.zip --seed 42

# Limitar pasos de simulación
uv run python run_simulation.py simulation.zip --sim-steps 2000

# Optimización dinámica de semáforos (requiere traffic-control corriendo)
uv run python run_simulation.py simulation.zip --dynamic-optimization
```

### Parámetro `--seed`

`--seed N` controla la estocasticidad del comportamiento de los conductores (gap acceptance, cambio de carril, variabilidad por `sigma`). Las rutas están prefijadas en `routes.rou.xml`; el seed solo afecta cómo los conductores las recorren.

```bash
uv run python run_simulation.py sim.zip --seed 42 --sim-steps 3600
```

### Estructura del ZIP

```
simulation.zip
├── edges.edg.xml
├── nodes.nod.xml
├── routes.rou.xml
├── simulation.sumocfg
└── traffic_lights.add.xml
```

---

## Pipeline A/B

Compara dos corridas (A = control, B = optimizada con IA) y genera 24 gráficos + reportes estadísticos automáticamente.

### Uso programático

```python
from visualization import generate_ab_test

report = generate_ab_test('sim_A', 'sim_B', labels=('Control', 'IA'))
print(report['statistical_results']['percent_improvement'])
```

### Uso desde `run_simulation.py`

```bash
# Correr sim_B y comparar automáticamente con sim_A
uv run python run_simulation.py sim_B.zip --extract-dir sim_B --compare-with sim_A
```

### Auto-detección de directorios

Los scripts detectan automáticamente los directorios de simulación sin necesidad de argumentos. Buscan en el directorio de trabajo actual en este orden:

1. `sim_A/`, `sim_B/`, `sim_C/` (nombres convencionales)
2. Cualquier directorio que termine en `_A`, `_B`, `_C` (ej. `ab_run_A`, `test_B`)

Esto permite usar cualquier nombre de directorio siempre que siga la convención de sufijo.

### Gráficos generados (24 archivos)

Guardados en `sim_B/logs/visualizations/ab_test/`:

| # | Archivo | Contenido |
|---|---|---|
| 01 | `01_duration_hist_cdf.png` | Histograma + CDF de duración de viajes |
| 02 | `02_duration_boxplot.png` | Boxplot comparativo de duración |
| 03 | `03_duration_violin.png` | Violin plot de tiempos de viaje |
| 04 | `04_timeloss_violin.png` | Violin plot de tiempo perdido |
| 05 | `05_multi_metric_violin.png` | Violin de duration, timeLoss, waitingTime, departDelay |
| 06 | `06_metric_comparison_bars.png` | Barras comparativas de métricas |
| 07 | `07_duration_time_series.png` | Serie temporal de duración |
| 08 | `08_timeloss_time_series.png` | Serie temporal de tiempo perdido |
| 09–10 | `09/10_*.png` | Congestión y métricas de summary.xml |
| 11 | `11_efficiency_comparison.png` | Eficiencia temporal |
| 12 | `12_speed_distribution.png` | Distribución de velocidades |
| 13 | `13_waiting_time_analysis.png` | Análisis de tiempos de espera |
| 14 | `14_percentile_comparison.png` | Comparación por percentiles |
| 15–16 | `15/16_correlation_heatmap_A/B.png` | Matrices de correlación |
| 17 | `17_fcd_comparison.png` | Comparación FCD (Floating Car Data) |
| 18 | `18_improvement_summary.png` | Resumen de mejoras |
| 19–20 | `19/20_time_series_mean_A/B.png` | Serie temporal individual por run |
| **21** | **`21_incomplete_histogram.png`** | **Histograma: completados vs incompletos** |
| **22** | **`22_incomplete_cdf.png`** | **CDF: completados vs incompletos** |
| **23** | **`23_incomplete_boxplot.png`** | **Boxplot: completados vs incompletos** |
| **24** | **`24_incomplete_scatter.png`** | **Scatter: depart time vs time_in_network** |

Los gráficos **21–24** analizan los **viajes incompletos** de sim_A: vehículos que entraron a la red (aparecen en `fcd.xml`) pero no completaron su ruta (ausentes en `tripinfo.xml`). Cuantifican cuántos viajes son excluidos del análisis A/B y por qué, detectando posible sesgo de selección en las métricas reportadas.

### Tests estadísticos (automáticos)

- **Permutation test** — p-value para diferencia de medias
- **Bootstrap CI 95%** — intervalo de confianza para la diferencia
- **Mann-Whitney U** — test no paramétrico
- **Cohen's d** — tamaño del efecto

### Reportes

```
sim_B/logs/visualizations/ab_test/
├── ab_summary.csv          # Tabla de métricas y tests
├── ab_report.json          # Reporte completo con estadísticas
└── incomplete_trips.csv    # Detalle de viajes incompletos en sim_A
```

---

## Experimento Multi-Seed

Repite el test A/B con múltiples seeds para cuantificar si el efecto observado es reproducible o un artefacto de una sola corrida. Aborda la limitación académica de resultados basados en un único seed.

```bash
# Ejecutar 5 seeds (crea results/seed_N/sim_{A,B}/tripinfo.xml)
uv run python scripts/run_multiseed.py --seeds 42 123 456 789 1337 --sim-steps 2000

# Agregar resultados y generar violin plots
uv run python scripts/aggregate_seeds.py --results-dir results/
```

### Qué genera `aggregate_seeds.py`

- `diag_multiseed_effect.png` — violin de duration/waitingTime/timeLoss por seed (A vs B)
- `diag_multiseed_improvement.png` — scatter de mejora% por seed
- `multiseed_stats_by_seed.csv` — mean ± std por (seed, condición)
- `multiseed_improvement.csv` — Δ% A→B por seed
- Wilcoxon signed-rank: ¿la mejora es consistente entre seeds?

### Opciones de `run_multiseed.py`

```bash
uv run python scripts/run_multiseed.py \
  --seeds 42 123 456 789 1337 \
  --sim-steps 2000 \
  --out results/ \
  --dynamic-optimization   # Habilitar IA en sim_B (requiere traffic-control)
```

---

## Comparador Webster (sim_C)

Genera un tercer baseline usando la **fórmula de Webster** para calcular tiempos de semáforo óptimos teóricos a partir de los volúmenes de tráfico reales. Permite responder si la IA supera al mejor fixed-time posible, no solo al naïf.

### Fórmula de Webster

```
C₀ = (1.5·L + 5) / (1 − Y)
gᵢ = (C₀ − L) · yᵢ / Y

L  = tiempo perdido total = n_fases_verdes × 4 s
yᵢ = volumen_crítico_fase_i / 1800 (veh/h/carril)
Y  = Σ yᵢ
```

### Workflow completo

```bash
# 1. Calcular tiempos Webster y crear sim_C/ (copia de sim_A con traffic_lights.add.xml nuevo)
uv run python scripts/generate_webster_timing.py

# 2. Ejecutar sim_C headless
uv run python scripts/run_sim_c.py --sim-steps 3600

# 3. Comparación 3-way: A (fixed-time naïf) vs B (IA) vs C (Webster óptimo)
uv run python scripts/compare_three_runs.py
```

### Qué genera `compare_three_runs.py`

- `diag_three_way.png` — boxplot de duration, waitingTime, timeLoss para las 3 condiciones
- `three_way_summary.csv` — mean, std, Δ% vs A, p-value Mann-Whitney (A-B, A-C, B-C)

---

## Uso Programático

```python
from visualization import generate_ab_test, analyze_incomplete_trips, quick_compare
from visualization.parsers import parse_tripinfo
from simulation_orchestrator import SimulationOrchestrator

# Pipeline A/B completo (24 gráficos + reportes)
report = generate_ab_test('sim_A', 'sim_B', labels=('Control', 'IA'))

# Solo estadísticas rápidas (sin gráficos)
stats = quick_compare('sim_A', 'sim_B')
print(f"Mejora: {stats['percent_improvement']:.1f}%")

# Solo análisis de viajes incompletos
df_trip = parse_tripinfo('sim_A/logs/sumo_output/tripinfo.xml')
analyze_incomplete_trips('sim_A', df_trip, out_dir='out/', label='A')

# SimulationOrchestrator con seed fijo
orch = SimulationOrchestrator('sim_A', sim_steps=2000, seed=42)
if orch.setup_simulation():
    orch.run_simulation()
```

---

## Configuración

```python
# config.py
BOTTLENECK_CONFIG = {
    "density_threshold": 50.0,    # vehículos/km
    "speed_threshold": 5.0,       # m/s (18 km/h)
    "queue_length_threshold": 10,
    "detection_interval": 30,     # pasos entre detecciones
}

TRAFFIC_CONTROL_CONFIG = {
    "base_url": "http://localhost:8003",
    "timeout": 30,
}
```

Variables de entorno (`.env`):

```env
TRAFFIC_CONTROL_URL=http://localhost:8003
LOG_LEVEL=INFO
```

---

## Troubleshooting

**SUMO no encontrado**
```bash
which sumo && sumo --version
```

**traffic-control no disponible** — los errores HTTP son esperados al correr localmente sin Docker. La simulación continúa en modo fixed-time sin interrupciones.

**FCD muy grande** — `parse_fcd()` acepta `sample_rate=N` para leer solo cada N timesteps. El análisis de viajes incompletos usa streaming sin cargar todo en memoria.

**Directorio de simulación no encontrado** — los scripts buscan `sim_A/` y `sim_B/` en el directorio actual. Si usas nombres distintos, asegúrate de que terminen en `_A` y `_B` (ej. `mi_exp_A`, `mi_exp_B`).
