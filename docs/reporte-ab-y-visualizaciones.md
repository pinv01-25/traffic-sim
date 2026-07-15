# Reporte: Test A/B, visualizaciones y validación de equivalencia

**Fecha:** 2026-07-15 · **Alcance:** pipeline A/B de traffic-sim (`run_ab.py`, `visualization/`, `scripts/`)

Este reporte responde tres preguntas:

1. ¿Qué compara exactamente el test A/B y con qué herramientas se puede *explicar* el resultado?
2. ¿Son la simulación A y la B **exactamente iguales** en condiciones e inputs, salvo por la optimización?
3. ¿Qué riesgos de sesgo existen y cómo cerrarlos?

---

## 1. Diseño del experimento

| Elemento | A (Baseline) | B (Optimizado) |
|----------|--------------|----------------|
| Red vial (`network.net.xml`) | idéntica | idéntica |
| Demanda (`routes.rou.xml`) | idéntica | idéntica |
| Semáforos iniciales | programa default de netconvert | mismo programa default **+ intervención** |
| Intervención | ninguna | `--green-time G` (estático) o `--dynamic-optimization` (pipeline traffic-control → traffic-sync) |
| Motor | mismo SUMO, mismos flags | mismo SUMO, mismos flags |
| Seed | mismo (default de SUMO) | mismo |

Variantes disponibles:

- **A/B simple** (`run_ab.py`): una corrida por condición, mismo seed → determinista.
- **Multi-seed** (`scripts/run_multiseed.py` + `aggregate_seeds.py`): repite A y B con N seeds (default 42, 123, 456, 789, 1337) para separar el efecto de la optimización del ruido estocástico de los conductores. Test pareado (Wilcoxon por seed).
- **Three-way** (`scripts/generate_webster_timing.py` + `run_sim_c.py` + `compare_three_runs.py`): agrega **sim_C (Webster)**, el fixed-time óptimo teórico calculado desde los volúmenes reales — responde "¿la IA supera al *mejor* fixed-time posible, no solo al naïf?".

## 2. Validación de equivalencia A vs B (verificado sobre el repo)

### 2.1 Inputs — VERIFICADO IDÉNTICOS ✅

Comparación byte a byte de `sim_A/` vs `sim_B/` (extraídos del mismo `sim.zip`):

| Archivo | Resultado |
|---------|-----------|
| `edges.edg.xml` | idéntico |
| `nodes.nod.xml` | idéntico |
| `routes.rou.xml` | idéntico |
| `simulation.sumocfg` | idéntico |
| `network.net.xml` | idéntico |
| `traffic_lights.add.xml` | idéntico (vacío — TLS los genera netconvert igual en ambos) |
| `simulation_metadata.json` | idéntico |

### 2.2 Demanda — DETERMINISTA ✅

`routes.rou.xml` define **5 000 vehículos con `depart` y ruta fijos** (no hay `<flow>` ni generación aleatoria). La secuencia de llegadas es exactamente la misma en A y B. La única estocasticidad es el comportamiento del conductor (`sigma=0.5`: aceleración imperfecta, gap acceptance), controlada por el seed de SUMO.

### 2.3 Motor y flags — IDÉNTICOS POR CONSTRUCCIÓN ✅ (en `run_ab.py`)

Ambos runs pasan por el mismo código (`run_simulation.py` → `SimulationOrchestrator`), con los mismos flags:
`--no-step-log`, `--time-to-teleport -1` (sin teletransporte que "resuelva" congestión artificialmente), mismos outputs (`tripinfo`, `summary`, `fcd`), mismo `sim_steps`, mismo `cycle-time`. La única diferencia de línea de comandos es el flag de optimización del run B (`run_ab.py:160-165`).

### 2.4 Seed — IGUAL PERO IMPLÍCITO ⚠️

`run_ab.py` **no pasa `--seed`**: ambos runs usan el seed default fijo de SUMO, así que son iguales — pero por accidente, no por diseño. Recomendación: exponer `--seed` en `run_ab.py`, pasarlo a ambos runs y registrarlo en `ab_report.json`. (En multiseed sí se pasa explícito y de forma pareada. ✅)

### 2.5 Aislamiento del baseline — **NO GARANTIZADO** ❌ (hallazgo crítico)

El baseline A también corre el detector de cuellos de botella y **envía datos a traffic-control `/ingest`**. Ese endpoint ejecuta el pipeline completo de optimización y devuelve las optimizaciones, y el worker legacy del orquestador **las aplica** (`simulation_orchestrator.py:252-259`). Como los IDs de semáforo de esta red son numéricos (nodos OSM), el mapeo coincide y la aplicación es efectiva.

**Con traffic-control corriendo, el "baseline" también se optimiza** (y en pasos no deterministas, porque el worker es asíncrono). Detalle completo en `reporte-revision-repo.md` (C1).

**Cómo garantizar el aislamiento hoy:** correr el run A con traffic-control apagado, o aplicar el fix propuesto (modo legacy nunca aplica optimizaciones / flag `--baseline`).

### 2.6 ¿La medición perturba la simulación? — NO ✅

El detector y el calculador de métricas solo hacen **lecturas** TraCI (`getIDList`, `getSpeed`, `getLanePosition`, …); ninguna lectura altera el estado de SUMO. En modo dinámico, la llamada HTTP es síncrona y bloquea el reloj de pared, pero **el tiempo de simulación queda congelado durante la llamada** — no introduce sesgo en las métricas de simulación. El diff actual además eliminó el `time.sleep(0.05)` del loop headless (solo afectaba wall-time).

### 2.7 Prueba de no-sesgo recomendada: test A/A

`run_multiseed.py` **sin** `--dynamic-optimization` corre sim_B como fixed-time puro → A y B deberían ser estadísticamente indistinguibles (mismos inputs, mismo seed ⇒ diferencia exactamente 0; entre seeds, distribuciones solapadas). Esta es la validación empírica más fuerte de que el pipeline no tiene sesgos ocultos: **si el A/A da diferencia distinta de 0 con el mismo seed, hay contaminación** (ver 2.5). Vale la pena correrlo y archivar el resultado como evidencia.

### 2.8 Sesgo de supervivencia — PARCIALMENTE CUBIERTO ⚠️

`tripinfo.xml` solo contiene **viajes completados**. Si B mejora el throughput, su población de completados incluye viajes lentos que en A nunca terminaron → comparar solo duraciones de completados puede subestimar (o hasta invertir) el efecto. El pipeline ya lo diagnostica (gráficos 21–24 + `incomplete_trips.csv`), pero **solo para el run A**. Recomendaciones:

1. Correr `analyze_incomplete_trips` también para B y comparar tasas de completitud.
2. Reportar siempre el **throughput** (n de `tripinfo` de cada run, ya está en `data_info` del JSON) junto a las medias.
3. Usar `--sim-steps` ≥ 3600 (duración completa) o `--tripinfo-output.write-unfinished` para reducir el censoring.

## 3. Catálogo de visualizaciones — múltiples maneras de explicar el mismo resultado

Todas se generan con `generate_ab_test()` en `sim_B/logs/visualizations/ab_test/`. Agrupadas por la pregunta que responden:

### "¿Cambió la distribución de tiempos de viaje?" (evidencia distribucional)

| Gráfico | Qué muestra | Cómo explicarlo |
|---------|-------------|-----------------|
| `01_duration_hist_cdf.png` | Histograma + CDF superpuestos | "La curva de B está desplazada a la izquierda: más viajes cortos" |
| `02_duration_boxplot.png` | Boxplot A vs B | Mediana, cuartiles y outliers de un vistazo |
| `03_duration_violin.png` | Violín de duración | Forma completa de la distribución (bimodalidades, colas) |
| `04_timeloss_violin.png` | Violín de tiempo perdido | El timeLoss aísla el efecto del semáforo mejor que la duración total |
| `05_multi_metric_violin.png` | Violines de duration/timeLoss/waitingTime/departDelay | Panorama en una lámina |
| `12_speed_distribution.png` | Distribución de velocidades efectivas | "Los vehículos circulan más rápido en B" |
| `14_percentile_comparison.png` | Comparación por percentiles (p50, p75, p90, p95…) | Clave para equidad: ¿mejora también el peor 10 % de los viajes o solo el promedio? |

### "¿Cuándo ocurre la mejora?" (evidencia temporal)

| Gráfico | Qué muestra |
|---------|-------------|
| `07_duration_time_series.png` | Duración media por bin de tiempo de salida, A vs B |
| `08_timeloss_time_series.png` | timeLoss en el tiempo — muestra si la optimización ayuda más en horas pico |
| `09_summary_comparison.png` | running/halting/entered/left a lo largo de la sim ⚠️ (depende del fix de `parse_summary`, ver reporte de revisión C3) |
| `10_congestion_timeline.png` | Ratio de congestión (halting/running) en el tiempo ⚠️ (ídem) |
| `19/20_time_series_mean_*.png` | Serie individual por run |

### "¿Cuánto y con qué confianza?" (evidencia estadística)

| Elemento | Qué aporta |
|----------|-----------|
| `06_metric_comparison_bars.png` | Barras con error (media ± σ) por métrica |
| `18_improvement_summary.png` | % de mejora por métrica — la lámina "ejecutiva" |
| Test de permutación (5 000 iter) | p-value sin supuestos de normalidad: "¿la diferencia de medias puede ser azar?" |
| Bootstrap CI 95 % | Rango plausible de la mejora en segundos — más informativo que el p-value |
| Mann-Whitney U | Confirma con enfoque de rangos (robusto a outliers) |
| Cohen's d | Magnitud del efecto (pequeño/mediano/grande) independiente del n |
| `ab_summary.csv` / `ab_report.json` | Todo lo anterior en formato tabular/programático |

### "¿La comparación es válida?" (diagnóstico)

| Gráfico | Qué muestra |
|---------|-------------|
| `11_efficiency_comparison.png` | Eficiencia temporal (1 − timeLoss/duration) |
| `13_waiting_time_analysis.png` | Análisis multi-panel del tiempo de espera |
| `15/16_correlation_heatmap_*.png` | Correlaciones entre métricas por run (sanity check de coherencia) |
| `17_fcd_comparison.png` | Velocidades desde FCD, incl. por edge — localiza *dónde* mejora |
| `21–24_incomplete_*.png` + `incomplete_trips.csv` | Viajes censurados (sesgo de supervivencia) — hoy solo para A ⚠️ |

### Fuera del A/B simple

- **Single-run** (`generate_visualizations()`): histogramas tripinfo, scatter de departDelay, timeline de summary, heatmap espacial de velocidad FCD, mapa de semáforos de la red. Útiles para explicar *una* simulación aislada.
- **Multi-seed** (`aggregate_seeds.py`): `diag_multiseed_effect.png` (violines por seed A vs B) y `diag_multiseed_improvement.png` (Δ% por seed) + Wilcoxon pareado. Es la respuesta a "¿esto es reproducible o suerte de un seed?".
- **Three-way** (`compare_three_runs.py`): `diag_three_way.png` (boxplots A/B/C) + tabla con p-values pareados incluyendo B vs C (Webster). Es la respuesta a "¿la IA supera al óptimo teórico fixed-time?".
- **SUMO nativo** (`--use-sumo-tools`): plots oficiales de SUMO (plot_summary, distribuciones tripinfo, net dump, trayectorias) — útiles como validación independiente del pipeline propio de matplotlib.

## 4. Guion sugerido para presentar el resultado (múltiples audiencias)

1. **Ejecutivo:** `18_improvement_summary.png` + una frase con el CI bootstrap ("B reduce el tiempo medio de viaje X s, IC95 [a, b]").
2. **Técnico de tráfico:** `07/08` (series temporales) + `17` (dónde mejora, por edge) + `10` (congestión).
3. **Estadístico/revisor académico:** `01/03` (distribuciones), `14` (percentiles), tabla de tests, multiseed (`diag_multiseed_*`) y three-way vs Webster, más el análisis de incompletos como discusión de limitaciones.
4. **Validación de equidad del experimento:** tabla de la sección 2 de este reporte + resultado del test A/A (2.7).

## 5. Checklist para poder afirmar "A y B solo difieren en la optimización"

- [x] Mismos archivos de entrada (verificado byte a byte).
- [x] Misma demanda determinista (5 000 vehículos con departs fijos).
- [x] Mismo motor, mismos flags SUMO, sin teleport.
- [x] Mismo seed (implícito — hacerlo explícito con `--seed`).
- [x] La medición no perturba (solo lecturas TraCI; sim-time congelado durante HTTP).
- [ ] **Baseline aislado de traffic-control** — hoy NO garantizado (fix C1 del reporte de revisión).
- [ ] Test A/A ejecutado y archivado como evidencia de no-sesgo.
- [ ] Análisis de incompletos + throughput para **ambos** runs.
- [ ] Seed y modo de optimización registrados en `ab_report.json`.

Con los cuatro pendientes cerrados, la afirmación queda sólida y defendible.
