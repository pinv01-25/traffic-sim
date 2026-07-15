# Reporte de revisión del repositorio — traffic-sim

**Fecha:** 2026-07-15 · **Base:** commit `5d71c20` + cambios sin commitear
**Método:** knowledge graph (graphify / code-review-graph: 580 nodos, 1053 edges, `detect-changes` risk score 0.80) + lectura dirigida de los 31 archivos fuente + verificación cruzada contra `traffic-control`.

---

## 1. Hallazgos críticos

### C1. El baseline (run A) puede recibir y aplicar optimizaciones — contamina el experimento A/B

**Cadena verificada:**

1. En modo legacy (sin `--dynamic-optimization`, que es exactamente cómo corre el **baseline A** en `run_ab.py`), el orquestador detecta cuellos de botella y encola payloads hacia traffic-control (`simulation_orchestrator.py:397-400`).
2. El worker HTTP los envía a `/ingest` (`simulation_orchestrator.py:246`).
3. `/ingest` en traffic-control **no es un endpoint de solo-ingesta**: ejecuta el pipeline completo (storage → traffic-sync → optimización) y **devuelve el batch de optimizaciones** (`traffic-control/api/server.py:73-91` → `ProcessService.process_data_batch` → `processing_success_with_optimization`).
4. El worker legacy **aplica esas optimizaciones** a los semáforos (`simulation_orchestrator.py:252-259` → `_apply_traffic_optimization`).
5. `_apply_traffic_optimization` usa el `traffic_light_id` normalizado (solo dígitos) sin mapear de vuelta al ID SUMO — pero en esta red **los IDs SUMO son puramente numéricos** (nodos OSM: `1437431229`, `20928101`, …), así que el ID normalizado coincide y `update_traffic_light()` **sí se ejecuta sobre el semáforo real**.

**Consecuencia:** si traffic-control está corriendo cuando se lanza el baseline, el run "A (Baseline)" también recibe timing optimizado — aplicado además en momentos no deterministas (worker asíncrono). El A/B deja de medir "sin optimizar vs optimizado".

**Mitigación actual (frágil):** el baseline solo queda limpio si traffic-control está apagado o falla. Eso no es una garantía de diseño.

**Fix recomendado:** en modo legacy no aplicar nunca las optimizaciones de la respuesta (solo telemetría), o agregar un flag `--no-control` / `baseline=True` que desactive por completo el cliente HTTP en el run A de `run_ab.py`.

### C2. Deadlock en la API REST

`SimulationManager.get_all_simulations()` adquiere `self.lock` y dentro del `with` llama a `self.get_simulation_status()`, que vuelve a adquirir el mismo `threading.Lock` (no reentrante) → **`GET /simulations` y `GET /simulation/status` (sin id) se cuelgan para siempre** (`api/server.py:189-199` + `api/server.py:169`). Fix: usar `threading.RLock()` o extraer un `_get_status_unlocked()`.

### C3. `parse_summary()` probablemente no parsea los summary.xml reales de SUMO

SUMO escribe el summary-output con elementos `<step time=... running=... halting=.../>`. El parser busca tags `interval` / `summary` (`visualization/parsers.py:51-65`) → para archivos reales devuelve un DataFrame **vacío silenciosamente**, y con eso los gráficos `09_summary_comparison.png`, `10_congestion_timeline.png`, `summary_timeline.png` y `compute_summary_statistics()` quedan vacíos o no se generan.
*Nota: no había ningún `summary.xml` en disco ni SUMO instalado localmente para confirmarlo al 100 % — verificar con una corrida real; el fix es trivial (aceptar también el tag `step`).*

### C4. La optimización dinámica reconstruye programas de semáforo con estados potencialmente inválidos

`TrafficLightController._create_optimized_program()` (`controllers/traffic_light_controller.py:194-247`):

- Toma `current_states[0]` como fase verde y `current_states[1]` como fase "roja" — en un programa SUMO típico (verde→amarillo→verde2→amarillo2) `states[1]` es la fase **amarilla**, así que la fase "Optimized_Red" puede quedar con estado amarillo y duración de rojo.
- Hardcodea estados de 16 caracteres (`"yyyyrrrryyyyrrrr"`) que solo son válidos si el semáforo tiene exactamente 16 links; con otro número de links `setCompleteRedYellowGreenDefinition` falla o corrompe la semántica.
- Colapsa cualquier programa multi-fase a 4 fases fijas.

En contraste, el camino estático (`utils/signal_utils.apply_timings_to_all_tls`) preserva los estados originales y solo modifica duraciones — es el enfoque correcto. **Inconsistencia entre los dos caminos de optimización**; el dinámico debería reutilizar la lógica de `signal_utils`. Esto afecta directamente la validez del run B en modo `--dynamic-optimization`.

---

## 2. Hallazgos mayores

| # | Hallazgo | Ubicación |
|---|----------|-----------|
| M1 | **Duplicación GUI/headless con divergencia de comportamiento**: `run_with_sumo_gui()` reimplementa ~300 líneas del orquestador. Divergencia real: el worker legacy del GUI **no** aplica optimizaciones de la respuesta, el del headless **sí** (ver C1). | `run_simulation.py:76-401` |
| M2 | **Análisis de viajes incompletos solo para el run A**: `analyze_incomplete_trips(run_a, ...)` nunca corre para B. Si B completa más viajes (mayor throughput), la comparación de duraciones de completados sufre sesgo de supervivencia asimétrico sin diagnóstico. | `visualization/ab_test.py:590-594` |
| M3 | **`run_ab.py` no expone `--seed`** aunque `run_simulation.py` lo soporta. A y B quedan idénticos solo porque SUMO usa un seed default fijo — funciona, pero es implícito y nadie lo registra en el reporte. | `run_ab.py:151-165` |
| M4 | **Asimetría mecánica en multiseed**: `sim_A` corre vía subprocess SUMO puro con `--end N`; `sim_B` con `--dynamic-optimization` corre vía orquestador TraCI que corta por conteo de pasos y hereda `end_time=3600` de `SIMULATION_CONFIG`. Equivalentes con `step_length=1.0` y `sim_steps<3600`, pero son dos mecanismos de terminación distintos; unificar. | `scripts/run_multiseed.py:59-148` |
| M5 | **`stop_simulation` de la API no detiene el loop**: solo marca estado y llama `_cleanup()`; el thread del orquestador muere por la excepción de `traci.close()` desde otro thread. Funciona por accidente. Además dos simulaciones simultáneas compartirían la conexión TraCI global (sin label). | `api/server.py:125-164` |
| M6 | **Borrado prematuro en A/B implícito**: en `main()` de `run_simulation.py`, si no hay `--keep-files` y `extract_dir` es None, se borra `simulation_dir` **antes** del bloque de comparación que lo usa. | `run_simulation.py:594-627` |
| M7 | **Wilcoxon one-sided (`alternative="less"`)** en el multiseed presupone la dirección de la mejora. Si B empeora, el p-value sale alto y se reporta "no significativo" en vez de "significativamente peor". Usar two-sided o declarar la hipótesis direccional. | `scripts/aggregate_seeds.py:120` |
| M8 | **No hay tests**: 0 archivos de test en el repo; `detect-changes` reporta 49 test gaps y risk 0.80 sobre los cambios sin commitear. Lo mínimo: un test del parser de tripinfo/summary con XML fixture y un test de `compute_webster_timing`. | repo completo |

---

## 3. Hallazgos menores / limpieza

- `import time` quedó sin uso en `simulation_orchestrator.py` tras eliminar el `time.sleep(0.05)` (diff sin commitear).
- `_should_stop_simulation()` retorna True para **continuar** — nombre invertido (`simulation_orchestrator.py:328`).
- `BottleneckDetector.get_intersection_data()`: `total_waiting_time` se acumula y nunca se usa; el cálculo de densidad agregada re-llama `get_visible_vehicles()` por edge dentro de un list-comprehension → O(n²) y doble consulta TraCI (`detectors/bottleneck_detector.py:121,164-177`).
- Claves inconsistentes: `intersection_edges.get(detection.intersection_id)` vs `.get(detection.traffic_light_id)` en distintos sitios — hoy inofensivo porque `intersection_id == traffic_light_id` por construcción, pero es una trampa latente.
- `config.py`: `min_detection_duration` se usa como *cooldown* entre detecciones de la misma severidad, no como "duración mínima para confirmar"; `detection_interval` está en pasos, no segundos (equivalen solo si `step_length=1.0`).
- `MetricsCalculator.calculate_vehicles_per_minute()`: si no hay entradas recientes **inventa** un valor (`vehículos_visibles × 3`) — heurística sin fuente que llega hasta el optimizador; documentar o eliminar.
- `descriptive_names` es un singleton global que al importar intenta cargar `simulation/network.net.xml` (path hardcodeado, casi nunca existe en runs A/B) y llama TraCI sin conexión — degrada a IDs con warnings. Debería inicializarse con el dir de simulación real.
- `parse_edge_data()` usa `elem.getparent()` (solo lxml) guardado por `hasattr` — variable muerta (`visualization/parsers.py:219`).
- `compare_multiple_metrics()` calcula "pct_improvement" también para `routeLength`, donde "mejora" no tiene sentido direccional (`visualization/ab_test.py:834`).
- Código legacy aparentemente muerto y aún exportado: `TrafficDataPayload`, `ClusterDataPayload`, `send_traffic_data()`, `send_traffic_data_batch()`, `send_cluster_data()` (marcado DEPRECATED) en `services/traffic_control_client.py`. Candidatos a borrar.
- `_run_via_orchestrator()` en multiseed crea symlinks a archivos de salida inexistentes (dangling) para redirigir outputs — funciona, pero un `--tripinfo-output` apuntando directo a `out_dir` sería más simple que el tmpdir simlinkeado.
- `generate_webster_timing.count_vehicles_per_edge()` depende del orden `<route>` seguido de su `<vehicle>` (adyacencia por suerte del generador). Si un vehículo referencia una ruta compartida definida antes, contaría la ruta equivocada.
- `run_ab.py:148`: si el usuario pasa explícitamente `--label-b "Optimized"`, se sobreescribe con "Dynamic Optimization" en modo dinámico.
- El log `logs/bottleneck_detection.log` se abre por append con path relativo al CWD — runs A y B mezclan sus líneas en el mismo archivo.

## 4. Lo que está bien (vale la pena decirlo)

- El grafo no detecta **ciclos de import**; la arquitectura por capas (detector → metrics → client / orchestrator arriba) es limpia.
- `utils/traci_helpers.py` centraliza bien el manejo de errores TraCI.
- La batería estadística del A/B (permutación + bootstrap CI + Mann-Whitney + Cohen's d) es correcta y está bien implementada, con RNG seedeado (reproducible).
- El diff sin commitear va en la dirección correcta: eliminó el `time.sleep(0.05)` del loop headless, quitó el piso artificial de 5 km/h en velocidad promedio y corrigió la documentación de unidades del umbral de velocidad.
- `signal_utils.apply_timings_to_all_tls` (camino estático) preserva estados y registra el CSV de duraciones asignadas — buen patrón de auditoría.

## 5. Acciones priorizadas

1. **C1** — aislar el baseline del pipeline de optimización (flag explícito). Es la amenaza directa a la validez del experimento.
2. **C4** — hacer que el camino dinámico reutilice `signal_utils` (preservar estados/fases).
3. **C3** — soportar tag `<step>` en `parse_summary` (+ fixture de test).
4. **C2** — RLock o método sin lock en la API.
5. **M2/M3** — incompletos para ambos runs + `--seed` explícito en `run_ab.py` (registrado en `ab_report.json`).
6. Suite mínima de tests para parsers y Webster.
