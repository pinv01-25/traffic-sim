# Graph Report - traffic-sim  (2026-07-15)

## Corpus Check
- 40 files · ~33,217 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 611 nodes · 1082 edges · 35 communities (33 shown, 2 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5d71c20e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- SimulationOrchestrator
- ab_test.py
- __init__.py
- MetricsCalculator
- compare_three_runs.py
- generate_webster_timing.py
- metrics_calculator.py
- SimulationManager
- TrafficLightController
- Traffic-Sim: Simulador de Tráfico Inteligente
- sumo_tools.py
- simulation_orchestrator.py
- run_simulation.py
- parsers.py
- TrafficControlClient
- TrafficDataPayload
- DescriptiveNames
- MetricsValidator
- ClusterDataPayload
- RawSimulationPayload
- plot_percentile_comparison
- traffic-sim
- .setup_simulation
- TrafficControlClient
- MCP Tools: code-review-graph
- Debug Issue
- Explore Codebase
- Refactor Safely
- Review Changes
- code-review-graph

## God Nodes (most connected - your core abstractions)
1. `SimulationOrchestrator` - 43 edges
2. `compare_runs()` - 35 edges
3. `_ensure_dir()` - 22 edges
4. `TrafficLightController` - 18 edges
5. `MetricsCalculator` - 17 edges
6. `BottleneckDetector` - 16 edges
7. `TrafficControlClient` - 16 edges
8. `generate_visualizations()` - 15 edges
9. `SimulationManager` - 12 edges
10. `setup_logger()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `SimulationStatus` --uses--> `SimulationOrchestrator`  [INFERRED]
  api/server.py → simulation_orchestrator.py
- `SimulationManager` --uses--> `SimulationOrchestrator`  [INFERRED]
  api/server.py → simulation_orchestrator.py
- `SimulationOrchestrator` --uses--> `TrafficLightController`  [INFERRED]
  simulation_orchestrator.py → controllers/traffic_light_controller.py
- `IntersectionData` --uses--> `MetricsCalculator`  [INFERRED]
  detectors/bottleneck_detector.py → services/metrics_calculator.py
- `BottleneckDetection` --uses--> `MetricsCalculator`  [INFERRED]
  detectors/bottleneck_detector.py → services/metrics_calculator.py

## Import Cycles
- None detected.

## Communities (35 total, 2 thin omitted)

### Community 0 - "SimulationOrchestrator"
Cohesion: 0.06
Nodes (28): BottleneckDetection, main(), Any, Verifica que SUMO esté instalado, Genera la red de simulación usando netconvert, Inicia la conexión TraCI con SUMO, Inicia el thread worker para manejar peticiones HTTP, Worker thread que procesa peticiones HTTP en segundo plano (+20 more)

### Community 1 - "ab_test.py"
Cohesion: 0.09
Nodes (37): analyze_incomplete_trips(), bootstrap_ci_diff(), cohens_d(), _compute_incomplete_trips(), _find_file(), _get_depart_times(), _get_last_fcd_times(), _load_fcd() (+29 more)

### Community 2 - "__init__.py"
Cohesion: 0.13
Nodes (37): ensure_dir(), generate_visualizations(), Visualization utilities for traffic-sim.  This module provides comprehensive vis, Generate a set of visualizations from SUMO outputs found under     `simulation_d, _ensure_dir(), plot_boxplot_two(), plot_congestion_timeline(), plot_correlation_heatmap() (+29 more)

### Community 3 - "MetricsCalculator"
Cohesion: 0.14
Nodes (10): MetricsCalculator, Calcula la tasa de vehículos por minuto basándose en entradas reales.          A, Calcula la velocidad promedio en km/h de los vehículos visibles.          Args:, Calcula el tiempo promedio de circulación (espera) en segundos.          Args:, Calcula la densidad de vehículos por kilómetro en el rango visible.          Arg, Calcula todas las métricas para un edge específico.          Args:             e, Clasifica los vehículos visibles por tipo de vehículo.          Args:, Limpia tracking de vehículos antiguos para evitar acumulación de memoria (+2 more)

### Community 4 - "compare_three_runs.py"
Cohesion: 0.10
Nodes (32): compute_improvement_per_seed(), compute_seed_stats(), load_seed_results(), main(), plot_improvement_scatter(), plot_violin_by_seed(), DataFrame, Path (+24 more)

### Community 5 - "generate_webster_timing.py"
Cohesion: 0.10
Nodes (31): compute_webster_timing(), count_vehicles_per_edge(), is_amber_state(), is_green_state(), main(), parse_tl_connections(), parse_tl_logics(), Path (+23 more)

### Community 6 - "metrics_calculator.py"
Cohesion: 0.09
Nodes (31): Calculador de métricas de tráfico para traffic-sim Implementa cálculos precisos, Métricas de un vehículo individual, Métricas de tráfico calculadas, Obtiene vehículos visibles en el rango especificado desde el semáforo., TrafficMetrics, VehicleMetrics, Validador de métricas de tráfico para verificar que los valores sean realistas, get_edge_length_from_lanes() (+23 more)

### Community 7 - "SimulationManager"
Cohesion: 0.08
Nodes (21): get_simulation_status(), health_check(), healthcheck(), list_simulations(), Any, Get simulation status, Run simulation in background thread, Setup simulation files from config (+13 more)

### Community 8 - "TrafficLightController"
Cohesion: 0.10
Nodes (17): Any, Crea un programa por defecto para un semáforo                  Args:, Actualiza un semáforo con nuevos tiempos de optimización                  Args:, Representa una fase de semáforo, Crea un programa optimizado basado en los nuevos tiempos                  Args:, Programa completo de un semáforo, Obtiene los estados actuales de un semáforo                  Args:             t, Aplica un programa a un semáforo usando setCompleteRedYellowGreenDefinition. (+9 more)

### Community 9 - "Traffic-Sim: Simulador de Tráfico Inteligente"
Cohesion: 0.07
Nodes (27): Arquitectura, Auto-detección de directorios, Características, Comparador Webster (sim_C), Configuración, Ejecución básica, Estructura del ZIP, Experimento Multi-Seed (+19 more)

### Community 10 - "sumo_tools.py"
Cohesion: 0.13
Nodes (20): _get_python_executable(), _get_sumo_tools_path(), plot_net_dump(), plot_net_speeds(), plot_summary(), plot_trajectories(), plot_tripinfo_distributions(), plot_xml_attributes() (+12 more)

### Community 11 - "simulation_orchestrator.py"
Cohesion: 0.12
Nodes (19): Configuración del sistema de simulación de tráfico, Controlador de semáforos para actualización dinámica de tiempos, Detector de cuellos de botella para intersecciones con semáforos, Logger, ClusterOptimizationResponse, Cliente HTTP para comunicación con traffic-control, Payload completo para enviar a traffic-control, Respuesta de optimización para un cluster de semáforos.      Parsea la respuesta (+11 more)

### Community 12 - "run_simulation.py"
Cohesion: 0.10
Nodes (26): main(), Run a single simulation instance., _run_instance(), extract_simulation_zip(), main(), Extrae un archivo ZIP con archivos de simulación SUMO          Args:         zip, Ejecuta la simulación con sumo (modo headless)      Args:         simulation_dir, Función principal para ejecutar la simulación (+18 more)

### Community 13 - "parsers.py"
Cohesion: 0.14
Nodes (22): compute_summary_statistics(), compute_trip_statistics(), _is_float(), parse_detector_output(), parse_edge_data(), parse_fcd(), parse_fcd_aggregated(), parse_fcd_by_edge() (+14 more)

### Community 14 - "TrafficControlClient"
Cohesion: 0.09
Nodes (20): ClusterDataPayload, Any, Payload con datos de múltiples semáforos para optimización de cluster.      Gene, Extrae solo los dígitos de un ID de semáforo., Convierte a dict compatible con TrafficData de traffic-control., Normaliza los campos para cumplir con el formato requerido por traffic-control (, Convierte el payload a diccionario para envío HTTP, Cliente para comunicación síncrona con traffic-control (+12 more)

### Community 15 - "TrafficDataPayload"
Cohesion: 0.10
Nodes (19): 1. Diseño del experimento, 2.1 Inputs — VERIFICADO IDÉNTICOS ✅, 2.2 Demanda — DETERMINISTA ✅, 2.3 Motor y flags — IDÉNTICOS POR CONSTRUCCIÓN ✅ (en `run_ab.py`), 2.4 Seed — IGUAL PERO IMPLÍCITO ⚠️, 2.5 Aislamiento del baseline — **NO GARANTIZADO** ❌ (hallazgo crítico), 2.6 ¿La medición perturba la simulación? — NO ✅, 2.7 Prueba de no-sesgo recomendada: test A/A (+11 more)

### Community 16 - "DescriptiveNames"
Cohesion: 0.18
Nodes (8): DescriptiveNames, Obtiene el nombre descriptivo de una calle, Obtiene el nombre descriptivo de una intersección, Obtiene las calles controladas por un semáforo, Manejador de nombres descriptivos para calles e intersecciones, Carga nombres de calles e intersecciones desde el archivo de red SUMO, Crea un nombre descriptivo para una calle basado en su ID, Crea un nombre descriptivo para una intersección

### Community 17 - "MetricsValidator"
Cohesion: 0.22
Nodes (9): MetricsValidationResult, MetricsValidator, Any, Valida una métrica individual, Valida patrones sospechosos en las métricas, Resultado de validación de métricas, Registra el resultado de validación, Validador de métricas de tráfico para detectar valores irreales (+1 more)

### Community 18 - "ClusterDataPayload"
Cohesion: 0.20
Nodes (6): BottleneckDetector, IntersectionData, log_to_file(), Calcula la longitud de cola (vehículos con velocidad < 1 m/s)., Escribe mensaje al archivo de log con timestamp, Verifica si ha pasado suficiente tiempo desde la última detección con misma seve

### Community 19 - "RawSimulationPayload"
Cohesion: 0.18
Nodes (10): 1. Hallazgos críticos, 2. Hallazgos mayores, 3. Hallazgos menores / limpieza, 4. Lo que está bien (vale la pena decirlo), 5. Acciones priorizadas, C1. El baseline (run A) puede recibir y aplicar optimizaciones — contamina el experimento A/B, C2. Deadlock en la API REST, C3. `parse_summary()` probablemente no parsea los summary.xml reales de SUMO (+2 more)

### Community 20 - "plot_percentile_comparison"
Cohesion: 0.15
Nodes (16): compare_runs(), Path, Comprehensive comparison of two simulation runs.      Generates all available vi, Write CSV summary of comparison results., Write JSON report with full analysis results., _write_csv_summary(), _write_json_report(), plot_metric_comparison_bars() (+8 more)

### Community 27 - ".setup_simulation"
Cohesion: 0.33
Nodes (5): Métricas de tráfico para enviar a traffic-control, Estadísticas de vehículos por tipo, Crea un payload de datos de tráfico a partir de métricas de simulación, TrafficMetrics, VehicleStats

### Community 28 - "TrafficControlClient"
Cohesion: 0.40
Nodes (5): get_available_outputs(), Path, Scan a simulation directory for available SUMO output files.      Returns dict m, generate_all_sumo_plots(), Generate all available SUMO native plots for A/B comparison.      Scans both sim

### Community 29 - "MCP Tools: code-review-graph"
Cohesion: 0.40
Nodes (4): Key Tools, MCP Tools: code-review-graph, When to use graph tools FIRST, Workflow

### Community 30 - "Debug Issue"
Cohesion: 0.40
Nodes (4): Debug Issue, Steps, Tips, Token Efficiency Rules

### Community 31 - "Explore Codebase"
Cohesion: 0.40
Nodes (4): Explore Codebase, Steps, Tips, Token Efficiency Rules

### Community 32 - "Refactor Safely"
Cohesion: 0.40
Nodes (4): Refactor Safely, Safety Checks, Steps, Token Efficiency Rules

### Community 33 - "Review Changes"
Cohesion: 0.40
Nodes (4): Output Format, Review Changes, Steps, Token Efficiency Rules

## Knowledge Gaps
- **62 isolated node(s):** `uvx`, `traffic-sim`, `Steps`, `Tips`, `Token Efficiency Rules` (+57 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SimulationOrchestrator` connect `SimulationOrchestrator` to `generate_webster_timing.py`, `SimulationManager`, `TrafficLightController`, `simulation_orchestrator.py`, `run_simulation.py`, `TrafficControlClient`, `ClusterDataPayload`?**
  _High betweenness centrality (0.256) - this node is a cross-community bridge._
- **Why does `generate_visualizations()` connect `__init__.py` to `compare_three_runs.py`, `run_simulation.py`, `parsers.py`?**
  _High betweenness centrality (0.186) - this node is a cross-community bridge._
- **Why does `generate_ab_test()` connect `run_simulation.py` to `__init__.py`, `plot_percentile_comparison`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `SimulationOrchestrator` (e.g. with `SimulationManager` and `SimulationStatus`) actually correct?**
  _`SimulationOrchestrator` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `MetricsCalculator` (e.g. with `BottleneckDetection` and `BottleneckDetector`) actually correct?**
  _`MetricsCalculator` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `uvx`, `traffic-sim`, `Steps` to the rest of the system?**
  _62 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `SimulationOrchestrator` be split into smaller, more focused modules?**
  _Cohesion score 0.05725490196078432 - nodes in this community are weakly interconnected._