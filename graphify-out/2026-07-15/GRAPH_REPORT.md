# Graph Report - traffic-sim  (2026-07-15)

## Corpus Check
- 41 files · ~32,844 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 612 nodes · 958 edges · 41 communities (36 shown, 5 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `177c224b`
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
- traffic-sim
- .setup_simulation
- TrafficControlClient
- MCP Tools: code-review-graph
- Debug Issue
- Explore Codebase
- Refactor Safely
- Review Changes
- code-review-graph
- metrics_calculator.py
- ._apply_traffic_optimization
- ._process_bottleneck_detection
- .run_simulation
- ._process_cluster_optimization
- Path
- Any

## God Nodes (most connected - your core abstractions)
1. `SimulationOrchestrator` - 37 edges
2. `_ensure_dir()` - 22 edges
3. `compare_runs()` - 17 edges
4. `MetricsCalculator` - 17 edges
5. `BottleneckDetector` - 16 edges
6. `TrafficLightController` - 15 edges
7. `parse_tripinfo()` - 13 edges
8. `TrafficControlClient` - 13 edges
9. `SimulationManager` - 12 edges
10. `analyze_incomplete_trips()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `IntersectionData` --uses--> `MetricsCalculator`  [INFERRED]
  detectors/bottleneck_detector.py → services/metrics_calculator.py
- `BottleneckDetection` --uses--> `MetricsCalculator`  [INFERRED]
  detectors/bottleneck_detector.py → services/metrics_calculator.py
- `SimulationOrchestrator` --uses--> `BottleneckDetection`  [INFERRED]
  simulation_orchestrator.py → detectors/bottleneck_detector.py
- `BottleneckDetector` --uses--> `MetricsCalculator`  [INFERRED]
  detectors/bottleneck_detector.py → services/metrics_calculator.py
- `SimulationOrchestrator` --uses--> `BottleneckDetector`  [INFERRED]
  simulation_orchestrator.py → detectors/bottleneck_detector.py

## Import Cycles
- None detected.

## Communities (41 total, 5 thin omitted)

### Community 0 - "SimulationOrchestrator"
Cohesion: 0.14
Nodes (11): main(), Verifica que SUMO esté instalado, Genera la red de simulación usando netconvert, Inicia la conexión TraCI con SUMO, Inicia el thread worker para manejar peticiones HTTP, Inicializa todos los componentes del sistema, Orquestador principal de la simulación de tráfico     Coordina todos los compone, Reanuda la simulación (+3 more)

### Community 1 - "ab_test.py"
Cohesion: 0.06
Nodes (69): ndarray, Tests de parsers de outputs SUMO., test_parse_summary_interval_format(), test_parse_summary_step_format(), test_parse_tripinfo(), _write(), analyze_incomplete_trips(), bootstrap_ci_diff() (+61 more)

### Community 2 - "__init__.py"
Cohesion: 0.09
Nodes (40): _ensure_dir(), plot_boxplot_two(), plot_congestion_timeline(), plot_correlation_heatmap(), plot_depart_delay_scatter(), plot_efficiency_comparison(), plot_fcd_comparison(), plot_fcd_speed_heatmap() (+32 more)

### Community 3 - "MetricsCalculator"
Cohesion: 0.13
Nodes (11): MetricsCalculator, Calcula la tasa de vehículos por minuto basándose en entradas reales.          A, Calcula la velocidad promedio en km/h de los vehículos visibles.          Args:, Calcula el tiempo promedio de circulación (espera) en segundos.          Args:, Calcula la densidad de vehículos por kilómetro en el rango visible.          Arg, Calcula todas las métricas para un edge específico.          Args:             e, Clasifica los vehículos visibles por tipo de vehículo.          Args:, Limpia tracking de vehículos antiguos para evitar acumulación de memoria (+3 more)

### Community 4 - "compare_three_runs.py"
Cohesion: 0.14
Nodes (20): format_pval(), load_all(), main(), mann_whitney(), plot_three_way(), print_summary_table(), Path, Series (+12 more)

### Community 5 - "generate_webster_timing.py"
Cohesion: 0.20
Nodes (17): compute_webster_timing(), count_vehicles_per_edge(), is_amber_state(), is_green_state(), main(), parse_tl_connections(), parse_tl_logics(), Path (+9 more)

### Community 6 - "metrics_calculator.py"
Cohesion: 0.11
Nodes (24): get_edge_length_from_lanes(), get_edge_vehicles(), get_lane_length(), get_vehicle_lane_id(), get_vehicle_lane_position(), get_vehicle_speed(), get_vehicle_type(), get_vehicle_waiting_time() (+16 more)

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
Cohesion: 0.11
Nodes (27): get_available_outputs(), Path, Scan a simulation directory for available SUMO output files.      Returns dict m, check_sumo_tools_available(), generate_all_sumo_plots(), _get_python_executable(), _get_sumo_tools_path(), plot_net_dump() (+19 more)

### Community 11 - "simulation_orchestrator.py"
Cohesion: 0.16
Nodes (13): Configuración del sistema de simulación de tráfico, Controlador de semáforos para actualización dinámica de tiempos, Logger, Cliente HTTP para comunicación con traffic-control, Orquestador principal de la simulación de tráfico Coordina detección de cuellos, ColoredFormatter, Colors, get_simulation_logger() (+5 more)

### Community 12 - "run_simulation.py"
Cohesion: 0.10
Nodes (24): main(), Run a single simulation instance., _run_instance(), extract_simulation_zip(), main(), Extrae un archivo ZIP con archivos de simulación SUMO          Args:         zip, Ejecuta la simulación con sumo (modo headless)      Args:         simulation_dir, Función principal para ejecutar la simulación (+16 more)

### Community 14 - "TrafficControlClient"
Cohesion: 0.07
Nodes (29): ClusterDataPayload, ClusterOptimizationResponse, Any, Payload con datos de múltiples semáforos para optimización de cluster.      Gene, Extrae solo los dígitos de un ID de semáforo., Convierte a dict compatible con TrafficData de traffic-control., Métricas de tráfico para enviar a traffic-control, Estadísticas de vehículos por tipo (+21 more)

### Community 15 - "TrafficDataPayload"
Cohesion: 0.10
Nodes (19): 1. Diseño del experimento, 2.1 Inputs — VERIFICADO IDÉNTICOS ✅, 2.2 Demanda — DETERMINISTA ✅, 2.3 Motor y flags — IDÉNTICOS POR CONSTRUCCIÓN ✅ (en `run_ab.py`), 2.4 Seed — IGUAL PERO IMPLÍCITO ⚠️, 2.5 Aislamiento del baseline — **NO GARANTIZADO** ❌ (hallazgo crítico), 2.6 ¿La medición perturba la simulación? — NO ✅, 2.7 Prueba de no-sesgo recomendada: test A/A (+11 more)

### Community 16 - "DescriptiveNames"
Cohesion: 0.18
Nodes (8): DescriptiveNames, Obtiene el nombre descriptivo de una calle, Obtiene el nombre descriptivo de una intersección, Obtiene las calles controladas por un semáforo, Manejador de nombres descriptivos para calles e intersecciones, Carga nombres de calles e intersecciones desde el archivo de red SUMO, Crea un nombre descriptivo para una calle basado en su ID, Crea un nombre descriptivo para una intersección

### Community 17 - "MetricsValidator"
Cohesion: 0.19
Nodes (10): MetricsValidationResult, MetricsValidator, Any, Validador de métricas de tráfico para verificar que los valores sean realistas, Valida una métrica individual, Valida patrones sospechosos en las métricas, Resultado de validación de métricas, Registra el resultado de validación (+2 more)

### Community 18 - "ClusterDataPayload"
Cohesion: 0.18
Nodes (8): BottleneckDetection, BottleneckDetector, IntersectionData, log_to_file(), Detector de cuellos de botella para intersecciones con semáforos, Calcula la longitud de cola (vehículos con velocidad < 1 m/s)., Escribe mensaje al archivo de log con timestamp, Verifica si ha pasado suficiente tiempo desde la última detección con misma seve

### Community 19 - "RawSimulationPayload"
Cohesion: 0.18
Nodes (10): 1. Hallazgos críticos, 2. Hallazgos mayores, 3. Hallazgos menores / limpieza, 4. Lo que está bien (vale la pena decirlo), 5. Acciones priorizadas, C1. El baseline (run A) puede recibir y aplicar optimizaciones — contamina el experimento A/B, C2. Deadlock en la API REST, C3. `parse_summary()` probablemente no parsea los summary.xml reales de SUMO (+2 more)

### Community 27 - ".setup_simulation"
Cohesion: 0.21
Nodes (16): compute_improvement_per_seed(), compute_seed_stats(), load_seed_results(), main(), plot_improvement_scatter(), plot_violin_by_seed(), DataFrame, Path (+8 more)

### Community 28 - "TrafficControlClient"
Cohesion: 0.31
Nodes (8): Path, main(), Ejecuta SUMO headless con un seed dado.      Para sim_A (fixed-time) se usa sumo, Corre SUMO como subprocess directo (sin TraCI). Más rápido, sin optimización., Corre la simulación via SimulationOrchestrator (con TraCI y optimización dinámic, run_sumo_headless(), _run_via_orchestrator(), _run_via_subprocess()

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

### Community 35 - "metrics_calculator.py"
Cohesion: 0.25
Nodes (6): Calculador de métricas de tráfico para traffic-sim Implementa cálculos precisos, Métricas de un vehículo individual, Métricas de tráfico calculadas, TrafficMetrics, VehicleMetrics, Utilidad para nombres descriptivos de calles e intersecciones Mapea IDs técnicos

### Community 36 - "._apply_traffic_optimization"
Cohesion: 0.22
Nodes (5): Any, Worker thread que procesa peticiones HTTP en segundo plano, Extrae datos de optimización de la respuesta, Aplica optimización de tráfico para un solo semáforo (legacy), Obtiene estadísticas de la simulación

### Community 37 - "._process_bottleneck_detection"
Cohesion: 0.22
Nodes (5): RawSimulationPayload, Maneja la detección de cuellos de botella, Procesa una detección de cuello de botella (versión no bloqueante), Crea el payload crudo para traffic-control /ingest (modo legacy single-sensor), Crea un payload crudo con datos de todos los semáforos del cluster.          No

### Community 38 - ".run_simulation"
Cohesion: 0.25
Nodes (4): Ejecuta la simulación principal, Determina si la simulación debe continuar (retorna True para continuar), Determina si es momento de detectar cuellos de botella, Limpia recursos de la simulación

### Community 39 - "._process_cluster_optimization"
Cohesion: 0.29
Nodes (4): ClusterOptimizationResponse, Procesa optimización de cluster: obtiene semáforos cercanos,         envía datos, Obtiene los semáforos cercanos al semáforo primario.          Args:, Aplica las optimizaciones recibidas de traffic-sync (vía traffic-control)

## Knowledge Gaps
- **63 isolated node(s):** `Steps`, `Tips`, `Token Efficiency Rules`, `Steps`, `Tips` (+58 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SimulationOrchestrator` connect `SimulationOrchestrator` to `._apply_traffic_optimization`, `._process_bottleneck_detection`, `.run_simulation`, `SimulationManager`, `._process_cluster_optimization`, `simulation_orchestrator.py`, `run_simulation.py`, `ClusterDataPayload`, `TrafficControlClient`?**
  _High betweenness centrality (0.229) - this node is a cross-community bridge._
- **Why does `generate_visualizations()` connect `ab_test.py` to `run_simulation.py`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Why does `parse_tripinfo()` connect `ab_test.py` to `.setup_simulation`, `compare_three_runs.py`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `SimulationOrchestrator` (e.g. with `SimulationManager` and `SimulationStatus`) actually correct?**
  _`SimulationOrchestrator` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `MetricsCalculator` (e.g. with `BottleneckDetection` and `BottleneckDetector`) actually correct?**
  _`MetricsCalculator` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `BottleneckDetector` (e.g. with `MetricsCalculator` and `SimulationOrchestrator`) actually correct?**
  _`BottleneckDetector` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Steps`, `Tips`, `Token Efficiency Rules` to the rest of the system?**
  _63 weakly-connected nodes found - possible documentation gaps or missing edges._