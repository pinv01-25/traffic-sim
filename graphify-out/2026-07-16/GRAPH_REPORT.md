# Graph Report - traffic-sim  (2026-07-16)

## Corpus Check
- 46 files · ~29,744 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 572 nodes · 861 edges · 37 communities (27 shown, 10 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `765f760e`
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
- TrafficDataPayload
- DescriptiveNames
- MetricsValidator
- ClusterDataPayload
- RawSimulationPayload
- traffic_control_client.py
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
- Path
- Any

## God Nodes (most connected - your core abstractions)
1. `SimulationOrchestrator` - 26 edges
2. `_ensure_dir()` - 22 edges
3. `MetricsCalculator` - 17 edges
4. `compare_runs()` - 17 edges
5. `SimulationManager` - 11 edges
6. `analyze_incomplete_trips()` - 11 edges
7. `parse_tripinfo()` - 11 edges
8. `parse_summary()` - 11 edges
9. `Traffic-Sim: Simulador de Tráfico Inteligente` - 11 edges
10. `_run_sumo_tool()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `IntersectionData` --uses--> `MetricsCalculator`  [INFERRED]
  detectors/bottleneck_detector.py → services/metrics_calculator.py
- `BottleneckDetection` --uses--> `MetricsCalculator`  [INFERRED]
  detectors/bottleneck_detector.py → services/metrics_calculator.py
- `BottleneckDetector` --uses--> `MetricsCalculator`  [INFERRED]
  detectors/bottleneck_detector.py → services/metrics_calculator.py
- `_run_via_orchestrator()` --calls--> `SimulationOrchestrator`  [EXTRACTED]
  scripts/run_multiseed.py → simulation_orchestrator.py
- `load_all()` --calls--> `parse_tripinfo()`  [EXTRACTED]
  scripts/compare_three_runs.py → visualization/parsers.py

## Import Cycles
- None detected.

## Communities (37 total, 10 thin omitted)

### Community 1 - "ab_test.py"
Cohesion: 0.06
Nodes (74): ndarray, main(), Run a single simulation instance., _run_instance(), Tests de parsers de outputs SUMO., test_parse_summary_interval_format(), test_parse_summary_step_format(), test_parse_tripinfo() (+66 more)

### Community 2 - "__init__.py"
Cohesion: 0.09
Nodes (40): _ensure_dir(), plot_boxplot_two(), plot_congestion_timeline(), plot_correlation_heatmap(), plot_depart_delay_scatter(), plot_efficiency_comparison(), plot_fcd_comparison(), plot_fcd_speed_heatmap() (+32 more)

### Community 4 - "compare_three_runs.py"
Cohesion: 0.14
Nodes (20): format_pval(), load_all(), main(), mann_whitney(), plot_three_way(), print_summary_table(), Path, Series (+12 more)

### Community 5 - "generate_webster_timing.py"
Cohesion: 0.15
Nodes (18): compute_webster_timing(), count_vehicles_per_edge(), is_amber_state(), is_green_state(), main(), parse_tl_connections(), parse_tl_logics(), Path (+10 more)

### Community 6 - "metrics_calculator.py"
Cohesion: 0.11
Nodes (24): get_edge_length_from_lanes(), get_edge_vehicles(), get_lane_length(), get_vehicle_lane_id(), get_vehicle_lane_position(), get_vehicle_speed(), get_vehicle_type(), get_vehicle_waiting_time() (+16 more)

### Community 7 - "SimulationManager"
Cohesion: 0.08
Nodes (21): get_simulation_status(), health_check(), healthcheck(), list_simulations(), Any, Get simulation status, Run simulation in background thread, Setup simulation files from config (+13 more)

### Community 9 - "Traffic-Sim: Simulador de Tráfico Inteligente"
Cohesion: 0.07
Nodes (27): Arquitectura, Auto-detección de directorios, Características, Comparador Webster (sim_C), Configuración, Ejecución básica, Estructura del ZIP, Experimento Multi-Seed (+19 more)

### Community 10 - "sumo_tools.py"
Cohesion: 0.11
Nodes (27): get_available_outputs(), Path, Scan a simulation directory for available SUMO output files.      Returns dict m, check_sumo_tools_available(), generate_all_sumo_plots(), _get_python_executable(), _get_sumo_tools_path(), plot_net_dump() (+19 more)

### Community 12 - "run_simulation.py"
Cohesion: 0.07
Nodes (26): BottleneckDetection, BottleneckDetector, IntersectionData, log_to_file(), Detector de cuellos de botella para intersecciones con semáforos, Calcula la longitud de cola (vehículos con velocidad < 1 m/s)., Escribe mensaje al archivo de log con timestamp, Verifica si ha pasado suficiente tiempo desde la última detección con misma seve (+18 more)

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
Cohesion: 0.17
Nodes (19): FakeLogic, FakePhase, _make_fake_traci(), Fixtures compartidas: fake de traci para testear sin SUMO.  `utils.signal_utils`, _program_4_phases(), Tests de la primitiva de aplicación de tiempos a semáforos., apply_timings_to_all_tls(green, cycle): rojo = ciclo - verde - amarillo., test_apply_durations_distributes_red_to_red_phases() (+11 more)

### Community 19 - "RawSimulationPayload"
Cohesion: 0.18
Nodes (10): 1. Hallazgos críticos, 2. Hallazgos mayores, 3. Hallazgos menores / limpieza, 4. Lo que está bien (vale la pena decirlo), 5. Acciones priorizadas, C1. El baseline (run A) puede recibir y aplicar optimizaciones — contamina el experimento A/B, C2. Deadlock en la API REST, C3. `parse_summary()` probablemente no parsea los summary.xml reales de SUMO (+2 more)

### Community 20 - "traffic_control_client.py"
Cohesion: 0.06
Nodes (31): Any, Configuración del sistema de simulación de tráfico, Controlador de semáforos para actualización dinámica de tiempos, Aplica tiempos de optimización a semáforos preservando la estructura     de fase, Actualiza un semáforo con nuevos tiempos de optimización.          Args:, TrafficLightController, Logger, ClusterOptimizationResponse (+23 more)

### Community 27 - ".setup_simulation"
Cohesion: 0.14
Nodes (23): DataFrame, Path, compute_improvement_per_seed(), compute_seed_stats(), load_seed_results(), main(), plot_improvement_scatter(), plot_violin_by_seed() (+15 more)

### Community 28 - "TrafficControlClient"
Cohesion: 0.06
Nodes (31): BottleneckDetection, ClusterOptimizationResponse, RawSimulationPayload, extract_simulation_zip(), main(), Resuelve el directorio A para comparación implícita (<base>_B → <base>_A)., Función principal para ejecutar la simulación, Extrae un archivo ZIP con archivos de simulación SUMO      Args:         zip_pat (+23 more)

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
- **63 isolated node(s):** `Steps`, `Tips`, `Token Efficiency Rules`, `Steps`, `Tips` (+58 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SimulationOrchestrator` connect `TrafficControlClient` to `.setup_simulation`, `traffic_control_client.py`?**
  _High betweenness centrality (0.325) - this node is a cross-community bridge._
- **Why does `_run_via_orchestrator()` connect `.setup_simulation` to `TrafficControlClient`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `get_available_outputs()` connect `sumo_tools.py` to `ab_test.py`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `MetricsCalculator` (e.g. with `BottleneckDetection` and `BottleneckDetector`) actually correct?**
  _`MetricsCalculator` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Steps`, `Tips`, `Token Efficiency Rules` to the rest of the system?**
  _63 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `ab_test.py` be split into smaller, more focused modules?**
  _Cohesion score 0.055501460564751706 - nodes in this community are weakly interconnected._
- **Should `__init__.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09024390243902439 - nodes in this community are weakly interconnected._