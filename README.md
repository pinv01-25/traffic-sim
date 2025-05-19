# Traffic-Sim: Simulación de Tráfico Urbano con SUMO

Traffic-Sim es el módulo generador de datos del sistema inteligente de gestión de tráfico. Simula tráfico urbano usando redes SUMO y genera observaciones estructuradas que son enviadas automáticamente a `traffic-control`. Expone una API FastAPI simple para control remoto del envío de datos.

---

## Estructura del Directorio
```
pinv01-25-traffic-sim/
├── README.md               # Este archivo
├── LICENSE                 # Licencia MIT
├── requirements.txt        # Dependencias de Python
├── run.sh                  # Script que lanza la API y ejecuta la simulación
├── server.py               # API FastAPI
├── data/                   # Archivos de red y OSM
│   ├── net/
│   │   ├── outer_edges.dst.xml
│   │   ├── outer_edges.src.xml
│   │   └── san_lorenzo.net.xml
│   └── osm/
│       └── san_lorenzo.osm
└── scripts/
└── simulation/
├── config_simulation.py       # Parámetros de simulación
├── generate_trips.py          # Generación de viajes
└── main.py                    # Lógica principal de simulación
```
---

## Requisitos Previos
* Python ≥ 3.10

* SUMO ≥ 1.18.0 (Simulation of Urban MObility)
Descarga: [https://www.eclipse.org/sumo/](https://www.eclipse.org/sumo/)

Asegúrate de que los comandos `sumo`, `sumo-gui`, `netconvert`, `duarouter` y `traci` estén disponibles en tu terminal.

---
## Primeros Pasos

### Instalar dependencias
```
pip install -r requirements.txt
```
### Ejecutar la simulación y la API
```
./run.sh
```
Visita: [http://localhost:8001](http://localhost:8001)

---

## API REST

### `POST /upload`

Recibe datos de tráfico desde el simulador y los reenvía a `traffic-control`.

**Ejemplo de carga:**
```json
{
"version": "1.0",
"type": "data",
"timestamp": "1682000000",
"traffic_light_id": "TL-101",
"controlled_edges": ["E1", "E2"],
"metrics": {
  "vehicles_per_minute": 30,
  "avg_speed_kmh": 40.5,
  "avg_circulation_time_sec": 35.2,
  "density": 85.3
  },
"vehicle_stats": {
  "motorcycle": 2,
  "car": 6,
  "bus": 1,
  "truck": 1
  }
}
```
**Respuesta esperada:**
```json
{
  "status": "ok",
  "control_response": {
    "version": "1.0",
    "type": "optimization",
    "timestamp": 1682000000,
    "traffic_light_id": "TL-101",
    "optimization": {
      "green_time_sec": 42,
      "red_time_sec": 48
    },
    "impact": {
      "original_congestion": 6,
      "optimized_congestion": 3,
      "original_category": "severe",
      "optimized_category": "mild"
    }
  }
}

```
---

## Arquitectura

### Simulación (`scripts/simulation/`)

* Crea vehículos, genera tráfico y captura métricas.

### API (`server.py`)

* Recibe los datos simulados.
* Envía los datos al endpoint `/process` de `traffic-control`.

---
## Autor
Majo Duarte
