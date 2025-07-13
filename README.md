# Traffic-Sim: Simulador de Tráfico Inteligente

Traffic-Sim es un sistema de simulación de tráfico urbano que utiliza SUMO (Simulation of Urban MObility) para detectar cuellos de botella en tiempo real y optimizar dinámicamente los semáforos mediante comunicación con el servicio traffic-control.

## Características

- **Detección automática de cuellos de botella**: Analiza densidad de vehículos, velocidad promedio y longitud de colas
- **Comunicación síncrona con traffic-control**: Envía datos de tráfico y recibe optimizaciones de semáforos
- **Control dinámico de semáforos**: Actualiza tiempos de semáforos en tiempo real
- **Pipeline completo**: Desde detección hasta aplicación de optimizaciones
- **Logging robusto**: Sistema completo de logs para monitoreo y debugging
- **Dos modos de ejecución**: GUI (interfaz gráfica) y Headless (consola)

## Arquitectura

```
traffic-sim/
├── config.py                    # Configuración del sistema
├── run_simulation.py           # Script principal de ejecución
├── simulation_orchestrator.py   # Orquestador principal
├── requirements.txt             # Dependencias Python
├── utils/                       # Utilidades
│   ├── logger.py               # Sistema de logging
├── detectors/                   # Detectores
│   ├── bottleneck_detector.py  # Detector de cuellos de botella
├── services/                    # Servicios externos
│   ├── traffic_control_client.py # Cliente HTTP para traffic-control
├── controllers/                 # Controladores
│   ├── traffic_light_controller.py # Controlador de semáforos
└── simulation/                  # Archivos de simulación SUMO
    ├── edges.edg.xml
    ├── nodes.nod.xml
    ├── routes.rou.xml
    ├── simulation.sumocfg
    └── traffic_lights.add.xml
```

## Pipeline de Funcionamiento

1. **Inicia la simulación** con SUMO y traci
2. **Entran los autos** según las rutas definidas
3. **Se detecta un cuello de botella** en una intersección con semáforo
4. **Se envía información** de la calle (vehículos, velocidad, densidad)
5. **Se pausa momentáneamente** la simulación mientras se reciben datos
6. **Se actualizan los tiempos** de los semáforos
7. **Se continúa la simulación** hasta detectar otro cuello de botella
8. **Se termina** cuando no hay autos o se alcanza el tiempo límite

## Requisitos

### Software
- Python ≥ 3.10
- SUMO ≥ 1.8.0

### Instalación de SUMO

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-gui sumo-doc
```

#### macOS:
```bash
brew install sumo
```

#### Windows:
Descargar desde [SUMO Downloads](https://sumo.dlr.de/docs/Downloads.php)

## Instalación

1. **Clonar el repositorio:**
```bash
git clone <repository-url>
cd traffic-sim
```

2. **Instalar dependencias Python:**
```bash
pip install -r requirements.txt
```

3. **Configurar variables de entorno:**
Crear archivo `.env` en la raíz del proyecto:
```env
TRAFFIC_CONTROL_URL=http://localhost:8003
LOG_LEVEL=INFO
LOG_FILE=traffic_sim.log
```

4. **Verificar instalación de SUMO:**
```bash
sumo --version
```

## Configuración

### Archivo config.py

El archivo `config.py` contiene todas las configuraciones del sistema:

```python
# Configuración de detección de cuellos de botella
BOTTLENECK_CONFIG = {
    "density_threshold": 50.0,  # vehículos por kilómetro
    "speed_threshold": 5.0,     # metros por segundo (18 km/h)
    "queue_length_threshold": 10, # número de vehículos en cola
    "detection_interval": 30,   # segundos entre detecciones
    "min_detection_duration": 10, # segundos mínimos para confirmar
}

# Configuración de comunicación con traffic-control
TRAFFIC_CONTROL_CONFIG = {
    "base_url": "http://localhost:8003",
    "timeout": 30,              # segundos
    "retry_attempts": 3,
    "retry_delay": 5,           # segundos
}
```

## Uso

### Ejecución Básica con Archivo ZIP

```bash
# Modo headless (por defecto) - detección de cuellos de botella
python run_simulation.py simulation.zip

# Modo GUI (interfaz gráfica SUMO)
python run_simulation.py simulation.zip --gui

# Extraer a directorio específico
python run_simulation.py simulation.zip --extract-dir ./mi_simulacion

# Mantener archivos extraídos
python run_simulation.py simulation.zip --keep-files
```

### Estructura del Archivo ZIP

El archivo ZIP debe contener los siguientes archivos SUMO:
```
simulation.zip
├── edges.edg.xml
├── nodes.nod.xml
├── routes.rou.xml
├── simulation.sumocfg
└── traffic_lights.add.xml (opcional)
```

### Modos de Ejecución

#### Modo Headless (Por defecto)
- Detecta cuellos de botella automáticamente
- Envía datos a traffic-control (simulado)
- Actualiza semáforos dinámicamente
- Muestra estadísticas en consola
- **Teleporting deshabilitado** para detección realista de cuellos de botella
- Ideal para ejecuciones automatizadas y análisis de datos

#### Modo GUI
- Abre la interfaz gráfica de SUMO
- Controla el avance de la simulación desde Python
- Visualización en tiempo real del tráfico
- Debug output en consola con detección de cuellos de botella
- **Teleporting deshabilitado** para simulación realista
- Útil para debugging y análisis visual

### Ejecución con Configuración Personalizada

```python
from simulation_orchestrator import SimulationOrchestrator

# Crear orquestador con directorio personalizado
orchestrator = SimulationOrchestrator("mi_simulacion")

# Configurar y ejecutar
if orchestrator.setup_simulation():
    orchestrator.run_simulation()
```

### Monitoreo en Tiempo Real

```python
# Obtener estadísticas de la simulación
stats = orchestrator.get_simulation_stats()
print(f"Tiempo: {stats['current_time']}s")
print(f"Vehículos: {stats['vehicle_count']}")
print(f"Detecciones: {stats['bottleneck_detections']}")
```

## Formato de Datos

### Payload para traffic-control

```json
{
  "version": "1.0",
  "type": "data",
  "timestamp": "2025-01-27T15:30:00",
  "traffic_light_id": "TL-105",
  "controlled_edges": ["E1", "E2"],
  "metrics": {
    "vehicles_per_minute": 42,
    "avg_speed_kmh": 39.1,
    "avg_circulation_time_sec": 31.8,
    "density": 0.82
  },
  "vehicle_stats": {
    "motorcycle": 4,
    "car": 17,
    "bus": 0,
    "truck": 1
  }
}
```

### Respuesta de traffic-control

```json
{
  "status": "success",
  "message": "Data processed and optimized successfully",
  "optimization": {
    "green_time_sec": 45,
    "red_time_sec": 35
  },
  "impact": {
    "original_congestion": 75,
    "optimized_congestion": 45,
    "original_category": "high",
    "optimized_category": "medium"
  }
}
```

## Logs

Los logs se guardan en el directorio `logs/` con el formato:
```
logs/simulation_YYYYMMDD_HHMMSS.log
```

### Niveles de Log
- **INFO**: Información general del sistema
- **WARNING**: Advertencias no críticas
- **ERROR**: Errores que requieren atención
- **DEBUG**: Información detallada para debugging

## Troubleshooting

### Error: "SUMO no está instalado"
```bash
# Verificar instalación
which sumo
sumo --version

# Si no está instalado, seguir instrucciones de instalación
```

### Error: "No se pudo conectar con traffic-control"
```bash
# Verificar variables de entorno
echo $TRAFFIC_CONTROL_URL

# Nota: El sistema actualmente imprime las peticiones en consola
# Para conexión real, modificar services/traffic_control_client.py
```

### Error: "Error en conexión traci"
```bash
# Verificar que SUMO esté instalado correctamente
sumo --version

# Verificar que el archivo de configuración sea válido
sumo -c simulation/simulation.sumocfg --check-route

# En modo GUI, asegurarse de que no haya otra instancia ejecutándose
```

### Error: "Error generando red"
```bash
# Verificar archivos de entrada
ls -la simulation/
# Debe contener: edges.edg.xml, nodes.nod.xml, routes.rou.xml, simulation.sumocfg

# Verificar permisos
chmod +x simulation/
```

### Error: "Error en conexión traci"
```bash
# Verificar puerto disponible
netstat -tuln | grep 8813

# Cambiar puerto en configuración si es necesario
```

## Desarrollo

### Estructura de Archivos de Simulación

Los archivos en `simulation/` deben seguir el formato SUMO:

- **edges.edg.xml**: Definición de calles
- **nodes.nod.xml**: Definición de intersecciones
- **routes.rou.xml**: Rutas de vehículos
- **simulation.sumocfg**: Configuración de simulación
- **traffic_lights.add.xml**: Configuración de semáforos

### Agregar Nuevos Detectores

```python
from detectors.bottleneck_detector import BottleneckDetector

class MiDetector(BottleneckDetector):
    def detect_bottlenecks(self):
        # Implementar lógica personalizada
        pass
```

### Personalizar Controlador de Semáforos

```python
from controllers.traffic_light_controller import TrafficLightController

class MiControlador(TrafficLightController):
    def update_traffic_light(self, traffic_light_id, optimization_data):
        # Implementar lógica personalizada
        pass
```

## Contribución

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

Para preguntas o soporte, contactar al equipo de desarrollo.

---

**Nota**: Este sistema imprime en consola las peticiones HTTP que haría a traffic-control. Para una implementación completa, modifica el cliente HTTP en `services/traffic_control_client.py` para realizar las peticiones reales.

## Ejemplos de Uso Rápido

### Ejecución Básica
```bash
# Descargar un archivo ZIP de simulación y ejecutar
python run_simulation.py mi_simulacion.zip
```

### Visualización con GUI
```bash
# Ejecutar con interfaz gráfica para análisis visual
python run_simulation.py mi_simulacion.zip --gui
```

### Desarrollo y Testing
```bash
# Mantener archivos extraídos para análisis posterior
python run_simulation.py mi_simulacion.zip --keep-files --extract-dir ./debug_sim
```