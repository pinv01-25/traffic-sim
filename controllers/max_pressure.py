"""Max-Pressure (Varaiya): control local reactivo por presión de carriles.

Reemplaza el reparto proporcional (`AdaptiveSplitController`, retirado):
en vez de repartir un presupuesto de ciclo una sola vez por ciclo, decide
en cada intervalo corto qué fase servir según la presión real de cada
carril, con garantía teórica de estabilidad ante cualquier demanda
admisible. El presupuesto que manda el pipeline PSO (traffic-sync) pasa a
ser un verde máximo de *referencia* por fase, no un reparto exacto: sigue
moldeando el comportamiento en régimen fluido sin poder impedir que
Max-Pressure extienda una fase realmente saturada.
"""
import logging
import os

import traci
from config import MAX_PRESSURE_CONFIG, TRAFFIC_LIGHT_CONFIG
from utils.signal_utils import (
    _phase_link_lanes,
    derive_yellow_state,
    phases_conflict,
)

# Logger bajo el árbol "traffic_sim" (ver utils/logger.setup_logger): hereda
# handlers y nivel del logger raíz de la simulación sin configurar nada acá.
_logger = logging.getLogger("traffic_sim.max_pressure")

# ponytail: diagnóstico temporal para el deadlock de demanda_alta — activar
# con DEBUG_MAX_PRESSURE=<path> para volcar cada decisión a CSV. Sacar una
# vez cerrado el diagnóstico.
_DEBUG_LOG = os.environ.get("DEBUG_MAX_PRESSURE")


def _debug_log(row):
    if not _DEBUG_LOG:
        return
    import csv

    is_new = not os.path.exists(_DEBUG_LOG)
    with open(_DEBUG_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def lane_queue_ratio(lane_id: str) -> float:
    """Cola normalizada por capacidad del carril (~7.5m por vehículo).

    Evita el sesgo de comparar carriles cortos y largos por conteo crudo:
    un carril corto satura en falso y "parece" tan presionado como uno
    largo con demanda real muy mayor.

    Sin techo: bajo saturación de corredor completo, varias fases en
    competencia llegan a cola >= capacidad nominal a la vez; clampear a 1.0
    aplana la presión a ~0 para todas por igual justo cuando hace falta
    discriminar, y la histéresis se queda pegada en la fase activa —
    deadlock estable observado en demanda_alta (40 vehículos detenidos sin
    variar durante >2000s simulados). Dejar crecer la razón más allá de 1.0
    conserva la señal.
    """
    try:
        halting = traci.lane.getLastStepHaltingNumber(lane_id)
        length = traci.lane.getLength(lane_id)
    except Exception:
        return 0.0
    capacity = max(length / 7.5, 1.0)
    return halting / capacity


#: Peso del término de cola de salida en `compute_pressure`. >1.0 penaliza
#: más fuerte una fase que empuja hacia un carril ya congestionado aguas
#: abajo — el descuento 1:1 original solo ve un salto (el carril
#: inmediato), y el atasco real de demanda_alta ocurre varios saltos más
#: abajo, invisible a ese único término; subir el peso hace que cualquier
#: señal de congestión de salida pese más en la decisión.
DOWNSTREAM_DISCOUNT_WEIGHT = MAX_PRESSURE_CONFIG["downstream_discount_weight"]


#: Ratio de cola asumido para un carril de salida cuyo cruce vecino aguas
#: abajo está marcado "estancado" (ver `MaxPressureController._blocked_lanes`),
#: aunque la lectura local de ese carril no muestre congestión todavía. El
#: descuento de 1 salto no ve un bloqueo que ocurre 2-3 cruces más abajo; esto
#: propaga la señal de "no sirve, el vecino no drena" cruce a cruce, sin red.
COORDINATION_BLOCK_RATIO = 2.0


def occupancy_ratio(lane_id: str) -> float:
    """Ocupación física del carril (fracción 0-1 del largo cubierto por vehículos).

    A diferencia de `lane_queue_ratio` (basada en `getLastStepHaltingNumber`,
    que solo cuenta vehículos con v<0.1 m/s), esto usa
    `getLastStepOccupancy`: ve un carril de SALIDA físicamente lleno pero
    avanzando lento (halting≈0, "parece" libre) — el caso que produjo el
    deadlock de demanda_media/seed_43 (vehículos varados dentro de carriles
    internos de intersección tras recibir verde hacia una salida ya llena).

    TraCI devuelve `getLastStepOccupancy` en % (0-100) en algunas versiones
    de la API y en fracción (0-1) en otras; se normaliza defensivamente
    asumiendo porcentaje si el valor crudo supera 1.0.
    """
    try:
        raw = traci.lane.getLastStepOccupancy(lane_id)
    except Exception:
        return 0.0
    return raw / 100.0 if raw > 1.0 else raw


#: Umbral de ocupación física de un carril de salida a partir del cual se
#: considera en riesgo inminente de spillback: un carril casi lleno que
#: recibe más vehículos empujados por la fase entrante se llena del todo,
#: y los vehículos que no caben quedan varados DENTRO del cruce, bloqueando
#: el tráfico transversal — el deadlock físico e irreversible observado en
#: demanda_media/seed_43 (43 vehículos congelados en carriles internos
#: ':1437431229_1_0', ':618491563_0_0', etc., t≈1000-1500). El gate duro
#: en `compute_pressure` hace que cualquier fase que alimente un carril en
#: ese estado pierda siempre frente a una alternativa que no.
SPILLBACK_OCCUPANCY_GATE = MAX_PRESSURE_CONFIG["spillback_occupancy_gate"]


#: Velocidad media (m/s) de un carril de salida por debajo de la cual se
#: considera realmente atascado. Un carril lleno pero DRENANDO (ocupación
#: alta, velocidad de crucero) no es spillback: es tráfico normal fluyendo
#: por un tramo saturado, y penalizarlo igual que un carril parado congela
#: el servicio y colapsa la inserción de SUMO (demanda_media/42-43,
#: demanda_baja/44). OJO: SUMO devuelve la velocidad MÁXIMA del carril
#: (no 0) cuando está vacío — un carril vacío jamás debe leer como
#: atascado, pero como su occupancy_ratio también será ~0, no pasa el gate
#: de todos modos.
SPILLBACK_SPEED_THRESHOLD = MAX_PRESSURE_CONFIG["spillback_speed_threshold"]

#: Umbral más laxo para el término BLANDO (no el gate duro): por debajo de
#: esta velocidad la ocupación física entra en el `max()` de out_ratio; por
#: encima, un carril lleno pero fluyendo no penaliza la fase. El doble del
#: umbral duro es simple de razonar (escalón, no mezcla continua) y basta
#: para el efecto buscado.
SOFT_SPEED_THRESHOLD = 2 * SPILLBACK_SPEED_THRESHOLD


def lane_speed(lane_id: str) -> float:
    """Velocidad media del carril (m/s) en el último paso de simulación."""
    try:
        return traci.lane.getLastStepMeanSpeed(lane_id)
    except Exception:
        return SOFT_SPEED_THRESHOLD  # sin lectura: no asumir atasco ni flujo libre


def compute_pressure(phase_links, blocked_lanes=frozenset(), ratio_cache=None) -> float:
    """Presión de una fase: Σ(cola_entrada − w·cola_salida) sobre sus links.

    El término de salida descuenta fases que empujarían vehículos a un
    carril ya congestionado aguas abajo — el patrón que produce gridlock.
    `blocked_lanes` extiende esa señal a carriles que alimentan a un cruce
    vecino que está estancado, aunque el carril inmediato no lo muestre.

    El término de salida también combina `lane_queue_ratio` (colas reales,
    por halting) con `occupancy_ratio` (ocupación física): un carril de
    salida lleno pero fluyendo (halting≈0) no dispara la primera, pero sí
    la segunda, y de las dos se toma el máximo — ver `occupancy_ratio`.
    Si la ocupación física cruza `SPILLBACK_OCCUPANCY_GATE`, el link se
    fuerza a `COORDINATION_BLOCK_RATIO` (mismo mecanismo que usa la
    coordinación de estancamiento vecino) para que la fase pierda siempre
    contra alternativas que no alimenten un carril casi lleno.

    `ratio_cache` es opcional: si se pasa un dict, memoiza `lane_queue_ratio`
    por carril para que `_decide` pueda reusar las mismas lecturas al medir
    la ocupación local sin duplicar llamadas traci. La ocupación física se
    cachea por separado en la misma estructura, con una clave distinta por
    carril, para no pisar la lectura de `lane_queue_ratio`.
    """
    def ratio(lane_id):
        if ratio_cache is None:
            return lane_queue_ratio(lane_id)
        if lane_id not in ratio_cache:
            ratio_cache[lane_id] = lane_queue_ratio(lane_id)
        return ratio_cache[lane_id]

    def occupancy(lane_id):
        key = (lane_id, "occupancy")
        if ratio_cache is None:
            return occupancy_ratio(lane_id)
        if key not in ratio_cache:
            ratio_cache[key] = occupancy_ratio(lane_id)
        return ratio_cache[key]

    def speed(lane_id):
        key = (lane_id, "speed")
        if ratio_cache is None:
            return lane_speed(lane_id)
        if key not in ratio_cache:
            ratio_cache[key] = lane_speed(lane_id)
        return ratio_cache[key]

    total = 0.0
    for i, o in phase_links:
        out_occupancy = occupancy(o)
        out_speed = speed(o)
        # Término blando: la ocupación física solo entra en juego si el
        # carril está por debajo del umbral laxo — lleno pero fluyendo no
        # penaliza.
        soft_occupancy = out_occupancy if out_speed <= SOFT_SPEED_THRESHOLD else 0.0
        out_ratio = max(ratio(o), soft_occupancy)
        # Gate duro: solo si además está realmente parado (umbral estricto),
        # no solo lleno.
        if out_occupancy >= SPILLBACK_OCCUPANCY_GATE and out_speed <= SPILLBACK_SPEED_THRESHOLD:
            out_ratio = max(out_ratio, COORDINATION_BLOCK_RATIO)
        out_ratio = max(out_ratio, COORDINATION_BLOCK_RATIO if o in blocked_lanes else 0.0)
        total += ratio(i) - DOWNSTREAM_DISCOUNT_WEIGHT * out_ratio
    return total


class MaxPressureController:
    """Decide fase por presión, una vez por `check_interval`, por TLS."""

    def __init__(self):
        cfg = MAX_PRESSURE_CONFIG
        self._check_interval = cfg["check_interval"]
        self._hysteresis = cfg["hysteresis_factor"]
        self._starvation_bonus = cfg["starvation_bonus_per_sec"]
        self._min_green_floor = cfg["min_green_floor"]
        self._green_max_multiplier = cfg["green_max_multiplier"]
        self._all_red_time = cfg["all_red_time"]
        self._yellow_time = TRAFFIC_LIGHT_CONFIG["yellow_time"]
        self._stall_limit = cfg["stall_limit"]
        self._stall_cooldown = cfg["stall_cooldown_sec"]
        self._stall_cooldown_max = cfg["stall_cooldown_max_sec"]
        self._congestion_ratio_threshold = cfg["congestion_ratio_threshold"]
        self._congested_hysteresis_boost = cfg["congested_hysteresis_boost"]
        self._congested_min_green_boost = cfg["congested_min_green_boost"]
        self._min_green_floor_high = cfg["min_green_floor_high"]
        self._occupancy_full_scale = cfg["occupancy_full_scale"]
        # --- Supervisor de gridlock físico (ver MAX_PRESSURE_CONFIG) ---
        self._freeze_check_interval = cfg["freeze_check_interval"]
        self._freeze_trigger_sec = cfg["freeze_trigger_sec"]
        self._freeze_halt_fraction = cfg["freeze_halt_fraction"]
        self._freeze_min_vehicles = cfg["freeze_min_vehicles"]
        self._recovery_duration_sec = cfg["recovery_duration_sec"]
        self._original_program = {}  # tls_id -> program id activo antes de que MP tome control
        self._bad_since = None       # marca de tiempo desde que la fracción detenida cruzó el umbral, sin cortes
        self._next_freeze_check = 0.0
        self._recovery_until = None  # None: MP decide normalmente; float: en recuperación hasta ese tiempo
        self._tls = {}
        self._lane_owner = None  # lane_id -> (tls_id, phase_idx); lazy, ver _lane_owner_map

    def register(self, tls_id):
        self._lane_owner = None  # invalidar: hay un TLS nuevo que mapear
        try:
            self._original_program[tls_id] = traci.trafficlight.getProgram(tls_id)
        except Exception:
            pass
        defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
        if not defs:
            return
        phases = list(getattr(defs[0], "phases", ()) or ())
        phase_links = _phase_link_lanes(tls_id, phases)
        if len(phase_links) < 2:
            return  # sin decisión posible: una sola fase verde

        base_green = sum(phases[i].duration for i in phase_links) / len(phase_links)
        try:
            current_phase = int(traci.trafficlight.getPhase(tls_id))
        except Exception:
            current_phase = next(iter(phase_links))
        if current_phase not in phase_links:
            current_phase = next(iter(phase_links))

        self._tls[tls_id] = {
            "phases": phases,
            "phase_links": phase_links,
            "phase_states": {i: phases[i].state for i in phase_links},
            "current_phase": current_phase,
            "last_served": dict.fromkeys(phase_links, 0.0),
            "green_start": 0.0,
            "base_green": base_green,
            "min_green": max(base_green * self._min_green_floor, 5.0),
            "max_green": base_green * self._green_max_multiplier,
            "green_max_ref": None,  # techo de referencia del PSO (opcional)
            "next_check": 0.0,
            "transition": None,
            "stall_count": dict.fromkeys(phase_links, 0),
            "last_halting": dict.fromkeys(phase_links, None),
            "stalled_until": dict.fromkeys(phase_links, 0.0),
            "stall_strikes": dict.fromkeys(phase_links, 0),
        }

    def set_cycle_budget(self, tls_id, budget):
        """Fija el verde máximo de *referencia* por fase (techo blando)."""
        info = self._tls.get(tls_id)
        if not info or budget <= 0:
            return False
        info["green_max_ref"] = max(budget / len(info["phase_links"]), info["min_green"])
        return True

    def tick(self, current_time):
        if self._recovery_until is not None:
            if current_time >= self._recovery_until:
                self._exit_recovery(current_time)
            else:
                return 0  # en recuperación: MP no decide, SUMO cicla con el plan original

        if current_time >= self._next_freeze_check:
            self._next_freeze_check = current_time + self._freeze_check_interval
            if self._check_freeze(current_time):
                self._enter_recovery(current_time)
                return 0

        applied = 0
        for tls_id, info in self._tls.items():
            if self._advance_transition(tls_id, info, current_time):
                applied += 1
                continue
            if info["transition"] is not None:
                continue  # ya se decidió un cambio, esperando a que termine
            if current_time < info["next_check"]:
                continue
            if self._decide(tls_id, info, current_time):
                applied += 1
            else:
                info["next_check"] = current_time + self._check_interval
        return applied

    def _check_freeze(self, current_time) -> bool:
        """Detector de red camino al gridlock: fracción de vehículos
        detenidos (v<0.1 m/s) sostenida sobre `freeze_halt_fraction` durante
        `freeze_trigger_sec` seguidos. Barato: <100 vehículos típicamente,
        una llamada getSpeed por vehículo con libsumo."""
        try:
            vehicle_ids = traci.vehicle.getIDList()
        except Exception:
            return False
        n = len(vehicle_ids)
        if n < self._freeze_min_vehicles:
            self._bad_since = None
            return False
        halted = sum(1 for vid in vehicle_ids if traci.vehicle.getSpeed(vid) < 0.1)
        frac = halted / n
        if frac < self._freeze_halt_fraction:
            self._bad_since = None
            return False
        if self._bad_since is None:
            self._bad_since = current_time
            return False
        return current_time - self._bad_since >= self._freeze_trigger_sec

    def _enter_recovery(self, current_time):
        """Devuelve cada TLS registrado a su programa original de SUMO y
        suspende las decisiones de MP por `recovery_duration_sec`: el plan
        fijo original nunca se bloquea y deshace la formación temprana del
        atasco."""
        _logger.warning(
            "MaxPressure: gridlock detectado en t=%.1f, entrando en modo "
            "recuperación por %.1fs (control vuelve al plan fijo original)",
            current_time, self._recovery_duration_sec,
        )
        for tls_id in self._tls:
            program = self._original_program.get(tls_id)
            if program is None:
                continue
            try:
                traci.trafficlight.setProgram(tls_id, program)
            except Exception:
                pass
        self._recovery_until = current_time + self._recovery_duration_sec
        self._bad_since = None  # reiniciar el detector: no re-disparar con datos viejos

    def _exit_recovery(self, current_time):
        """Sale de recuperación: re-sincroniza el estado de cada TLS con la
        fase que SUMO dejó activa y limpia todo el estado de estancamiento
        (borrón y cuenta nueva) antes de devolverle el control a MP."""
        _logger.warning(
            "MaxPressure: fin de recuperación en t=%.1f, MP retoma control",
            current_time,
        )
        self._recovery_until = None
        self._bad_since = None
        self._next_freeze_check = current_time + self._freeze_check_interval
        for tls_id, info in self._tls.items():
            try:
                phase = int(traci.trafficlight.getPhase(tls_id))
            except Exception:
                phase = next(iter(info["phase_links"]))
            if phase not in info["phase_links"]:
                phase = next(iter(info["phase_links"]))
            info["current_phase"] = phase
            info["green_start"] = current_time
            info["next_check"] = current_time
            info["transition"] = None
            info["stall_count"] = dict.fromkeys(info["phase_links"], 0)
            info["stall_strikes"] = dict.fromkeys(info["phase_links"], 0)
            info["stalled_until"] = dict.fromkeys(info["phase_links"], 0.0)
            info["last_halting"] = dict.fromkeys(info["phase_links"], None)
            info["last_served"] = dict.fromkeys(info["phase_links"], current_time)

    def _advance_transition(self, tls_id, info, current_time):
        trans = info["transition"]
        if trans is None or current_time < trans["until"]:
            return False
        # Con all_red_time = 0 la etapa se salta por completo: aplicarla con
        # duración 0 igual consumiría un paso entero de simulación (1s), un
        # costo por conmutación que el plan fijo del baseline no paga.
        if trans["stage"] == "yellow" and self._all_red_time > 0:
            traci.trafficlight.setRedYellowGreenState(tls_id, trans["all_red_state"])
            trans["stage"] = "all_red"
            trans["until"] = current_time + self._all_red_time
            return True
        target = trans["target"]
        traci.trafficlight.setRedYellowGreenState(tls_id, info["phase_states"][target])
        info["current_phase"] = target
        info["green_start"] = current_time
        info["last_served"][target] = current_time
        info["next_check"] = current_time + self._check_interval
        info["transition"] = None
        self._update_stall(info, target)
        if _DEBUG_LOG:
            halts = {
                lane: traci.lane.getLastStepHaltingNumber(lane)
                for pair in info["phase_links"][target] for lane in pair
            }
            _debug_log({
                "t": current_time, "tls": tls_id, "cur": target, "elapsed": 0.0,
                "pressures": halts, "best": target, "action": "GREEN_APPLIED", "force_switch": None,
                "stall_count": info["stall_count"][target],
            })
        return True

    def _update_stall(self, info, target):
        """Si la cola de entrada de `target` no bajó respecto al último
        servicio, el bloqueo está aguas abajo — no es algo que repartir más
        verde local resuelva. Tras `stall_limit` servicios sin mejora, se
        excluye la fase; el cooldown se duplica en cada re-estancamiento
        (backoff exponencial, tope `stall_cooldown_max_sec`) porque un
        primer intento de 120s suele reabrir la fase justo cuando el
        corredor sigue lleno, retrigger inmediato y sin ganar nada — mejor
        esperar cada vez más antes de volver a probar.
        """
        halting = sum(
            traci.lane.getLastStepHaltingNumber(pair[0])
            for pair in info["phase_links"][target]
        )
        prior = info["last_halting"][target]
        if prior is not None and halting > 0 and halting >= prior:
            info["stall_count"][target] += 1
        else:
            info["stall_count"][target] = 0
            info["stall_strikes"][target] = 0  # se recuperó: resetear el backoff
        info["last_halting"][target] = halting
        if info["stall_count"][target] >= self._stall_limit:
            info["stall_strikes"][target] += 1
            cooldown = min(
                self._stall_cooldown * (2 ** (info["stall_strikes"][target] - 1)),
                self._stall_cooldown_max,
            )
            info["stalled_until"][target] = info["green_start"] + cooldown
            info["stall_count"][target] = 0

    def _effective_green_max(self, info):
        if info["green_max_ref"] is None:
            return info["max_green"]
        return min(info["green_max_ref"], info["max_green"])

    def _lane_owner_map(self):
        """lane_id (carril de entrada) -> (tls_id, phase_idx) que lo sirve.

        Construido una vez sobre todos los TLS registrados: es lo que le
        permite a un cruce ver si el carril al que empuja es la entrada de
        un vecino, sin ninguna llamada de red — están en el mismo proceso.
        """
        if self._lane_owner is None:
            self._lane_owner = {
                in_lane: (tls_id, phase_idx)
                for tls_id, info in self._tls.items()
                for phase_idx, links in info["phase_links"].items()
                for in_lane, _ in links
            }
        return self._lane_owner

    def _blocked_lanes(self, current_time):
        """Carriles de entrada cuyo cruce dueño los tiene marcados
        estancados ahora mismo — coordinación cruce a cruce sin red."""
        blocked = set()
        for lane_id, (tls_id, phase_idx) in self._lane_owner_map().items():
            if current_time < self._tls[tls_id]["stalled_until"].get(phase_idx, 0.0):
                blocked.add(lane_id)
        return blocked

    def _decide(self, tls_id, info, current_time):
        cur = info["current_phase"]
        elapsed = current_time - info["green_start"]
        force_switch = elapsed >= self._effective_green_max(info)

        # Ocupación local barata: media de lane_queue_ratio sobre los
        # carriles de entrada de todas las fases. Se cachea por carril para
        # que el cómputo de presiones más abajo reuse estas mismas lecturas
        # (mismas llamadas traci, sin duplicar).
        ratio_cache = {}
        input_lanes = {i for links in info["phase_links"].values() for i, _o in links}
        for lane in input_lanes:
            ratio_cache[lane] = lane_queue_ratio(lane)
        occupancy = sum(ratio_cache.values()) / max(len(input_lanes), 1)
        congested = occupancy >= self._congestion_ratio_threshold
        hysteresis = self._hysteresis * self._congested_hysteresis_boost if congested else self._hysteresis
        # Verde mínimo escalado por demanda, continuo: con poca ocupación un
        # piso corto conserva la adaptividad que gana en demanda asimétrica
        # (corredor); con mucha ocupación el piso sube hacia el verde del
        # plan fijo para no pagar más conmutaciones que el baseline justo
        # cuando cada segundo perdido pesa más (demanda alta simétrica). El
        # riesgo de gridlock por verdes largos lo absorbe el supervisor de
        # congelamiento (modo recuperación).
        span = max(self._min_green_floor_high - self._min_green_floor, 0.0)
        frac = min(occupancy / self._occupancy_full_scale, 1.0) if self._occupancy_full_scale > 0 else 1.0
        floor = self._min_green_floor + span * frac
        min_green = max(info["base_green"] * floor, 5.0)

        if elapsed < min_green and not force_switch:
            return False

        # La fase actual está siendo servida ahora mismo: su "tiempo sin
        # servicio" es 0, no lo que acumuló desde su último inicio de verde.
        info["last_served"][cur] = current_time
        blocked_lanes = self._blocked_lanes(current_time)
        pressures = {
            i: compute_pressure(links, blocked_lanes, ratio_cache=ratio_cache)
            + self._starvation_bonus * (current_time - info["last_served"][i])
            for i, links in info["phase_links"].items()
        }
        candidates = {
            i: p for i, p in pressures.items()
            if i == cur or not phases_conflict(info["phases"], cur, i)
        }
        not_stalled = {
            i: p for i, p in candidates.items()
            if i == cur or current_time >= info["stalled_until"].get(i, 0.0)
        }
        if set(not_stalled) - {cur}:
            # Al menos una alternativa no está estancada: usarla como pool.
            candidates = not_stalled
        elif len(candidates) > 1:
            # Todas las alternativas están estancadas. Reactivarlas todas a
            # la vez (comportamiento previo) es lo que se sospecha detrás de
            # los colapsos autoinfligidos raros observados en demanda_baja/
            # demanda_media: entra "modo drenaje" en su lugar, priorizando
            # solo la fase con mayor cola de entrada en volumen ABSOLUTO
            # (no ratio: bajo saturación total el volumen absoluto discrimina
            # mejor cuál es el corredor realmente crítico) y limpiándole el
            # estado de estancamiento para poder servirla de forma sostenida.
            drain_phase = max(
                candidates,
                key=lambda i: sum(
                    traci.lane.getLastStepHaltingNumber(pair[0])
                    for pair in info["phase_links"][i]
                ),
            )
            info["stall_count"][drain_phase] = 0
            info["stall_strikes"][drain_phase] = 0
            info["stalled_until"][drain_phase] = 0.0
            candidates = {drain_phase: candidates[drain_phase]}
        best = max(candidates, key=candidates.get)

        action = "extend"
        if best == cur:
            if not force_switch:
                _debug_log({
                    "t": current_time, "tls": tls_id, "cur": cur, "elapsed": round(elapsed, 1),
                    "pressures": pressures, "best": best, "action": "extend", "force_switch": force_switch,
                })
                return False
            others = {i: p for i, p in candidates.items() if i != cur}
            if not others:
                info["green_start"] = current_time  # sin alternativa: reinicia ventana
                _debug_log({
                    "t": current_time, "tls": tls_id, "cur": cur, "elapsed": round(elapsed, 1),
                    "pressures": pressures, "best": best, "action": "force_no_alt", "force_switch": force_switch,
                })
                return False
            best = max(others, key=others.get)
            action = "force_switch"
        elif not force_switch and pressures[best] < pressures[cur] * hysteresis:
            _debug_log({
                "t": current_time, "tls": tls_id, "cur": cur, "elapsed": round(elapsed, 1),
                "pressures": pressures, "best": best, "action": "blocked_hysteresis", "force_switch": force_switch,
            })
            return False  # histéresis: el margen no justifica el corte

        _debug_log({
            "t": current_time, "tls": tls_id, "cur": cur, "elapsed": round(elapsed, 1),
            "pressures": pressures, "best": best, "action": action, "force_switch": force_switch,
        })
        self._start_transition(tls_id, info, cur, best, current_time)
        return True

    def _start_transition(self, tls_id, info, cur, target, current_time):
        cur_state = info["phase_states"][cur]
        traci.trafficlight.setRedYellowGreenState(tls_id, derive_yellow_state(cur_state))
        info["transition"] = {
            "until": current_time + self._yellow_time,
            "target": target,
            "stage": "yellow",
            "all_red_state": "r" * len(cur_state),
        }
