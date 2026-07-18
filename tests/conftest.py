"""Fixtures compartidas: fake de traci para testear sin SUMO.

`utils.signal_utils` y compañía hacen `import traci` a nivel de módulo, así
que el fake debe estar en sys.modules ANTES de cualquier import del código.
"""
import sys
import types
from dataclasses import dataclass, field

import pytest


@dataclass
class FakePhase:
    duration: float
    state: str
    minDur: float = -1.0
    maxDur: float = -1.0


@dataclass
class FakeLogic:
    programID: str = '0'
    type: int = 0
    currentPhaseIndex: int = 0
    phases: tuple = ()
    subParameter: dict = field(default_factory=dict)


def _make_fake_traci():
    traci = types.ModuleType('traci')

    class _TL:
        def __init__(self):
            self.programs = {}      # tls_id -> FakeLogic
            self.applied = []       # (tls_id, FakeLogic) de cada set
            self.phase = {}         # tls_id -> índice de fase actual
            self.links = {}         # tls_id -> [[(in_lane, out_lane, via)], ...]
            self.states = {}        # tls_id -> [state, ...] de setRedYellowGreenState
            self.program_id = {}    # tls_id -> program id "activo" (getProgram/setProgram)
            self.set_program_calls = []  # (tls_id, program_id) de cada setProgram

        def getIDList(self):
            return list(self.programs.keys())

        def getPhase(self, tls_id):
            return self.phase.get(tls_id, 0)

        def getProgram(self, tls_id):
            return self.program_id.get(tls_id, '0')

        def setProgram(self, tls_id, program_id):
            self.program_id[tls_id] = program_id
            self.set_program_calls.append((tls_id, program_id))

        def getControlledLinks(self, tls_id):
            # lista por índice de señal: [(in_lane, out_lane, via)]
            return self.links.get(tls_id, [])

        def getCompleteRedYellowGreenDefinition(self, tls_id):
            return [self.programs[tls_id]]

        def setCompleteRedYellowGreenDefinition(self, tls_id, logic):
            self.programs[tls_id] = logic
            self.applied.append((tls_id, logic))

        def setRedYellowGreenState(self, tls_id, state):
            self.states.setdefault(tls_id, []).append(state)

        Phase = FakePhase
        Logic = FakeLogic

    class _Lane:
        def __init__(self):
            self.lengths = {}    # lane_id -> float
            self.halting = {}    # lane_id -> int
            self.occupancy = {}  # lane_id -> float (fracción 0-1)
            # m/s; default alto (velocidad libre) porque SUMO real devuelve
            # la velocidad máxima del carril, no 0, cuando está vacío — un
            # carril no tocado por el test nunca debe leer como "parado".
            self.speeds = {}

        def getLength(self, lane_id):
            return self.lengths.get(lane_id, 7.5)

        def getLastStepHaltingNumber(self, lane_id):
            return self.halting.get(lane_id, 0)

        def getLastStepOccupancy(self, lane_id):
            return self.occupancy.get(lane_id, 0.0)

        def getLastStepMeanSpeed(self, lane_id):
            return self.speeds.get(lane_id, 13.89)

    class _Vehicle:
        def __init__(self):
            self.vehicle_speeds = {}  # vehicle_id -> float m/s
            self.order = []           # ids en orden de inserción, para getIDList determinista

        def getIDList(self):
            return list(self.order)

        def getSpeed(self, vehicle_id):
            return self.vehicle_speeds.get(vehicle_id, 13.89)

    traci.trafficlight = _TL()
    traci.lane = _Lane()
    traci.vehicle = _Vehicle()

    exceptions = types.ModuleType('traci.exceptions')

    class TraCIException(Exception):
        pass

    class FatalTraCIError(Exception):
        pass

    exceptions.TraCIException = TraCIException
    exceptions.FatalTraCIError = FatalTraCIError
    traci.exceptions = exceptions
    return traci


# Instalar un fake por defecto para que los imports de módulos que hacen
# `import traci` no fallen al colectar tests en máquinas sin SUMO.
# Los módulos del proyecto hacen `import traci` a nivel de módulo y quedan
# ligados a ESTE objeto, así que el fixture debe resetearlo, no reemplazarlo.
if 'traci' not in sys.modules:
    _default = _make_fake_traci()
    sys.modules['traci'] = _default
    sys.modules['traci.exceptions'] = _default.exceptions


@pytest.fixture
def fake_traci():
    traci = sys.modules['traci']
    traci.trafficlight.programs.clear()
    traci.trafficlight.applied.clear()
    traci.trafficlight.phase.clear()
    traci.trafficlight.links.clear()
    traci.trafficlight.states.clear()
    traci.trafficlight.program_id.clear()
    traci.trafficlight.set_program_calls.clear()
    traci.lane.lengths.clear()
    traci.lane.halting.clear()
    traci.lane.occupancy.clear()
    traci.lane.speeds.clear()
    traci.vehicle.vehicle_speeds.clear()
    traci.vehicle.order.clear()
    return traci
