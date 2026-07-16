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

        def getIDList(self):
            return list(self.programs.keys())

        def getPhase(self, tls_id):
            return self.phase.get(tls_id, 0)

        def getCompleteRedYellowGreenDefinition(self, tls_id):
            return [self.programs[tls_id]]

        def setCompleteRedYellowGreenDefinition(self, tls_id, logic):
            self.programs[tls_id] = logic
            self.applied.append((tls_id, logic))

        Phase = FakePhase
        Logic = FakeLogic

    traci.trafficlight = _TL()

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
    return traci
