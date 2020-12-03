"""
    This module contains code to create and simplify circuits of mixed ZX phase gadgets,
    by conjugation with topologically-aware random circuits of CX gates.
"""

from pauliopt.phase.circuits import (Z, X, PhaseGadget, PhaseCircuit, CXCircuit, CXCircuitLayer,
                                     PhaseCircuitView, CXCircuitView, CXCircuitLayerView)
from pauliopt.phase.optimizers import PhaseCircuitOptimizer
from pauliopt.utils import pi, π
