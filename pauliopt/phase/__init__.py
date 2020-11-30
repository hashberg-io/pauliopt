"""
    This module contains code to create and simplify circuits of mixed ZX phase gadgets,
    by conjugation with topologically-aware random circuits of CX gates.
"""

from pauliopt.phase.circuits import (X, Z, ZX, PhaseGadget, PhaseCircuit, CXCircuit, CXCircuitLayer,
                                     PhaseCircuitView, CXCircuitView, CXCircuitLayerView)
from pauliopt.phase.optimizers import PhaseCircuitOptimizer, CostFun, cx_count_cost_fun, cx_count
