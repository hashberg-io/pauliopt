from abc import ABC, abstractmethod
from enum import Enum

from pauliopt.pauli.pauli_gadget import PauliGadget
from pauliopt.pauli.utils import X, Y, Z, I
import numpy as np


class CliffordType(Enum):
    CX = "cx"
    CY = "cy"
    CZ = "cz"
    H = "h"
    S = "s"
    V = "v"


class CliffordGate(ABC):
    def __init__(self, c_type):
        self.c_type = c_type

    @abstractmethod
    def propagate_pauli(self, gadget: PauliGadget):
        ...


class SingleQubitGate(CliffordGate, ABC):
    rules = None

    def __init__(self, type, qubit):
        super().__init__(type)
        self.qubit = qubit

    def propagate_pauli(self, gadget: PauliGadget):
        if self.rules is None:
            raise Exception(f"{self} has no rules defined for propagation!")
        p_string = gadget.paulis[self.qubit].value
        new_p, phase_change = self.rules[p_string]
        gadget.paulis[self.qubit] = new_p
        gadget.angle *= phase_change
        return gadget


class ControlGate(CliffordGate, ABC):
    rules = None

    def __init__(self, type, control, target):
        super().__init__(type)
        self.control = control
        self.target = target

    def propagate_pauli(self, gadget: PauliGadget):
        if self.rules is None:
            raise Exception(f"{self} has no rules defined for propagation!")
        pauli_size = len(gadget)
        if self.control >= pauli_size or self.target >= pauli_size:
            raise Exception(
                f"Control: {self.control} or Target {self.target} out of bounds: {pauli_size}")
        p_string = gadget.paulis[self.control].value + gadget.paulis[self.target].value
        p_c, p_t, phase_change = self.rules[p_string]
        gadget.paulis[self.control] = p_c
        gadget.paulis[self.target] = p_t
        gadget.angle *= phase_change
        return gadget


class CX(ControlGate):
    rules = {'XX': (X, I, 1),
             'XY': (Y, Z, 1),
             'XZ': (Y, Y, -1),
             'XI': (X, X, 1),
             'YX': (Y, I, 1),
             'YY': (X, Z, -1),
             'YZ': (X, Y, 1),
             'YI': (Y, X, 1),
             'ZX': (Z, X, 1),
             'ZY': (I, Y, 1),
             'ZZ': (I, Z, 1),
             'ZI': (Z, I, 1),
             'IX': (I, X, 1),
             'IY': (Z, Y, 1),
             'IZ': (Z, Z, 1),
             'II': (I, I, 1)}

    def __init__(self, control, target):
        super().__init__(CliffordType.CX, control, target)


class CZ(ControlGate):
    rules = {'XX': (Y, Y, 1),
             'XY': (Y, X, -1),
             'XZ': (X, I, 1),
             'XI': (X, Z, 1),
             'YX': (X, Y, -1),
             'YY': (X, X, 1),
             'YZ': (Y, I, 1),
             'YI': (Y, Z, 1),
             'ZX': (I, X, 1),
             'ZY': (I, Y, 1),
             'ZZ': (Z, Z, 1),
             'ZI': (Z, I, 1),
             'IX': (Z, X, 1),
             'IY': (Z, Y, 1),
             'IZ': (I, Z, 1),
             'II': (I, I, 1)}

    def __init__(self, control, target):
        super().__init__(CliffordType.CZ, control, target)


class CY(ControlGate):
    rules = {'XX': (Y, Z, -1),
             'XY': (X, I, 1),
             'XZ': (Y, X, 1),
             'XI': (X, Y, 1),
             'YX': (X, Z, 1),
             'YY': (Y, I, 1),
             'YZ': (X, X, -1),
             'YI': (Y, Y, 1),
             'ZX': (I, X, 1),
             'ZY': (Z, Y, 1),
             'ZZ': (I, Z, 1),
             'ZI': (Z, I, 1),
             'IX': (Z, X, 1),
             'IY': (I, Y, 1),
             'IZ': (Z, Z, 1),
             'II': (I, I, 1)}

    def __init__(self, control, target):
        super().__init__(CliffordType.CY, control, target)


class H(SingleQubitGate):
    rules = {'X': (Z, 1),
             'Y': (Y, -1),
             'Z': (X, 1),
             'I': (I, 1)}

    def __init__(self, qubit):
        super().__init__(CliffordType.H, qubit)


class S(SingleQubitGate):
    rules = {'X': (Y, -1),
             'Y': (X, 1),
             'Z': (Z, 1),
             'I': (I, 1)}

    def __init__(self, qubit):
        super().__init__(CliffordType.S, qubit)


class V(SingleQubitGate):
    rules = {'X': (X, 1),
             'Y': (Z, -1),
             'Z': (Y, 1),
             'I': (I, 1)}

    def __init__(self, qubit):
        super().__init__(CliffordType.V, qubit)


# For the gates X, Y, Z there won't be a change of Pauli matrices
# Refused to implement "higher order gates" like NCX, SWAP, DCX, ... but with this structure this can easily be done


def generate_random_clifford(c_type: CliffordType, n_qubits: int):
    qubit = np.random.choice(list(range(n_qubits)))
    if c_type == CliffordType.CX:
        control = np.random.choice([i for i in range(n_qubits) if i != qubit])
        return CX(control, qubit)
    elif c_type == CliffordType.CY:
        control = np.random.choice([i for i in range(n_qubits) if i != qubit])
        return CY(control, qubit)
    elif c_type == CliffordType.CZ:
        control = np.random.choice([i for i in range(n_qubits) if i != qubit])
        return CZ(control, qubit)
    elif c_type == CliffordType.H:
        return H(qubit)
    elif c_type == CliffordType.S:
        return S(qubit)
    elif c_type == CliffordType.V:
        return V(qubit)
    else:
        raise TypeError(f"Unknown Clifford Type: {c_type}")


def clifford_to_qiskit(clifford: CliffordGate):
    try:
        from qiskit import QuantumCircuit
    except:
        raise Exception("Please install qiskit to export Clifford Gates")

    if isinstance(clifford, ControlGate):
        qc = QuantumCircuit(max(clifford.control, clifford.target) + 1)
        if clifford.c_type == CliffordType.CX:
            qc.cx(clifford.control, clifford.target)
        elif clifford.c_type == CliffordType.CY:
            qc.cy(clifford.control, clifford.target)
        elif clifford.c_type == CliffordType.CZ:
            qc.cz(clifford.control, clifford.target)
        else:
            raise TypeError(f"Undefined Control gate {clifford.c_type}")
    elif isinstance(clifford, SingleQubitGate):
        qc = QuantumCircuit(clifford.qubit + 1)
        if clifford.c_type == CliffordType.H:
            qc.h(clifford.qubit)
        elif clifford.c_type == CliffordType.S:
            qc.s(clifford.qubit)
        elif clifford.c_type == CliffordType.V:
            qc.sx(clifford.qubit)
        else:
            raise TypeError(f"Undefined Single qubit gate: {clifford.c_type}")
    else:
        raise TypeError(f"Gate must be either single qubit or control")
    return qc
