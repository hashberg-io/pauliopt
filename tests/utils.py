import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


def verify_equality(qc_in, qc_out):
    return Operator.from_circuit(qc_in) \
        .equiv(Operator.from_circuit(qc_out))


def random_clifford_circuit(nr_gates=20, nr_qubits=4, gate_choice=None):
    qc = QuantumCircuit(nr_qubits)
    if gate_choice is None:
        gate_choice = ["CY", "CZ", "CX", "H", "S", "V"]
    for _ in range(nr_gates):
        gate_t = np.random.choice(gate_choice)
        if gate_t == "CX":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cx(control, target)
        elif gate_t == "CY":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cy(control, target)
        elif gate_t == "CZ":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cz(control, target)
        elif gate_t == "H":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.h(qubit)
        elif gate_t == "S":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.s(qubit)
        elif gate_t == "V":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.sx(qubit)
        elif gate_t == "CX":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cx(control, target)
    return qc


def random_hscx_circuit(nr_gates=20, nr_qubits=4):
    gate_choice = ["H", "S", "CX"]
    return random_clifford_circuit(nr_gates=nr_gates,
                                   nr_qubits=nr_qubits,
                                   gate_choice=gate_choice)


def apply_permutation(qc: QuantumCircuit, permutation: list):
    if len(qc.qregs) != 1:
        raise ValueError("Quantum circuit must have exactly one quantum register")
    register = qc.qregs[0]
    qc_out = QuantumCircuit(register)
    for instruction in qc:
        op_qubits = [register[permutation[register.index(q)]] for q in instruction.qubits]
        instruction.qubits = tuple(op_qubits)
        qc_out.append(instruction, instruction.qubits)
    return qc_out


################################
# Decorator to repeat a unittest
################################
def repeat(times):
    def repeatHelper(f):
        def callHelper(*args):
            for i in range(0, times):
                f(*args)

        return callHelper

    return repeatHelper
