import time

import numpy as np
import pauliopt.pauliopt

from pauliopt.clifford_.tableau_ import CliffordTableau
from qiskit import QuantumCircuit


def tableau_from_circuit(tableau, circ: QuantumCircuit):
    for op in circ:
        if op.operation.name == "h":
            tableau.append_h(op.qubits[0].index)
        elif op.operation.name == "s":
            tableau.append_s(op.qubits[0].index)
        elif op.operation.name == "cx":
            tableau.append_cnot(op.qubits[0].index, op.qubits[1].index)
        else:
            raise TypeError(
                f"Unrecongnized Gate type: {op.operation.name} for Clifford Tableaus")
    return tableau


def verify_equality(qc_in, qc_out):
    try:
        from qiskit.quantum_info import Operator
    except:
        raise Exception("Please install qiskit to compare to quantum circuits")
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
    gate_choice = ["H", "S"]
    return random_clifford_circuit(nr_gates=nr_gates,
                                   nr_qubits=nr_qubits,
                                   gate_choice=gate_choice)


if __name__ == '__main__':
    n_qubits = 10
    circ = random_hscx_circuit(1000, n_qubits)
    ct = pauliopt.pauliopt.clifford.CliffordTableau(n_qubits)
    start = time.time()
    ct = tableau_from_circuit(ct, circ)
    print(time.time() - start)

    ct_ = CliffordTableau(n_qubits)
    start = time.time()
    ct_ = tableau_from_circuit(ct_, circ)
    print(time.time() - start)

    print(np.allclose(ct.get_tableau(), ct_.tableau))
    print(np.allclose(ct.get_signs(), ct_.signs))

    # ct = pauliopt.pauliopt.clifford.CliffordTableau(2)
    #
    # ct.append_s(0)
    # #ct.append_h(0)
    #
    # print(ct.get_tableau())
    #
    # print(ct)


