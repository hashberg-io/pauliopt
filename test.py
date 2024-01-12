from math import pi

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.random import random_circuit


def get_qubits(qubits, qreg):
    qubits_ = []
    for qubit in qubits:
        qubits_.append(qreg.index(qubit))

    return tuple(qubits_)


def main():
    qc = random_circuit(10, 4)

    qreg = qc.qregs[0]

    ci = CircuitInstruction
    for op in qc:
        for qubit in op.qubits:
            print(qubit)


if __name__ == "__main__":
    main()
