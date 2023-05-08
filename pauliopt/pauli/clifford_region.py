from qiskit import QuantumCircuit

from pauliopt.pauli.clifford_gates import *


class CliffordRegion:
    def __init__(self, gates=None):
        if gates is None:
            gates = []
        self.gates: [CliffordGate] = gates
        self.num_qubits = 1

    def add_gate(self, gate: CliffordGate):
        self.num_qubits = max(self.num_qubits, gate.num_qubits)
        self.gates.append(gate)

    def add_gate_simplify(self, gate: CliffordGate):
        for idx, other in enumerate(self.gates):
            pass

    def to_qiskit(self):
        qc = QuantumCircuit(self.num_qubits)
        for gate in self.gates:
            qc.compose(gate.to_qiskit(), inplace=True)

        return qc
