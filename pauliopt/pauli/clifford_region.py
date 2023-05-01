from qiskit import QuantumCircuit

from .clifford_gates import *
import qiskit.quantum_info as qi


class CliffordRegion:
    commutation_rule_set = {('V', 1, 0, 'H', 0, 1),
                            ('H', 0, 1, 'S', 1, 0),
                            ('S', 1, 0, 'CX', 1, 0),
                            ('CX', 0, 1, 'S', 0, 1),
                            ('H', 1, 0, 'V', 0, 1),
                            ('S', 0, 1, 'CX', 0, 1),
                            ('S', 1, 0, 'CZ', 0, 1),
                            ('S', 1, 0, 'CZ', 1, 0),
                            ('S', 0, 1, 'H', 1, 0),
                            ('H', 0, 1, 'V', 1, 0),
                            ('CX', 0, 1, 'V', 1, 0),
                            ('V', 0, 1, 'H', 1, 0),
                            ('S', 1, 0, 'H', 0, 1),
                            ('V', 1, 0, 'CX', 0, 1),
                            ('CX', 1, 0, 'V', 0, 1),
                            ('S', 0, 1, 'V', 1, 0),
                            ('H', 1, 0, 'S', 0, 1),
                            ('S', 0, 1, 'CY', 0, 1),
                            ('CX', 1, 0, 'S', 1, 0),
                            ('S', 1, 0, 'CY', 1, 0),
                            ('CZ', 1, 0, 'S', 1, 0),
                            ('V', 0, 1, 'CX', 1, 0),
                            ('S', 0, 1, 'CZ', 1, 0),
                            ('CY', 1, 0, 'S', 1, 0),
                            ('S', 1, 0, 'V', 0, 1),
                            ('V', 0, 1, 'S', 1, 0),
                            ('CY', 0, 1, 'S', 0, 1),
                            ('CZ', 0, 1, 'S', 0, 1),
                            ('S', 0, 1, 'CZ', 0, 1),
                            ('CZ', 0, 1, 'S', 1, 0),
                            ('CZ', 1, 0, 'S', 0, 1),
                            ('V', 1, 0, 'S', 0, 1)}

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
