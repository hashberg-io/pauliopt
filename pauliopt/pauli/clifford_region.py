from pauliopt.pauli.clifford_gates import *


class CliffordRegion:
    def __init__(self, num_qubits, gates=None):
        if gates is None:
            gates = []
        self.gates: [CliffordGate] = gates
        self.num_qubits = num_qubits

    def add_gate(self, gate: CliffordGate):
        if isinstance(gate, SingleQubitGate) and gate.qubit >= self.num_qubits:
            raise Exception(
                f"Gate with {gate.qubit} is out of bounds for Clifford Region with Qubits: {self.num_qubits}")
        if isinstance(gate, ControlGate) and gate.control >= self.num_qubits and gate.target >= self.num_qubits:
            raise Exception(
                f"Control Gate  with {gate.control}, {gate.target} is out of bounds for Clifford Region with Qubits: {self.num_qubits}")
        self.gates.append(gate)

    def to_qiskit(self):
        try:
            from qiskit import QuantumCircuit
        except:
            raise Exception("Please install qiskit to export Clifford Regions")
        qc = QuantumCircuit(self.num_qubits)
        for gate in self.gates:
            qc.compose(clifford_to_qiskit(gate), inplace=True)
        return qc
