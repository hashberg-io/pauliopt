from pauliopt.pauli.clifford_gates import CliffordGate
from pauliopt.pauli.pauli_gadget import PauliGadget

from pauliopt.topologies import Topology


class PauliPolynomial:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.pauli_gadgets = []

    def __irshift__(self, gadget: PauliGadget):
        if not len(gadget) == self.num_qubits:
            raise Exception(
                f"Pauli Polynomial has {self.num_qubits}, but Pauli gadget has: {len(gadget)}"
            )
        self.pauli_gadgets.append(gadget)
        return self

    def __rshift__(self, pauli_polynomial):
        for gadget in pauli_polynomial.pauli_gadgets:
            self.pauli_gadgets.append(gadget)
        return self

    def __repr__(self):
        return "\n".join(map(repr, self.pauli_gadgets))

    def __len__(self):
        return len(self.pauli_gadgets)

    def to_qiskit(self, topology=None):
        num_qubits = self.num_qubits
        if topology is None:
            topology = Topology.complete(num_qubits)
        try:
            from qiskit import QuantumCircuit
        except:
            raise Exception("Please install qiskit to export Clifford Regions")

        qc = QuantumCircuit(num_qubits)
        for gadget in self.pauli_gadgets:
            qc.compose(gadget.to_qiskit(topology), inplace=True)

        return qc

    def propagate(self, gate: CliffordGate):
        pp_ = PauliPolynomial(self.num_qubits)
        for gadget in self.pauli_gadgets:
            pp_ >>= gate.propagate_pauli(gadget)
        return pp_

    def copy(self):
        pp_ = PauliPolynomial(self.num_qubits)
        for gadget in self.pauli_gadgets:
            pp_ >>= gadget.copy()
        return pp_

    def two_qubit_count(self, topology, leg_cache=None):
        if leg_cache is None:
            leg_cache = {}
        count = 0
        for gadget in self.pauli_gadgets:
            count += gadget.two_qubit_count(topology, leg_cache=leg_cache)
        return count
