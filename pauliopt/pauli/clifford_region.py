"""Helper class that displays a clifford region."""

from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.topologies import Topology


class CliffordRegion:
    def __init__(self, num_qubits, gates=None):
        if gates is None:
            gates = []
        self.gates: [CliffordGate] = gates
        self.num_qubits = num_qubits

    def compose(self, other: "CliffordRegion"):
        self.gates += other.gates
        return self

    def __add__(self, other: "CliffordRegion"):
        return self.compose(other)

    def append_gate(self, gate: CliffordGate):
        if isinstance(gate, SingleQubitClifford) and gate.qubit >= self.num_qubits:
            raise Exception(
                f"Gate with {gate.qubit} is out of bounds for Clifford Region with Qubits: {self.num_qubits}"
            )
        if (
            isinstance(gate, TwoQubitClifford)
            and gate.control >= self.num_qubits
            and gate.target >= self.num_qubits
        ):
            raise Exception(
                f"Control Gate  with {gate.control}, {gate.target} is out of bounds for Clifford Region with Qubits: {self.num_qubits}"
            )
        self.gates.append(gate)

    def prepend_gate(self, gate: CliffordGate):
        if isinstance(gate, SingleQubitClifford) and gate.qubit >= self.num_qubits:
            raise Exception(
                f"Gate with {gate.qubit} is out of bounds for Clifford Region with Qubits: {self.num_qubits}"
            )
        if (
            isinstance(gate, TwoQubitClifford)
            and gate.control >= self.num_qubits
            and gate.target >= self.num_qubits
        ):
            raise Exception(
                f"Control Gate  with {gate.control}, {gate.target} is out of bounds for Clifford Region with Qubits: {self.num_qubits}"
            )
        self.gates.insert(0, gate)

    def to_tableau(self):
        ct = CliffordTableau(self.num_qubits)
        for gate in self.gates:
            ct.append_gate(gate)
        return ct

    def to_circuit(self, method="ct_resynthesis", **kwargs):
        if method == "ct_resynthesis":
            # check if topology and include swaps is in kwargs
            if "topology" not in kwargs:
                # set complete topology into kwargs
                topo = Topology.complete(self.num_qubits)
            else:
                topo = kwargs["topology"]
            if "include_swaps" not in kwargs:
                kwargs["include_swaps"] = False
            ct = self.to_tableau()
            return ct.to_clifford_circuit_arch_aware(
                topo, include_swaps=kwargs["include_swaps"]
            )
        elif method == "ct_resynthesis_perm_row_col":
            if "topology" not in kwargs:
                # set complete topology into kwargs
                topo = Topology.complete(self.num_qubits)
            else:
                topo = kwargs["topology"]
            if "include_swaps" not in kwargs:
                kwargs["include_swaps"] = False
            ct = self.to_tableau()
            return ct.to_clifford_circuit_perm_row_col(topo, include_swaps=False)

        else:
            raise NotImplementedError(f"Method {method} not implemented")

    def to_qiskit(
        self, method="naive_apply", topology: Topology = None, include_swaps=False
    ):
        if method == "ct_resynthesis":
            ct = self.to_tableau()
            return ct.to_clifford_circuit_arch_aware(
                topology, include_swaps=include_swaps
            )
        elif method == "naive_apply":
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(self.num_qubits)
            for gate in self.gates:
                if isinstance(gate, CX):
                    qc.cx(gate.control, gate.target)
                elif isinstance(gate, CY):
                    qc.cy(gate.control, gate.target)
                elif isinstance(gate, CZ):
                    qc.cz(gate.control, gate.target)
                elif isinstance(gate, H):
                    qc.h(gate.qubit)
                elif isinstance(gate, V):
                    qc.sx(gate.qubit)
                elif isinstance(gate, S):
                    qc.s(gate.qubit)
            return qc, None
        elif method == "qiskit":
            from qiskit.synthesis import synth_clifford_full

            qc, _ = self.to_qiskit(method="naive_apply")
            ct = Clifford.from_circuit(qc)
            return synth_clifford_full(ct, method="greedy"), None
        else:
            raise NotImplementedError(f"Method {method} not implemented")
