import unittest

from parameterized import parameterized
from qiskit import QuantumCircuit

from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import synthesize_tableau
from tests.utils import verify_equality, random_hscx_circuit
from pauliopt.topologies import Topology


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


class TestPauliConversion(unittest.TestCase):

    @parameterized.expand([
        (5, 1000, Topology.line(5), True),
        (6, 1000, Topology.line(6), True),
        (8, 1000, Topology.line(8), True),
        (4, 1000, Topology.grid(2, 2), True),
        (8, 1000, Topology.grid(2, 4), True),

        (5, 1000, Topology.line(5), False),
        (8, 1000, Topology.line(8), False),
        (4, 1000, Topology.grid(2, 2), False),
        (8, 1000, Topology.grid(2, 4), False),
    ])
    def test_clifford_synthesis(self, n_qubits, n_gates, topo, include_swaps):
        print(topo)
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=n_gates)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit(ct, circuit)

        qc, perm = synthesize_tableau(ct, topo, include_swaps=include_swaps)
        # qc = apply_permutation(qc, perm)

        self.assertTrue(verify_equality(circuit, qc),
                        "The Synthesized circuit does not equal to original")
