import unittest

from parameterized import parameterized
from qiskit import QuantumCircuit

from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import synthesize_tableau
from tests.tableau.utils import tableau_from_circuit
from tests.utils import verify_equality, random_hscx_circuit
from pauliopt.topologies import Topology


class TestTableauSynthesis(unittest.TestCase):
    @parameterized.expand(
        [
            ("line_5", 5, 1000, Topology.line(5), True),
            ("line_6", 6, 1000, Topology.line(6), True),
            ("line_8", 8, 1000, Topology.line(8), True),
            ("grid_4", 4, 1000, Topology.grid(2, 2), True),
            ("grid_8", 8, 1000, Topology.grid(2, 4), True),
            ("line_5", 5, 1000, Topology.line(5), False),
            ("line_8", 8, 1000, Topology.line(8), False),
            ("grid_4", 4, 1000, Topology.grid(2, 2), False),
            ("grid_8", 8, 1000, Topology.grid(2, 4), False),
        ]
    )
    def test_clifford_synthesis(self, _, n_qubits, n_gates, topo, include_swaps):
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=n_gates)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit(ct, circuit)

        qc, perm = synthesize_tableau(ct, topo, include_swaps=include_swaps)
        qc = qc.to_qiskit()

        self.assertTrue(
            verify_equality(circuit, qc),
            "The Synthesized circuit does not equal to original",
        )
