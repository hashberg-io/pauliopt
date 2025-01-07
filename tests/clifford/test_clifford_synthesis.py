import itertools
import unittest

from parameterized import parameterized
from qiskit.circuit.library import Permutation

from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import (
    synthesize_tableau,
    synthesize_tableau_permutation,
    synthesize_tableau_perm_row_col,
)
from pauliopt.topologies import Topology
from tests.clifford.utils import tableau_from_circuit
from tests.utils import verify_equality, random_hscx_circuit


def enumerate_row_col_permutations(n):
    for perm in itertools.permutations(range(n)):
        yield list(zip(range(n), perm))


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
            verify_equality(circuit.to_qiskit(), qc),
            "The Synthesized circuit does not equal to original",
        )

    @parameterized.expand(
        [
            ("line_5", 5, 1000, Topology.line(5)),
            ("line_6", 6, 1000, Topology.line(6)),
            ("line_8", 8, 1000, Topology.line(8)),
            ("grid_4", 4, 1000, Topology.grid(2, 2)),
            ("grid_8", 8, 1000, Topology.grid(2, 4)),
            ("line_5", 5, 1000, Topology.line(5)),
            ("line_8", 8, 1000, Topology.line(8)),
            ("grid_4", 4, 1000, Topology.grid(2, 2)),
            ("grid_8", 8, 1000, Topology.grid(2, 4)),
        ]
    )
    def test_clifford_perm_row_col_synthesis(self, _, n_qubits, n_gates, topo):
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=n_gates)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit(ct, circuit)

        qc, perm = synthesize_tableau_perm_row_col(ct, topo)
        qc = qc.to_qiskit()

        self.assertTrue(
            verify_equality(circuit.to_qiskit(), qc),
            "The Synthesized circuit does not equal to original",
        )

    @parameterized.expand(
        [
            ("line_3", 3, 1000, Topology.line(3)),
        ]
    )
    def test_clifford_permutation_synthesis(self, _, n_qubits, n_gates, topo):
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=n_gates)
        for permutation in enumerate_row_col_permutations(n_qubits):
            ct = CliffordTableau(n_qubits)
            ct = tableau_from_circuit(ct, circuit.copy())
            qc = synthesize_tableau_permutation(ct, topo, permutation)
            qc.final_permutation = [
                source for source, target in sorted(permutation, key=lambda x: x[1])
            ]
            qc = qc.to_qiskit()

            self.assertTrue(
                verify_equality(circuit.to_qiskit(), qc),
                "The Synthesized circuit does not equal to original",
            )
