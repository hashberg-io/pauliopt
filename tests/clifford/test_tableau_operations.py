import os
import unittest

import numpy as np
from parameterized import parameterized
from qiskit import QuantumCircuit, transpile

from pauliopt.circuits import Circuit
from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import synthesize_tableau
from pauliopt.topologies import Topology
from tests.clifford.utils import tableau_from_circuit_prepend, tableau_from_circuit
from tests.utils import verify_equality, random_hscx_circuit, random_clifford_circuit


class TestTableauOperations(unittest.TestCase):
    @parameterized.expand([(5,), (10,), (15,)])
    def test_inverse(self, n_qubits):
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit(ct, circuit)

        ct_ = ct.inverse()

        ct_ = ct_.inverse()

        self.assertTrue(
            np.allclose(ct.tableau, ct_.tableau),
            "The twice inverted clifford did not match",
        )

        self.assertTrue(
            np.allclose(ct.signs, ct_.signs),
            "The twice inverted clifford did not match",
        )

    @parameterized.expand([(5,), (10,), (15,)])
    def test_tableau_construction(self, n_qubits):
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit(ct, circuit)

        ct_ = CliffordTableau.from_tableau(ct.tableau, ct.signs)

        self.assertTrue(
            np.allclose(ct.tableau, ct_.tableau),
            "The twice inverted clifford did not match",
        )

        self.assertTrue(
            np.allclose(ct.signs, ct_.signs),
            "The twice inverted clifford did not match",
        )

    @parameterized.expand([(5,), (10,), (15,)])
    def test_tableau_apply(self, n_qubits):
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit(ct, circuit)

        circuit_ = transpile(
            circuit.to_qiskit().inverse(), basis_gates=["h", "s", "cx"]
        )
        circuit_ = Circuit.from_qiskit(circuit_)
        ct_ = CliffordTableau(n_qubits)
        ct_ = tableau_from_circuit(ct_, circuit_)

        ct = ct.apply(ct_)
        self.assertTrue(
            np.allclose(ct.tableau, np.eye(2 * n_qubits)),
            "The inverted clifford did not match the identity",
        )

    @parameterized.expand([(5,), (6,)])
    def test_tableau_construction_prepend(self, n_qubits):
        topo = Topology.complete(n_qubits)

        circuit = random_clifford_circuit(nr_qubits=n_qubits, nr_gates=1000)
        ct = CliffordTableau(n_qubits)
        for gate in circuit.gates:
            ct.prepend_gate(gate)

        qc, perm = synthesize_tableau(ct, topo, include_swaps=False)
        qc = qc.to_qiskit()
        self.assertTrue(
            verify_equality(circuit.to_qiskit().reverse_ops(), qc),
            "The Synthesized circuit does not equal "
            "to the original with reversed ops",
        )

    def test_string_representation(self):
        ct = CliffordTableau(5)
        ct.append_h(0)
        ct.append_s(1)
        ct.append_cnot(2, 3)
        ct.append_cnot(2, 4)
        ct.append_cnot(3, 4)

        with open(f"{os.getcwd()}/tests/data/clifford/representation.txt", "r") as f:
            self.assertEqual(
                str(ct),
                f.read(),
                "The string representation of the clifford is incorrect",
            )
