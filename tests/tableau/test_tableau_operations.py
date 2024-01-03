import unittest

import numpy as np
from parameterized import parameterized
from qiskit import QuantumCircuit, transpile

from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import synthesize_tableau
from pauliopt.topologies import Topology
from tests.tableau.utils import tableau_from_circuit_prepend, tableau_from_circuit
from tests.utils import verify_equality, random_hscx_circuit


class TestTableauOperations(unittest.TestCase):

    @parameterized.expand([(5,), (10,), (15,)])
    def test_inverse(self, n_qubits):
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit(ct, circuit)

        ct_ = ct.inverse()

        ct_ = ct_.inverse()

        self.assertTrue(np.allclose(ct.tableau, ct_.tableau),
                        "The twice inverted tableau did not match")

        self.assertTrue(np.allclose(ct.signs, ct_.signs),
                        "The twice inverted tableau did not match")

    @parameterized.expand([(5,), (10,), (15,)])
    def test_tableau_construction(self, n_qubits):
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit(ct, circuit)

        ct_ = CliffordTableau.from_tableau(ct.tableau, ct.signs)

        self.assertTrue(np.allclose(ct.tableau, ct_.tableau),
                        "The twice inverted tableau did not match")

        self.assertTrue(np.allclose(ct.signs, ct_.signs),
                        "The twice inverted tableau did not match")

    @parameterized.expand([(5,), (10,), (15,)])
    def test_tableau_apply(self, n_qubits):
        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit(ct, circuit)

        circuit_ = transpile(circuit.inverse(), basis_gates=['h', 's', 'cx'])

        ct_ = CliffordTableau(n_qubits)
        ct_ = tableau_from_circuit(ct_, circuit_)

        ct = ct.apply(ct_)
        self.assertTrue(np.allclose(ct.tableau, np.eye(2 * n_qubits)),
                        "The inverted tableau did not match the identity")

    @parameterized.expand([(5,), (6,)])
    def test_tableau_construction_prepend(self, n_qubits):
        topo = Topology.complete(n_qubits)

        circuit = random_hscx_circuit(nr_qubits=n_qubits, nr_gates=1000)

        ct = CliffordTableau(n_qubits)
        ct = tableau_from_circuit_prepend(ct, circuit)

        qc, perm = synthesize_tableau(ct, topo, include_swaps=False)
        qc = qc.to_qiskit()
        self.assertTrue(verify_equality(circuit.reverse_ops(), qc),
                        "The Synthesized circuit does not equal "
                        "to the original with reversed ops")

    def test_string_representation(self):
        ct = CliffordTableau(5)
        ct.append_h(0)
        ct.append_s(1)
        ct.append_cnot(2, 3)
        ct.append_cnot(2, 4)
        ct.append_cnot(3, 4)

        with open("./data/clifford/representation.txt", "w") as f:
            f.write(str(ct))
            # self.assertEqual(str(ct), f.read(),
            #                  "The string representation of the tableau is incorrect")
